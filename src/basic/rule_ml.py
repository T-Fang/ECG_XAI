# from dataclasses import dataclass
import os
from pathlib import Path
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
# from pmlayer.torch.layers import HLattice
from src.models.lattice import HL
# from pmlayer.torch.hierarchical_lattice_layer import HLattice
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, MultilabelAveragePrecision
from src.basic.dx_and_feat import Diagnosis, Feature


########################################
# Data
########################################
class Signal:

    def get_data(self):
        raise NotImplementedError

    def get_feat(self):
        raise NotImplementedError

    def get_diagnoses(self):
        raise NotImplementedError

    def get_input(self):
        return [self.get_data(), self.get_feat()]

    def getitem(self):
        return (self.get_input(), self.get_diagnoses())

    def calc_feat(self):
        raise NotImplementedError


class SignalDataset(Dataset):

    def __init__(self):
        self.signals: list[Signal] = []

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index].getitem()

    def calc_feat(self):
        for signal in self.signals:
            signal.calc_feat()

    def count_diagnoses(self):
        all_diagnoses = [signal.get_diagnoses() for signal in self.signals]
        all_diagnoses = torch.stack(all_diagnoses, dim=0)
        return torch.sum(all_diagnoses, dim=0)


########################################
# Step and Pipeline
########################################


class StepModule(nn.Module):
    """
    Parent class of each processing step's module

    - Each step module should have a unique id

    - A step module should implements a forward() method
        that process the input, store the results in its mid_output dict,
        The forward() method should return a loss dict.

        For example, for ``EcgStep``s, the loss dict should contain 'feat' (loss for correspondence between objective feat and feat impressions)
        and 'delta' (regularization loss for the delta of comparison operators/predicates)
    """

    def __init__(self,
                 id: str,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 hparams: dict = {},
                 is_using_hard_rule: bool = False):
        super().__init__()
        self.id: str = id
        self.all_mid_output: dict[str, dict[str, torch.Tensor]] = all_mid_output
        self.all_mid_output[id] = {}
        self.hparams = hparams
        self.is_using_hard_rule: bool = is_using_hard_rule

    def clear_mid_output(self):
        self.all_mid_output[self.id].clear()

    def use_hard_rule(self):
        self.is_using_hard_rule = True

    def use_soft_rule(self):
        self.is_using_hard_rule = False

    @property
    def module_name(self):
        return self.__class__.__name__

    @property
    def mid_output(self):
        return self.all_mid_output[self.id]

    @property
    def extra_loss_to_log(self) -> list[tuple[str, str]]:
        pass

    @property
    def extra_terms_to_log(self) -> list[tuple[str, str]]:
        pass

    @property
    def mid_output_to_agg(self) -> list[tuple[str, str]]:
        """
        Aggregate feature impressions, which can later be saved to a csv file after each validation epoch if required
        """
        pass

    @property
    def compared_agg(self) -> list[tuple[str, str]]:
        """
        Compared aggregated mid_output, each tuple in ``compared_agg`` is a pair of names of aggregated mid_output to be compared.
        """
        pass

    @property
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        """
        Contains the dict describing how to ensemble different diagnoses, e.g. {'AVB': [('BlockModule', 'AVB_imp')]}
        """
        pass

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        """
        Add explanation to the given report file object according to the aggregated mid_output.
        """
        pass

    def get_mid_output_and_name(self, mid_output_agg: pd.Series, mid_output_imp_name: str):
        """
        Get the impression value and name of the given mid_output impression name.
        """
        mid_output_imp = mid_output_agg[f"{self.module_name}_{mid_output_imp_name}"]
        mid_output_name = mid_output_imp_name.split('_imp')[0]
        return mid_output_imp, mid_output_name

    def add_obj_feat_exp(self, mid_output_agg: pd.Series, report_file_obj, obj_feat_name: str):
        """
        Add explanation for objective feature to the given report file object according to the aggregated mid_output.
        """
        obj_feat = mid_output_agg[f"{obj_feat_name}"]
        report_file_obj.write(f"*{obj_feat_name} is* {obj_feat:.3f}\n")

    def add_comp_exp(self, mid_output_agg: pd.Series, report_file_obj, feat_imp_name: str):
        """
        Add explanation for comparison to the given report file object according to the aggregated mid_output.
        """
        feat_imp, feat_name = self.get_mid_output_and_name(mid_output_agg, feat_imp_name)
        report_file_obj.write(f"*{self.module_name}'s impression for {feat_name} is* {feat_imp:.3f}\n")

    def add_imply_exp(self, mid_output_agg: pd.Series, report_file_obj, imply_formula: str, atcd_imp_name: str,
                      consequent_imp_name: str):
        """
        Add explanation for implication to the given report file object according to the aggregated mid_output.
        """

        consequent_imp, consequent_name = self.get_mid_output_and_name(mid_output_agg, consequent_imp_name)

        exp = f"- By {imply_formula}, {self.module_name}'s impression for {consequent_name} is {consequent_imp:.3f}"
        if atcd_imp_name.endswith('_atcd'):
            atcd_imp = mid_output_agg[f"{self.module_name}_{atcd_imp_name}"]
            exp += f" and the antecedent impression is {atcd_imp:.3f}"
        exp += '\n'
        report_file_obj.write(exp)


def get_agg_col_name(module_name: str, mid_output_name: str):
    return f"{module_name}_{mid_output_name}"


class SeqSteps(StepModule):
    """
    A step module that contains a list of sequential steps
    """

    def __init__(self,
                 id: str,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 steps: list[StepModule],
                 hparams: dict,
                 is_using_hard_rule: bool = False):
        super().__init__(id, all_mid_output, hparams, is_using_hard_rule)
        self.steps = nn.ModuleList(steps)

    def clear_mid_output(self):
        super().clear_mid_output()
        for step in self.steps:
            step.clear_mid_output()

    def use_hard_rule(self):
        self.is_using_hard_rule = True
        for step in self.steps:
            step.use_hard_rule()

    def use_soft_rule(self):
        self.is_using_hard_rule = False
        for step in self.steps:
            step.use_soft_rule()

    def forward(self, x):
        all_feat_loss = []
        all_delta_loss = []
        for step in self.steps:
            loss = step(x)
            feat_loss, delta_loss = loss['feat'], loss['delta']
            if feat_loss:
                all_feat_loss.append(feat_loss)
            if delta_loss:
                all_delta_loss.append(delta_loss)
        loss = {'feat': sum(all_feat_loss), 'delta': sum(all_delta_loss)}
        self.mid_output['loss'] = loss
        return loss

    @property
    def extra_loss_to_log(self) -> list[tuple[str, str]]:
        extra_loss_to_log = []
        for step in self.steps:
            extra_loss_to_log.extend(step.extra_loss_to_log)
        return extra_loss_to_log

    @property
    def extra_terms_to_log(self) -> list[tuple[str, str]]:
        extra_term_to_log = []
        for step in self.steps:
            extra_term_to_log.extend(step.extra_terms_to_log)
        return extra_term_to_log

    @property
    def mid_output_to_agg(self) -> list[tuple[str, str]]:
        mid_output_to_agg = []
        for step in self.steps:
            mid_output_to_agg.extend(step.mid_output_to_agg)
        return mid_output_to_agg

    @property
    def compared_agg(self) -> list[tuple[str, str]]:
        compared_agg = []
        for step in self.steps:
            compared_agg.extend(step.compared_agg)
        return compared_agg

    @property
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        ensemble_dict = {}
        for step in self.steps:
            for dx, terms in step.dx_ensemble_dict.items():
                if dx not in ensemble_dict:
                    ensemble_dict[dx] = []
                ensemble_dict[dx].extend(terms)
        return ensemble_dict

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        for step in self.steps:
            step.add_explanation(mid_output_agg, report_file_obj)


RHO = 8


class PipelineModule(pl.LightningModule):
    """
    The pipeline module that combines step modules.
    It contains some pre-defined methods and attributes such as metrics from torchmetrics.
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters()  # hyperparameters are saved as self.hparams.hparams
        # get individual hyperparams
        self.feat_loss_weight = hparams['feat_loss_weight']
        self.delta_loss_weight = hparams['delta_loss_weight']
        self.is_agg_mid_output: bool = hparams['is_agg_mid_output'] if 'is_agg_mid_output' in hparams else True
        self.is_using_hard_rule: bool = hparams['is_using_hard_rule'] if 'is_using_hard_rule' in hparams else False

        # get individual optimizer hyperparams
        self.lr = hparams['optim']['lr']
        self.beta1 = hparams['optim']['beta1']
        self.beta2 = hparams['optim']['beta2']
        self.eps = hparams['optim']['eps']
        self.exp_lr_gamma = hparams['optim']['exp_lr_gamma']

        # * May be changed
        self.loss_fn = nn.BCELoss()
        self.metric_average: str = 'macro'

        # build pipeline
        self.all_mid_output: dict[str, dict[str, torch.Tensor]] = {}
        self.pipeline: StepModule = self._build_pipeline()

        # init logging, mid_output to be aggregated, and ensemble layers
        self.extra_loss_to_log: list[tuple[str, str]] = self.pipeline.extra_loss_to_log
        self.extra_terms_to_log: list[tuple[str, str]] = self.pipeline.extra_terms_to_log
        self.mid_output_to_agg: list[tuple[str, str]] = self.pipeline.mid_output_to_agg
        self._init_mid_output_agg()
        self.compared_agg: list[tuple[str, str]] = self.pipeline.compared_agg
        self.dx_ensemble_dict: dict[str, list[tuple[str, str]]] = self.pipeline.dx_ensemble_dict
        self._init_ensemble_layers()

        self._init_metrics()

    def _init_metrics(self):
        self.train_metric = MetricCollection({
            'train_metrics/acc': MultilabelAccuracy(num_labels=len(Diagnosis), average=self.metric_average),
        })
        self.val_metric = MetricCollection({
            'val_metrics/acc': MultilabelAccuracy(num_labels=len(Diagnosis), average=self.metric_average),
            'val_metrics/auprc': MultilabelAveragePrecision(num_labels=len(Diagnosis), average=self.metric_average),
            'val_metrics/auroc': MultilabelAUROC(num_labels=len(Diagnosis), average=self.metric_average)
        })
        self.test_metric = MetricCollection({
            'test_metrics/acc': MultilabelAccuracy(num_labels=len(Diagnosis), average=self.metric_average),
            'test_metrics/auprc': MultilabelAveragePrecision(num_labels=len(Diagnosis), average=self.metric_average),
            'test_metrics/auroc': MultilabelAUROC(num_labels=len(Diagnosis), average=self.metric_average)
        })

    def _init_mid_output_agg(self):
        self.mid_output_agg = {}
        for dx in Diagnosis:
            self.mid_output_agg[dx.name] = []
            self.mid_output_agg[f'{dx.name}_hat'] = []

        for feat in Feature:
            self.mid_output_agg[feat.name] = []

        for module_name, mid_output_name in self.mid_output_to_agg:
            self.mid_output_agg[get_agg_col_name(module_name, mid_output_name)] = []

    def _init_ensemble_layers(self):
        # assert len(self.dx_ensemble_dict) == len(Diagnosis)
        for dx_name, imp_names in self.dx_ensemble_dict.items():
            if len(imp_names) == 1:
                continue
            self.add_module(f'{dx_name}_ensemble', nn.Linear(len(imp_names), 1))

    def _build_pipeline(self) -> StepModule:
        """
        Build the pipeline module using the self.hparams.hparams (self.all_mid_output has already been initialized).
        """
        raise NotImplementedError

    def clear_mid_output(self):
        self.pipeline.clear_mid_output()

    def use_hard_rule(self):
        self.pipeline.use_hard_rule()

    def use_soft_rule(self):
        self.pipeline.use_soft_rule()

    def forward(self, batched_ecg: torch.Tensor, batched_obj_feat: torch.Tensor):
        self.pipeline((batched_ecg, batched_obj_feat))

        all_dx_pred = []
        all_dx_name = []
        for dx in Diagnosis:
            imp_names = self.dx_ensemble_dict[dx.name]
            if len(imp_names) == 1:
                module_name, imp_name = imp_names[0]
                dx_pred = self.all_mid_output[module_name][imp_name]
            else:
                ensemble_layer = getattr(self, f'{dx.name}_ensemble')
                imps = [self.all_mid_output[module_name][imp_name] for module_name, imp_name in imp_names]
                dx_pred = torch.sigmoid(ensemble_layer(torch.stack(imps, dim=1)).squeeze())

            all_dx_pred.append(dx_pred)
            all_dx_name.append(dx.name)
        y_hat = torch.stack(all_dx_pred, dim=1)
        return y_hat

    def get_y_and_loss(self, batch):
        (batched_ecg, batched_obj_feat), y = batch
        y_hat = self(batched_ecg, batched_obj_feat)
        loss = self.pipeline.mid_output['loss']
        feat_loss, delta_loss = loss['feat'], loss['delta']

        with torch.cuda.amp.autocast(enabled=False):
            if self.is_using_hard_rule:
                y_hat = torch.clamp(y_hat, 0, 1)
            dx_loss = self.loss_fn(y_hat.float(), y.float())
        y = y.int()
        return (y_hat, y), (dx_loss, feat_loss, delta_loss)

    def log_loss(self, phase: str, dx_loss, feat_loss, delta_loss):
        self.log(f"{phase}_loss/dx_loss", dx_loss)
        self.log(f"{phase}_loss/feat_loss", feat_loss)
        self.log(f"{phase}_loss/delta_loss", delta_loss)
        total_loss = dx_loss + self.feat_loss_weight * feat_loss + self.delta_loss_weight * delta_loss
        self.log(f"{phase}_loss/total_loss", total_loss)
        return total_loss

    def log_extra_loss(self, phase: str):
        for module_name, loss_name in self.extra_loss_to_log:
            loss = self.all_mid_output[module_name]['loss'][loss_name]
            self.log(f"{phase}_extra_loss/{module_name}/{loss_name}", loss)

    def log_extra_terms(self, phase: str):
        for module_name, term_name in self.extra_terms_to_log:
            term = self.all_mid_output[module_name][term_name]
            self.log(f"{phase}_extra_terms/{module_name}/{term_name}", term)

    def add_mid_output_to_agg(self, y_hat: torch.Tensor, batch):
        """
        Add the mid output to be aggregated to the self.mid_output_agg
        """
        x, y = batch
        batched_ecg, batched_obj_feat = x

        for dx in Diagnosis:
            # save y and y_hat to self.mid_output_to_agg
            dx_ground_truth = y[:, dx.value]
            dx_pred = y_hat[:, dx.value]
            if dx.name not in self.mid_output_agg:
                self.mid_output_agg[dx.name] = []
                self.mid_output_agg[f'{dx.name}_hat'] = []
            self.mid_output_agg[dx.name].append(dx_ground_truth)
            self.mid_output_agg[f'{dx.name}_hat'].append(dx_pred)

        for feat in Feature:
            # save objective features to self.mid_output_to_agg
            obj_feat = batched_obj_feat[:, feat.value]
            if feat.name not in self.mid_output_agg:
                self.mid_output_agg[feat.name] = []
            self.mid_output_agg[feat.name].append(obj_feat)

        for module_name, mid_output_name in self.mid_output_to_agg:
            mid_output = self.all_mid_output[module_name][mid_output_name]
            col_name = get_agg_col_name(module_name, mid_output_name)
            if col_name not in self.mid_output_agg:
                self.mid_output_agg[col_name] = []
            self.mid_output_agg[col_name].append(mid_output)

    @property
    def example_input(self):
        eg_batched_ecg = torch.rand([32, 12, 5000], dtype=torch.float32)
        eg_batched_obj_feat = torch.rand([32, len(Feature)], dtype=torch.float32)
        batched_input = (eg_batched_ecg, eg_batched_obj_feat)
        return batched_input

    @property
    def tb_logger(self) -> TensorBoardLogger:
        return self.logger.experiment

    def on_fit_start(self):
        # self.tb_logger.add_graph(model=self, input_to_model=self.example_input)
        if self.trainer.datamodule:
            self.tb_logger.add_text('batch_size', str(self.trainer.datamodule.hparams.batch_size), 0)
        self.tb_logger.add_text("Imply's rho: ", str(RHO), 0)

    def training_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, delta_loss) = self.get_y_and_loss(batch)

        total_loss = self.log_loss('train', dx_loss, feat_loss, delta_loss)
        self.log_extra_loss('train')
        self.log_extra_terms('train')

        self.train_metric(y_hat, y)
        self.log_dict(self.train_metric)

        self.clear_mid_output()
        return total_loss

    def validation_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, delta_loss) = self.get_y_and_loss(batch)

        self.log_loss('val', dx_loss, feat_loss, delta_loss)
        self.log_extra_loss('val')
        if self.is_agg_mid_output and self.current_epoch == self.trainer.max_epochs - 1:
            self.add_mid_output_to_agg(y_hat, batch)

        self.val_metric.update(y_hat, y)
        self.log_dict(self.val_metric)

        self.clear_mid_output()

    def test_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, delta_loss) = self.get_y_and_loss(batch)

        self.log_loss('test', dx_loss, feat_loss, delta_loss)
        self.log_extra_loss('test')
        if self.is_agg_mid_output:
            self.add_mid_output_to_agg(y_hat, batch)

        self.test_metric.update(y_hat, y)
        self.log_dict(self.test_metric)

        self.clear_mid_output()

    @property
    def mid_output_agg_dir(self) -> str:
        return self.logger.log_dir

    @property
    def fig_dir(self):
        fig_d = os.path.join(self.mid_output_agg_dir, 'figures')
        Path(fig_d).mkdir(parents=True, exist_ok=True)
        return fig_d

    def agg_mid_output(self, phase: str):
        """
        Aggregate mid_output_agg into a pandas DataFrame and save to csv
        """
        for col_name, all_values in self.mid_output_agg.items():
            self.mid_output_agg[col_name] = torch.cat(all_values, dim=0).float().cpu()
        mid_output_agg_df = pd.DataFrame(self.mid_output_agg)
        mid_output_agg_path = os.path.join(self.mid_output_agg_dir,
                                           f'{phase}_epoch_{self.current_epoch}_mid_output_agg.csv')
        mid_output_agg_df.to_csv(mid_output_agg_path)

    def compare_agg_via_scatter(self, phase: str):
        """
        Compare the aggregated mid output via scatter plot
        """
        fig, ax = plt.subplots()
        for xlabel, ylabel in self.compared_agg:
            x = self.mid_output_agg[xlabel]
            y = self.mid_output_agg[ylabel]
            corr = torch.corrcoef(torch.stack([x, y], dim=0))[0, 1]
            ax.set_title(f'{ylabel} vs {xlabel} (r={corr:.4f})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.scatter(x, y)
            self.tb_logger.add_figure(f'{phase}_fig/{ylabel}_vs_{xlabel}', fig)
            fig.savefig(os.path.join(self.fig_dir, f'{phase}_{ylabel}_vs_{xlabel}.png'))
            ax.clear()
        plt.close(fig)

    def on_validation_epoch_end(self):
        if self.is_agg_mid_output and self.current_epoch == self.trainer.max_epochs - 1:
            self.agg_mid_output('val')
            self.compare_agg_via_scatter('val')
            self.mid_output_agg.clear()

    def on_test_epoch_end(self):
        if self.is_agg_mid_output:
            self.agg_mid_output('test')
            self.compare_agg_via_scatter('test')
            self.mid_output_agg.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=[self.beta1, self.beta2], eps=self.eps)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_lr_gamma)
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # need to remove optimizer_idx if migrated to pytorch-lightning 2.0
        self.log_dict(grad_norm(self, norm_type=2))
        # pass

    def generate_report(self, path_to_agg: str, patient_idx: int = 0):
        """
        Generate a report for the model

        Args:
            mid_output_agg_file_path: path to the aggregated mid output csv file
            report_file_path: path to the report file
        """
        all_mid_output_agg_df = pd.read_csv(path_to_agg, header=0, index_col=0)
        mid_output_agg: pd.Series = all_mid_output_agg_df.loc[patient_idx]
        report_file_path = os.path.splitext(path_to_agg)[0] + f'_report_for_patient_at_{patient_idx}.md'
        with open(report_file_path, 'w') as f:
            f.write('# Diagnosis Report\n')
            self.pipeline.add_explanation(mid_output_agg, f)


########################################
# Rules
########################################
class Rule(nn.Module):
    """
    Parent class of all rules such as formula rules (i.e., logical rules) and shape rules
    """
    pass


class Formula(Rule):
    """
    Parent class of predicate and logical connectives

    A formula takes in multiple ground terms and evaluates to a boolean/probability value.
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str = ''):
        super().__init__()
        self.step_module_container = [step_module]
        self.save_to_mid_output = save_to_mid_output

    @property
    def step_module(self):
        return self.step_module_container[0]

    @property
    def mid_output(self):
        return self.step_module.mid_output

    @property
    def is_using_hard_rule(self):
        return self.step_module.is_using_hard_rule

    @property
    def is_saving(self):
        return self.save_to_mid_output != ''

    def apply_soft_rule(self, x):
        raise NotImplementedError

    def apply_hard_rule(self, x):
        return self.apply_soft_rule(x)

    def forward(self, x):
        if self.is_saving and self.save_to_mid_output in self.mid_output:
            return self.mid_output[self.save_to_mid_output]
        output = self.apply_hard_rule(x) if self.is_using_hard_rule else self.apply_soft_rule(x)
        if self.is_saving:
            self.mid_output[self.save_to_mid_output] = output
        return output


class LogicConnect(Formula):
    """
    Parent class of all (soft) logical connectives
    """
    pass


class And(LogicConnect):
    # a.k.a conjunction
    def apply_soft_rule(self, x: list[torch.Tensor]):
        # x is the list of batched evaluation results of formulae involved in this logical connective,
        # where each result is of size N.
        return torch.clamp(torch.sum(torch.stack(x, dim=1), dim=1) - len(x) + 1.0, min=0)


class Or(LogicConnect):
    # a.k.a disjunction
    def __init__(self, step_module: StepModule, at_least_k_is_true: int = 1):
        super().__init__(step_module, "")
        self.at_least_k_is_true = at_least_k_is_true

    def apply_soft_rule(self, x: list[torch.Tensor]):
        # x is the list of batched evaluation results of formulae involved in this logical connective,
        # where each result is of size N.
        return torch.clamp(torch.sum(torch.stack(x, dim=1), dim=1) / self.at_least_k_is_true, max=1)


class Not(LogicConnect):
    # a.k.a negation
    def apply_soft_rule(self, x):
        return 1 - x


def inverse_sigmoid(t: torch.Tensor) -> torch.Tensor:
    EPS = 1e-7
    return torch.log((t + EPS) / (1 - t + EPS))


class Imply(LogicConnect):
    # a.k.a implication
    def __init__(self, step_module: StepModule, hparams: dict):
        # in Imply, we save the consequent to mid_output with key specified by ``save_to_mid_output``
        super().__init__(step_module, "")
        self.antecedent: str = hparams['antecedent']
        self.negate_atcd: bool = hparams['negate_atcd']
        self.consequents: list[str] = hparams['consequents']
        self.negate_consequents: list[bool] = hparams['negate_consequents']
        self.input_dim: int = hparams['input_dim']
        self.output_dims: list[int] = hparams['output_dims']
        # if we use the method of modifying pre-activated values,
        # then it is assumed that the first index is the evaluation result of the antecedent
        self.use_mpav: bool = hparams['use_mpav'] if 'use_mpav' in hparams else False
        self.lattice_sizes: list[int] = hparams['lattice_sizes'] if 'lattice_sizes' in hparams else []
        if 'lattice_inc_indices' in hparams:
            self.lattice_inc_indices: list[int] = hparams['lattice_inc_indices']
        elif self.lattice_sizes:
            self.lattice_inc_indices: list[int] = [0]
        else:
            self.lattice_inc_indices: list[int] = []

        # self.rho = nn.Parameter(torch.tensor(RHO, dtype=torch.float32), requires_grad=True)
        self.rho = RHO

        # lattice is not compatible with negated consequent
        assert not (sum(self.negate_consequents) and self.use_lattice)

        # if both lattice and MPAV methods are used,
        # then the first index is the evaluation result of the antecedent and lattice_inc_indices must be [0]
        if self.use_mpav and self.use_lattice:
            assert self.lattice_inc_indices == [0]

        # define models
        if self.use_mlp:
            self.init_mlp()
        elif self.use_lattice:
            self.init_lattice()

    def cat_atcd_with(self, focused_embed: torch.Tensor) -> torch.Tensor:
        """
        Concatenate desired mid_output with the focused embed, and negate the mid_output if needed.
        """
        atcd = torch.unsqueeze(self.mid_output[self.antecedent], 1)
        if self.negate_atcd:
            atcd = self.step_module.NOT(atcd)
        # if self.use_lattice:
        #     # atcd = inverse_sigmoid(atcd)
        #     atcd = atcd * LATTICE_MULTIPLIER
        return torch.cat((atcd, focused_embed), dim=1)

    @property
    def non_mono_input_dim(self):
        return self.input_dim - len(self.lattice_sizes) if self.use_lattice else self.input_dim - 1

    @property
    def use_mlp(self):
        return not bool(self.lattice_sizes)

    @property
    def use_lattice(self):
        return bool(self.lattice_sizes)

    @property
    def is_mpa_and_lattice(self):
        return self.use_mpav and self.lattice_sizes

    @property
    def n_consequents(self):
        return len(self.consequents)

    def init_mlp(self):
        self.add_module("mlp_output", nn.Linear(self.output_dims[-1], self.n_consequents))

    def init_lattice(self):
        sizes = torch.tensor(self.lattice_sizes, dtype=torch.long)
        if torch.cuda.is_available():
            sizes = sizes.to(torch.device('cuda:0'))

        # For Imply without lattice, all consequents will share the embed layer and have their own output layer
        for i in range(self.n_consequents):
            self.add_module(f"l{i}", HL(self.input_dim, sizes, self.lattice_inc_indices))

    def apply_soft_rule(self, x):
        focused_embed, decision_embed = x
        # decision_embed (of size batch_size x (output_dims[-1])) is results after passing non-monotonic input through an MLP
        x = self.cat_atcd_with(focused_embed)
        # x is the batched input of size batch_size x input_dim

        if self.use_mlp:
            mlp_out = getattr(self, 'mlp_output')(decision_embed)

        for i, consequent in enumerate(self.consequents):
            pre_activated_modification = self.rho * x[:, 0] if self.use_mpav else 0
            if self.negate_consequents[i]:
                pre_activated_modification = -pre_activated_modification

            if self.use_mlp:
                self.mid_output[consequent] = torch.sigmoid(mlp_out[:, i] + pre_activated_modification)
            elif self.use_lattice:
                self.mid_output[consequent] = torch.sigmoid(
                    torch.squeeze(getattr(self, f"l{i}")(x[:, [0]]), dim=1) + pre_activated_modification)

    def apply_hard_rule(self, x):
        for consequent in self.consequents:
            self.mid_output[consequent] = self.mid_output[self.antecedent]


class Predicate(Formula):
    """
    Parent class of all predicates
    """
    pass


class ComparisonOp(Predicate):
    """
    Parent class of all comparison operators
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str, threshold, is_gt: bool):
        super().__init__(step_module, save_to_mid_output)
        self.threshold = threshold
        self.is_gt = is_gt

        self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.delta = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    @property
    def sign(self) -> int:
        return 1 if self.is_gt else -1

    def apply_soft_rule(self, term):
        # term is of shape (batch_size,)
        return torch.sigmoid(self.sign * torch.abs(self.w) * (term - self.threshold * (1 + self.delta)))

    def apply_hard_rule(self, term):
        thresh_tensor = torch.tensor(self.threshold, device=term.device)
        return torch.gt(term, thresh_tensor).float() if self.is_gt else torch.lt(term, thresh_tensor).float()

    @property
    def delta_loss(self) -> torch.Tensor:
        return self.delta * self.delta


class GT(ComparisonOp):
    """
    A predicate that evaluates to true if the left term is greater than the right term.
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str, threshold):
        super().__init__(step_module, save_to_mid_output, threshold, is_gt=True)


class LT(ComparisonOp):
    """
    A predicate that evaluates to true if the left term is less than the right term.
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str, threshold):
        super().__init__(step_module, save_to_mid_output, threshold, is_gt=False)
