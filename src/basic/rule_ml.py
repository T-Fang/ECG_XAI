# from dataclasses import dataclass
import os
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from enum import Enum
from torch.utils.data import Dataset
# from pmlayer.torch.layers import HLattice
from src.models.lattice import HL
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC, MultilabelAveragePrecision
from src.basic.dx_and_feat import Diagnosis, Feature, pad_vector


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
        The forward() method should return a loss dict,
        containing 'feat' (loss for correspondence between objective feat and feat impressions)
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
        self.is_using_hard_rule = is_using_hard_rule

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
        pass

    @property
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        """
        Contains the dict describing how to ensemble different diagnoses, e.g. {'AVB': [('BlockModule', 'AVB_imp')]}
        """
        pass


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
        self.steps: list[StepModule] = steps

    def clear_mid_output(self):
        super().clear_mid_output()
        for step in self.steps:
            step.clear_mid_output()

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
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        ensemble_dict = {}
        for step in self.steps:
            for dx, terms in step.dx_ensemble_dict.items():
                if dx not in ensemble_dict:
                    ensemble_dict[dx] = []
                ensemble_dict[dx].extend(terms)
        return ensemble_dict


# Flow Controller
# class StepController():

#     def __init__(self):
#         self.step_modules: list[StepModule] = []

# DX_TO_MODULE = {'NORM': 'NormModule', 'AVB': 'BlockModule', 'LBBB': 'BlockModule', 'RBBB': 'BlockModule'}
# EXTRA_LOSS_TO_LOG = [('BlockModule', 'feat_loss'), ('BlockModule', 'delta_loss')]

# mid_output to be aggregated
# AGG_MID_OUTPUT = [('BlockModule', 'LPR_imp'), ('BlockModule', 'LQRS_imp')]


class PipelineModule(pl.LightningModule):
    """
    The pipeline module that combines step modules.
    It contains some pre-defined methods and attributes such as metrics from torchmetrics.
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters()
        # get individual hyperparams
        self.feat_loss_weight = hparams['feat_loss_weight']
        self.delta_loss_weight = hparams['delta_loss_weight']
        self.mid_output_agg_dir: str = hparams['mid_output_agg_dir']

        # get individual optimizer hyperparams
        self.initial_lr = hparams['initial_lr']
        self.exp_lr_gamma = hparams['exp_lr_gamma']

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
            col_name = f'{module_name}_{mid_output_name}'
            self.mid_output_agg[col_name] = []

    def _init_ensemble_layers(self):
        assert len(self.dx_ensemble_dict) == len(Diagnosis)
        for dx_name, imp_names in self.dx_ensemble_dict.items():
            if len(imp_names) == 1:
                continue
            self.add_module(f'{dx_name}_ensemble', nn.Linear(len(imp_names), 1))

    def _build_pipeline(self) -> StepModule:
        """
        Build the pipeline module using the self.hparams (self.all_mid_output has already been initialized).
        """
        raise NotImplementedError

    def clear_mid_output(self):
        self.pipeline.clear_mid_output()

    # TODO: remove
    def log_metric(self, phase: str, metric_result, metric_names: list[str], on_epoch: bool):
        suffix = '_epoch' if on_epoch else '_step'
        for metric_name in metric_names:
            self.log(f"{phase}{suffix}/{metric_name}",
                     metric_result[metric_name],
                     on_step=not on_epoch,
                     on_epoch=on_epoch)

    def get_y_hat(self):
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
                dx_pred = ensemble_layer(torch.stack(imps, dim=1)).squeeze(dim=1)

            all_dx_pred.append(dx_pred)
            all_dx_name.append(dx.name)
        y_hat = torch.stack(all_dx_pred, dim=1)
        return y_hat

    def get_y_and_loss(self, batch):
        x, y = batch
        loss = self.pipeline(x)
        feat_loss, delta_loss = loss['feat'], loss['delta']

        # TODO: replace with get_y_hat()
        AVB = self.all_mid_output['BlockModule']['AVB']
        LBBB = self.all_mid_output['BlockModule']['LBBB']
        RBBB = self.all_mid_output['BlockModule']['RBBB']

        values = torch.stack([AVB, LBBB, RBBB], dim=1)
        y_hat = pad_vector(values, ['AVB', 'LBBB', 'RBBB'], Diagnosis)

        dx_loss = self.loss_fn(y_hat, y)
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
            self.mid_output_agg[dx.name].append(dx_ground_truth)
            self.mid_output_agg[f'{dx.name}_hat'].append(dx_pred)

        for feat in Feature:
            # save objective features to self.mid_output_to_agg
            obj_feat = batched_obj_feat[:, feat.value]
            self.mid_output_agg[feat.name].append(obj_feat)

        for module_name, mid_output_name in self.mid_output_to_agg:
            mid_output = self.all_mid_output[module_name][mid_output_name]
            col_name = f'{module_name}_{mid_output_name}'
            self.mid_output_agg[col_name].append(mid_output)

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
        self.add_mid_output_to_agg(y_hat, batch)

        self.val_metric.update(y_hat, y)
        self.log_dict(self.val_metric)

        self.clear_mid_output()

    def test_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, delta_loss) = self.get_y_and_loss(batch)

        self.log_loss('test', dx_loss, feat_loss, delta_loss)
        self.log_extra_loss('test')
        self.add_mid_output_to_agg(y_hat, batch)

        self.test_metric.update(y_hat, y)
        self.log_dict(self.test_metric)

        self.clear_mid_output()

    # TODO: aggregate printed output
    def agg_mid_output(self, phase: str):
        """
        Aggregate mid_output_agg into a pandas DataFrame and save to csv
        """
        for col_name, all_values in self.mid_output_agg.items():
            self.mid_output_agg[col_name] = torch.cat(all_values, dim=0)
        mid_output_agg_df = pd.DataFrame(self.mid_output_agg)
        mid_output_agg_path = os.path.join(self.mid_output_agg_dir, f'{phase}_{self.current_epoch}_mid_output_agg.csv')
        mid_output_agg_df.to_csv(mid_output_agg_path)

    def validation_epoch_end(self, outputs):
        self.agg_mid_output('val')
        self.mid_output_agg.clear()

    def test_epoch_end(self, outputs):
        self.agg_mid_output('test')
        self.mid_output_agg.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_lr_gamma)
        return [optimizer], [scheduler]


class EcgPipeline(PipelineModule):

    def _build_pipeline(self):
        raise NotImplementedError


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
        self.step_module = step_module
        self.save_to_mid_output = save_to_mid_output

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
        return F.relu(torch.sum(torch.stack(x, dim=1), dim=1) - len(x) + 1)


class Or(LogicConnect):
    # a.k.a disjunction
    def __init__(self, step_module: StepModule, at_least_k_is_true: int = 1):
        super().__init__(step_module, "")
        self.at_least_k_is_true = at_least_k_is_true

    def apply_soft_rule(self, x: list[torch.Tensor]):
        # x is the list of batched evaluation results of formulae involved in this logical connective,
        # where each result is of size N.
        return -F.relu(1 - torch.sum(torch.stack(x, dim=1), dim=1) / self.at_least_k_is_true) - 1


class Not(LogicConnect):
    # a.k.a negation
    def apply_soft_rule(self, x):
        return 1 - x


class Imply(LogicConnect):
    # a.k.a implication
    def __init__(self, step_module: StepModule, hparams: dict):
        # in Imply, we save the consequent to mid_output with key specified by ``save_to_mid_output``
        super().__init__(step_module, "")
        self.consequents = hparams['consequents']
        self.negate_consequents = hparams['negate_consequents']
        self.input_dim = hparams['input_dim']
        self.output_dims = hparams['output_dims']
        # if we use the method of modifying pre-activated values,
        # then it is assumed that the first index is the evaluation result of the antecedent
        self.use_mpav = hparams['use_mpav'] if 'use_mpav' in hparams else False
        self.lattice_inc_indices = hparams['lattice_inc_indices'] if 'lattice_inc_indices' in hparams else []
        self.lattice_sizes = hparams['lattice_sizes'] if 'lattice_sizes' in hparams else []

        self.rho = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)

        # lattice is not compatible with negated consequent
        assert not (sum(self.negate_consequents) and self.use_lattice)

        # if both lattice and MPAV methods are used,
        # then the first index is the evaluation result of the antecedent and lattice_inc_indices must be [0]
        if self.use_mpav and self.use_lattice:
            assert self.lattice_inc_indices == [0]

        # * May change to other embedding layer other than MLP
        self.decision_embed = self._mlp_embed_layers()
        # define models
        if self.use_mlp:
            self.init_mlp()
        elif self.use_lattice:
            self.init_lattice()

    @property
    def non_mono_input_dim(self):
        return self.input_dim - len(self.lattice_inc_indices) if self.use_lattice else self.input_dim - 1

    @property
    def use_mlp(self):
        return not bool(self.lattice_inc_indices)

    @property
    def use_lattice(self):
        return bool(self.lattice_inc_indices)

    @property
    def is_mpa_and_lattice(self):
        return self.use_mpav and self.lattice_inc_indices

    @property
    def n_consequents(self):
        return len(self.consequents)

    def _mlp_embed_layers(self):
        layers = []
        input_dim = self.non_mono_input_dim
        for output_dim in self.output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        return nn.Sequential(*layers)

    def init_mlp(self):
        self.add_module("mlp_output", nn.Linear(self.output_dims[-1], self.n_consequents))

    def init_lattice(self):
        sizes = torch.tensor(self.lattice_sizes, dtype=torch.int)

        # all consequents will share the embed layer and have their own output layer
        for i in range(self.n_consequents):
            self.add_module(f"l{i}", HL(self.input_dim, sizes, self.lattice_inc_indices, self.output_dims[-1]))

    def apply_soft_rule(self, x):
        # x is the batched input of size batch_size x input_dim
        xn = x[:, 1:]  # xn is the non-monotonic input of size batch_size x (input_dim - 1)
        decision_embed = self.decision_embed(xn)

        if self.use_mlp:
            mlp_output = self.mlp_output(decision_embed)

        for i, consequent in enumerate(self.consequents):
            pre_activated_modification = self.rho * x[:, 0] if self.use_mpav else 0
            if self.negate_consequents[i]:
                pre_activated_modification = -pre_activated_modification

            if self.use_mlp:
                self.mid_output[consequent] = torch.sigmoid(mlp_output[:, i] + pre_activated_modification)
            elif self.use_lattice:
                self.mid_output[consequent] = torch.sigmoid(
                    torch.squeeze(getattr(self, f"l{i}")(x, decision_embed)) + pre_activated_modification)

    def apply_hard_rule(self, x):
        # x is the batched input of size batch_size x input_dim
        for consequent in self.consequents:
            self.mid_output[consequent] = x[:, 0]


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

    def apply_soft_rule(self, term):
        # term is of shape (batch_size,)
        return torch.sigmoid(int(self.is_gt) * torch.abs(self.w) * (term - self.threshold * (1 + self.delta)))

    def apply_hard_rule(self, term):
        return (term > self.threshold).int() if self.is_gt else (term < self.threshold).int()

    @property
    def delta_loss(self):
        return self.delta * self.delta


class GT(Predicate):
    """
    A predicate that evaluates to true if the left term is greater than the right term.
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str, threshold):
        super().__init__(step_module, save_to_mid_output, threshold, is_gt=True)


class LT(Predicate):
    """
    A predicate that evaluates to true if the left term is less than the right term.
    """

    def __init__(self, step_module: StepModule, save_to_mid_output: str, threshold):
        super().__init__(step_module, save_to_mid_output, threshold, is_gt=False)
