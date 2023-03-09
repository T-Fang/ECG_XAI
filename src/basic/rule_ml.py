# from dataclasses import dataclass
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from enum import Enum
from torch.utils.data import Dataset
from pmlayer.torch.layers import HLattice
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from src.basic.dx_and_feat import Diagnosis, pad_vector
from src.basic.constants import FEAT_LOSS_WEIGHT, REG_LOSS_WEIGHT


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

    def __init__(self, mid_output: dict, save_to_mid_output: str):
        super().__init__()
        self.mid_output = mid_output
        self.save_to_mid_output = save_to_mid_output

    @property
    def is_saving(self):
        return self.save_to_mid_output != ''

    def get_output(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self.is_saving and self.save_to_mid_output in self.mid_output:
            return self.mid_output[self.save_to_mid_output]
        output = self.get_output(x)
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
    def get_output(self, x: list[torch.Tensor]):
        # * x is the list of batched evaluation results of formulae involved in this logical connective,
        # * where each result is of size N.
        return F.relu(torch.sum(torch.stack(x, dim=1), dim=1) - len(x) + 1)


class Or(LogicConnect):
    # a.k.a disjunction
    def get_output(self, x: list[torch.Tensor]):
        # x is the list of batched evaluation results of formulae involved in this logical connective,
        # where each result is of size N.
        return -F.relu(1 - torch.sum(torch.stack(x, dim=1), dim=1)) - 1


class Not(LogicConnect):
    # a.k.a negation
    def get_output(self, x):
        return 1 - x


RHO = 0.5


class Imply(LogicConnect):
    # a.k.a implication
    def __init__(self,
                 mid_output: dict,
                 consequents: list[str],
                 negate_consequents: list[bool],
                 input_dim: int,
                 output_dims: list[int],
                 use_mpa: bool = False,
                 lattice_inc_indices: list[int] = [],
                 lattice_sizes: list[int] = []):
        # in Imply, we save the consequent to mid_output with key specified by ``save_to_mid_output``
        super().__init__(mid_output, "")
        self.consequents = consequents
        self.negate_consequents = negate_consequents
        self.input_dim = input_dim
        self.output_dims = output_dims
        # if we use the method of modifying pre-activated values,
        # then it is assumed that the first index is the evaluation result of the antecedent
        self.use_mpa = use_mpa
        self.lattice_inc_indices = lattice_inc_indices
        self.lattice_sizes = lattice_sizes

        # lattice is not compatible with negated consequent
        assert not (sum(self.negate_consequents) and self.use_lattice)

        # if both lattice and MPA methods are used,
        # then the first index is the evaluation result of the antecedent and lattice_inc_indices must be [0]
        if self.use_mpa and self.use_lattice:
            assert self.lattice_inc_indices == [0]

        # define models
        if self.use_mlp:
            self.init_mlp()
        elif self.use_lattice:
            self.init_lattice()

    @property
    def use_mlp(self):
        return not bool(self.lattice_inc_indices)

    @property
    def use_lattice(self):
        return bool(self.lattice_inc_indices)

    @property
    def is_mpa_and_lattice(self):
        return self.use_mpa and self.lattice_inc_indices

    @property
    def n_consequents(self):
        return len(self.consequents)

    def _mlp_embed_layers(self):
        layers = []
        input_dim = self.input_dim
        for output_dim in self.output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        return nn.Sequential(*layers)

    def init_mlp(self):
        layers = [self._mlp_embed_layers()]
        layers.append(nn.Linear(self.output_dims[-1], self.n_consequents))
        self.add_module("mlp", nn.Sequential(*layers))

    def init_lattice(self):
        sizes = torch.tensor(self.lattice_sizes, dtype=torch.int)

        # all consequents will share the embed layer and have their own output layer
        # * May change to other embedding layer other than MLP
        self.embed_layer = self._mlp_embed_layers()
        for i in range(self.n_consequents):
            output_layer = nn.Linear(self.output_dims[-1], 1)
            consequent_model = nn.Sequential(self.embed_layer, output_layer)
            self.add_module(f"l{i}", HLattice(self.input_dim, sizes, self.lattice_inc_indices, consequent_model))

    def get_output(self, x):
        # x is the batched input of size Nxinput_dim, where N is the batch size
        if self.use_mlp:
            mlp_output = self.mlp(x)
        for i, consequent in enumerate(self.consequents):
            pre_activated_modification = RHO * x[:, 0] if self.use_mpa else 0
            if self.negate_consequents[i]:
                pre_activated_modification = -pre_activated_modification

            if self.use_mlp:
                self.mid_output[consequent] = torch.sigmoid(mlp_output[:, i] + pre_activated_modification)
            elif self.use_lattice:
                # TODO: can be further optimized
                self.mid_output[consequent] = torch.sigmoid(getattr(self, f"l{i}")(x) + pre_activated_modification)


class Predicate(Formula):
    """
    Parent class of all predicates
    """
    pass


# class ID(Predicate):
#     # TODO: Remove
#     """
#     A predicate that acts as a wrapper for the given boolean/probability variable(s)
#       and simply return the given variable(s).
#     """

#     def __init__(self, variables):
#         super().__init__({}, '')
#         self.variables = variables

#     def forward(self):
#         return self.variables


class GE(Predicate):
    """
    A predicate that evaluates to true if the left term is greater than or equal to the right term.
    """

    def __init__(self, mid_output: dict, save_to_mid_output: str, threshold):
        super().__init__(mid_output, save_to_mid_output)
        self.threshold = threshold

        self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.delta = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def get_output(self, term):
        # term is of size N
        return torch.sigmoid(self.w * (term - (self.threshold + self.delta)))

    @property
    def reg_loss(self):
        return self.delta * self.delta


# @dataclass
# class MidOutputItem:
#     """
#     The description of a mid output item
#     """
#     step_modules_id: str
#     mid_output_key: str

# class RuleDesc:
#     pass

# @dataclass
# class FormulaRuleDesc(RuleDesc):
#     """
#     The description of a rule
#     """
#     formula: Formula
#     terms: list[MidOutputItem]
#     # when save_to_mid_output is not empty,
#     # the result of this rule will be saved to the step module's mid_output dict with the given key
#     save_to_mid_output: str = ''

# RuleDesc(LogicConnect.IMPLY, [RuleDesc(GE, ['QRS', 120], 'LQRS'), RuleDesc(LogicConnect.AND, ['LBBB', 'RBBB'])])

########################################
# Step and Pipeline
########################################
# class EnsembleMethod(Enum):
#     AVG = 1
#     FIRST = 1
#     LAST = 2

# def default_ensemble_methods():
#     return [EnsembleMethod.AVG]


class StepModule(nn.Module):
    """
    Parent class of each processing step's module

    - Each step module should have a unique id

    - A step module should implements a forward() method
        that process the input, store the results in its mid_output dict,
        The forward() method should return a feat_loss (loss for correspondence between objective feat and feat impressions)
        and reg_loss (regularization loss, mainly come from delta of comparison operators/predicates)
    """

    def __init__(self, id: str, all_mid_output: dict[str, dict[str, torch.Tensor]]):
        super().__init__()
        self.id: str = id
        self.all_mid_output: dict[str, dict[str, torch.Tensor]] = all_mid_output
        self.all_mid_output[id] = {}

    def clear_mid_output(self):
        self.all_mid_output[self.id].clear()

    @property
    def mid_output(self):
        return self.all_mid_output[self.id]


# @dataclass
# class StepDesc:
#     """
#     The description of a step
#     """
#     step_module: StepModule
#     rules: list[RuleDesc]


class SeqSteps(StepModule):
    """
    A step module that contains a list of sequential steps
    """

    def __init__(self, id: str, all_mid_output: dict[str, dict[str, torch.Tensor]], steps: list[StepModule]):
        super().__init__(id, all_mid_output)
        self.steps: list[StepModule] = steps

    def clear_mid_output(self):
        super().clear_mid_output()
        for step in self.steps:
            step.clear_mid_output()

    def forward(self, x):
        all_feat_loss = []
        all_reg_loss = []
        for step in self.steps:
            feat_loss, reg_loss = step(x)
            if feat_loss:
                all_feat_loss.append(feat_loss)
            if reg_loss:
                all_reg_loss.append(reg_loss)
        return sum(all_feat_loss), sum(all_reg_loss)


class ParallelSteps(StepModule):
    """
    A step module that contains a list of parallel steps
    """

    def __init__(self, id: str, all_mid_output: dict[str, dict[str, torch.Tensor]], steps: list[StepModule]):
        super().__init__(id, all_mid_output)
        self.steps: list[StepModule] = steps

    def clear_mid_output(self):
        super().clear_mid_output()
        for step in self.steps:
            step.clear_mid_output()

    def forward(self, x):
        # TODO: implement parallelism
        all_feat_loss = []
        all_reg_loss = []
        for step in self.steps:
            feat_loss, reg_loss = step(x)
            if feat_loss:
                all_feat_loss.append(feat_loss)
            if reg_loss:
                all_reg_loss.append(reg_loss)
        return sum(all_feat_loss), sum(all_reg_loss)


# Flow Controller
# class StepController():

#     def __init__(self):
#         self.step_modules: list[StepModule] = []


class PipelineModule(pl.LightningModule):
    """
    The pipeline module that combines step modules.
    It contains some pre-defined methods and attributes such as metrics from torchmetrics.
    """

    def __init__(self, pipeline: StepModule):
        super().__init__()
        # * May be tuned
        self.loss_fn = nn.BCELoss()
        self.metric_average = 'macro'
        self.pipeline = pipeline

        self.all_mid_output: dict[str, dict[str, torch.Tensor]] = {}
        self.train_metric = MetricCollection({
            'acc': Accuracy(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'f1': F1Score(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average)
        })
        self.val_metric = MetricCollection({
            'acc': Accuracy(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'precision': Precision(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'recall': Recall(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'f1': F1Score(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average)
        })
        self.test_metric = MetricCollection({
            'acc': Accuracy(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'precision': Precision(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'recall': Recall(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average),
            'f1': F1Score(task='multilabel', num_classes=len(Diagnosis), average=self.metric_average)
        })

    def clear_mid_output(self):
        self.pipeline.clear_mid_output()

    def log_train_metric(self, metric_result, on_epoch: bool):
        suffix = '_epoch' if on_epoch else '_step'
        train_acc, train_f1 = metric_result['acc'], metric_result['f1']

        self.log(f"train{suffix}/acc", train_acc)
        self.log(f"train{suffix}/f1", train_f1)

    def log_val_metric(self, metric_result, on_epoch: bool):
        suffix = '_epoch' if on_epoch else '_step'
        val_acc, val_f1, val_precision, val_recall = metric_result['acc'], metric_result['f1'], metric_result[
            'precision'], metric_result['recall']

        self.log(f"val{suffix}/precision", val_precision, on_step=not on_epoch, on_epoch=on_epoch)
        self.log(f"val{suffix}/recall", val_recall, on_step=not on_epoch, on_epoch=on_epoch)

        self.log(f"val{suffix}/acc", val_acc, on_step=not on_epoch, on_epoch=on_epoch)
        self.log(f"val{suffix}/f1", val_f1, on_step=not on_epoch, on_epoch=on_epoch)

    def log_test_metric(self, metric_result, on_epoch: bool):
        suffix = '_epoch' if on_epoch else '_step'
        test_acc, test_f1, test_precision, test_recall = metric_result['acc'], metric_result['f1'], metric_result[
            'precision'], metric_result['recall']

        self.log(f"test{suffix}/precision/GOOD", test_precision, on_step=not on_epoch, on_epoch=on_epoch)
        self.log(f"test{suffix}/recall/GOOD", test_recall, on_step=not on_epoch, on_epoch=on_epoch)

        self.log(f"test{suffix}/acc", test_acc, on_step=not on_epoch, on_epoch=on_epoch)
        self.log(f"test{suffix}/f1", test_f1, on_step=not on_epoch, on_epoch=on_epoch)

    def get_y_and_loss(self, batch):
        x, y = batch
        feat_loss, reg_loss = self.pipeline(x)
        AVB = self.all_mid_outputs['BlockModule']['AVB']
        LBBB = self.all_mid_outputs['BlockModule']['LBBB']
        RBBB = self.all_mid_outputs['BlockModule']['RBBB']

        values = torch.stack([AVB, LBBB, RBBB], dim=1)
        y_hat = pad_vector(values, ['AVB', 'LBBB', 'RBBB'], Diagnosis)

        dx_loss = self.loss_fn(y_hat, y)
        return (y_hat, y), (dx_loss, feat_loss, reg_loss)

    def training_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, reg_loss) = self.get_y_and_loss(batch)

        self.log("train_step/dx_loss", dx_loss, prog_bar=False)
        self.log("train_step/feat_loss", feat_loss, prog_bar=False)
        self.log("train_step/reg_loss", reg_loss, prog_bar=False)

        train_metric = self.train_metric(y_hat, y)
        self.log_train_metric(train_metric, False)

        loss = dx_loss + FEAT_LOSS_WEIGHT * feat_loss + REG_LOSS_WEIGHT * reg_loss
        return loss

    def validation_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, reg_loss) = self.get_y_and_loss(batch)

        self.val_metric.update(y_hat, y)

        val_loss = self.loss_fn(y_hat, y)
        self.log("val_step/loss", val_loss, prog_bar=False, on_step=True, on_epoch=False)

        # val_metric = self.val_metric(y_hat, y)
        # self.log_val_metric(val_metric, False)

    def test_step(self, batch, batch_idx):
        (y_hat, y), (dx_loss, feat_loss, reg_loss) = self.get_y_and_loss(batch)

        self.test_metric.update(y_hat, y)
        # test_metric = self.test_metric(y_hat, y)
        # self.log_test_metric(test_metric, False)

    def training_epoch_end(self, outputs):
        self.clear_mid_output()

        train_metric = self.train_metric.compute()
        self.log_train_metric(train_metric, True)
        self.train_metric.reset()

    def validation_epoch_end(self, outputs):
        self.clear_mid_output()

        val_metric = self.val_metric.compute()
        self.log_val_metric(val_metric, True)
        self.val_metric.reset()

    def test_epoch_end(self, outputs):
        self.clear_mid_output()

        test_metric = self.test_metric.compute()
        self.log_test_metric(test_metric, True)
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [scheduler]
