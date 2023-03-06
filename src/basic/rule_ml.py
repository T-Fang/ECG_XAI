from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from enum import Enum

# import pytorch_lightning as pl
# from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


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
# Logic
########################################
class LogicConnect(Enum):
    """
    The available Logical Connectives
    """
    AND = 1  # conjunction
    OR = 2  # disjunction
    NOT = 3  # negation
    IMPLY = 4  # implication


class Formula:
    pass


class Predicate:
    """
    Parent class of all predicates

    A predicate takes in multiple terms and evaluates to a boolean/probability value.
    """

    def eval(self, *args):
        raise NotImplementedError


class ID(Predicate):
    """
    A predicate that acts as a wrapper and simply return the given boolean/probability value.
    """

    def eval(self, value):
        return value


class GE(Predicate):
    """
    A predicate that evaluates to true if the left term is greater than or equal to the right term.
    """

    def eval(self, value, threshold, weight, delta):
        prob = torch.sigmoid(torch.tensor(weight * (value - (threshold + delta))))
        return prob


@dataclass
class RuleDesc:
    """
    The description of a rule
    """
    # Logical connective is a type of logical symbols and predicate is a type of non-logical symbols
    symbol: LogicConnect | Predicate
    terms: list
    save_to_mid_output: str = ''  # when save_to_mid_output is not empty, the result of this rule will be saved to mid_output


RuleDesc(LogicConnect.IMPLY, [RuleDesc(GE, ['QRS', 120], 'LQRS'), RuleDesc(LogicConnect.AND, ['LBBB', 'RBBB'])])


########################################
# Step and Pipeline
########################################
class EnsembleMethod(Enum):
    AVG = 1
    FIRST = 1
    LAST = 2


def default_ensemble_methods():
    return [EnsembleMethod.AVG]


class StepModule(nn.Module):
    """
    Parent class of each processing step's module
    """

    def forward(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Process the batch, which is a list containing the batched signal (embedding) and batched mid_output
        """
        raise NotImplementedError


@dataclass
class StepDesc:
    """
    The description of a step
    """
    step_module: StepModule
    rules: list[RuleDesc]


class SeqSteps(StepModule):
    pass


class ParallelSteps(StepModule):
    pass


# class PipelineModule(pl.LightningModule):
#     """
#     The pipeline module that combines step modules.
#     It contains some pre-defined methods and attributes such as metrics from torchmetrics.
#     """

#     def __init__(self, loss_fn=torch.nn.CrossEntropyLoss()):
#         super().__init__()
#         self.loss_fn = loss_fn
#         self.metric_average_method = "macro" if NUM_CLASSES > 2 else "none"
#         self.train_metric = MetricCollection({
#             'acc': Accuracy(num_classes=NUM_CLASSES, average=self.metric_average_method),
#             'f1': F1Score(num_classes=NUM_CLASSES, average=self.metric_average_method)
#         })
#         self.val_metric = MetricCollection({
#             'acc': Accuracy(num_classes=NUM_CLASSES, average=self.metric_average_method),
#             'precision': Precision(num_classes=NUM_CLASSES, average=None),
#             'recall': Recall(num_classes=NUM_CLASSES, average=None),
#             'f1': F1Score(num_classes=NUM_CLASSES, average=self.metric_average_method)
#         })
#         self.test_metric = MetricCollection({
#             'acc': Accuracy(num_classes=NUM_CLASSES, average=self.metric_average_method),
#             'precision': Precision(num_classes=NUM_CLASSES, average=None),
#             'recall': Recall(num_classes=NUM_CLASSES, average=None),
#             'f1': F1Score(num_classes=NUM_CLASSES, average=self.metric_average_method)
#         })

#     def log_train_metric(self, metric_result, on_epoch: bool):
#         suffix = '_epoch' if on_epoch else '_step'
#         train_acc, train_f1 = metric_result['acc'], metric_result['f1']

#         if NUM_CLASSES > 2:
#             self.log(f"train{suffix}/acc", train_acc)
#             self.log(f"train{suffix}/f1", train_f1)
#         else:
#             self.log(f"train{suffix}/acc", train_acc[GOOD_LABEL])
#             self.log(f"train{suffix}/f1", train_f1[GOOD_LABEL])

#     def log_val_metric(self, metric_result, on_epoch: bool):
#         suffix = '_epoch' if on_epoch else '_step'
#         val_acc, val_f1, val_precision, val_recall = metric_result['acc'], metric_result['f1'], \
#                                                     metric_result['precision'], metric_result['recall']

#         self.log(f"val{suffix}/precision/GOOD", val_precision[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)
#         self.log(f"val{suffix}/recall/GOOD", val_recall[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)

#         if NUM_CLASSES > 2:
#             self.log(f"val{suffix}/acc", val_acc, on_step=not on_epoch, on_epoch=on_epoch)
#             self.log(f"val{suffix}/f1", val_f1, on_step=not on_epoch, on_epoch=on_epoch)
#         else:
#             self.log(f"val{suffix}/acc", val_acc[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)
#             self.log(f"val{suffix}/f1", val_f1[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)

#     def log_test_metric(self, metric_result, on_epoch: bool):
#         suffix = '_epoch' if on_epoch else '_step'
#         test_acc, test_f1, test_precision, test_recall = metric_result['acc'], metric_result['f1'], \
#                                                         metric_result['precision'], metric_result['recall']
#         test_precision_GOOD, test_recall_GOOD = test_precision[GOOD_LABEL], test_recall[GOOD_LABEL]
#         self.log(f"test{suffix}/precision/GOOD", test_precision_GOOD, on_step=not on_epoch, on_epoch=on_epoch)
#         self.log(f"test{suffix}/recall/GOOD", test_recall_GOOD, on_step=not on_epoch, on_epoch=on_epoch)

#         if NUM_CLASSES > 2:
#             self.log(f"test{suffix}/acc", test_acc, on_step=not on_epoch, on_epoch=on_epoch)
#             self.log(f"test{suffix}/f1", test_f1, on_step=not on_epoch, on_epoch=on_epoch)
#         else:
#             self.log(f"test{suffix}/acc", test_acc[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)
#             self.log(f"test{suffix}/f1", test_f1[GOOD_LABEL], on_step=not on_epoch, on_epoch=on_epoch)

#     def get_y_hat_and_y(self, batch):
#         x, y = batch
#         y_hat = self(x)
#         return y_hat, y

#     def training_step(self, batch, batch_idx):
#         y_hat, y = self.get_y_hat_and_y(batch)

#         loss = self.loss_fn(y_hat, y)
#         self.log("train_step/loss", loss, prog_bar=True)

#         train_metric = self.train_metric(y_hat, y)
#         self.log_train_metric(train_metric, False)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         y_hat, y = self.get_y_hat_and_y(batch)

#         self.val_metric.update(y_hat, y)

#         val_loss = self.loss_fn(y_hat, y)
#         self.log("val_step/loss", val_loss, prog_bar=False, on_step=True, on_epoch=False)

#         # val_metric = self.val_metric(y_hat, y)
#         # self.log_val_metric(val_metric, False)

#     def test_step(self, batch, batch_idx):
#         y_hat, y = self.get_y_hat_and_y(batch)

#         self.test_metric.update(y_hat, y)
#         # test_metric = self.test_metric(y_hat, y)
#         # self.log_test_metric(test_metric, False)

#     def training_epoch_end(self, outputs):
#         train_metric = self.train_metric.compute()
#         self.log_train_metric(train_metric, True)
#         self.train_metric.reset()

#     def validation_epoch_end(self, outputs):
#         val_metric = self.val_metric.compute()
#         self.log_val_metric(val_metric, True)
#         self.val_metric.reset()

#     def test_epoch_end(self, outputs):
#         test_metric = self.test_metric.compute()
#         self.log_test_metric(test_metric, True)
#         self.test_metric.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
