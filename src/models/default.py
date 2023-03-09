import optuna
import torch
from pytorch_lightning import Trainer, loggers as pl_loggers
from src.models.ecg_step_module import EcgEmbed, BlockModule
from src.basic.constants import LOG_INTERVAL
from src.basic.rule_ml import SeqSteps, PipelineModule
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import get_trainer_callbacks


def get_default_pipeline():
    pipeline_module = PipelineModule()
    all_mid_output = pipeline_module.all_mid_output
    seq2 = SeqSteps('seq2', all_mid_output, [EcgEmbed(all_mid_output), BlockModule(all_mid_output)])
    pipeline_module.add_pipeline(seq2)
    return pipeline_module


def objective_default(trial: optuna.Trial, data_module: EcgDataModule, save_dir: str):
    # training params
    max_epochs = 50
    save_top_k = 5

    # Define hyperparameters here

    # model with suggested hyperparameters
    pipeline_module = get_default_pipeline()
    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, save_top_k),
        #   limit_train_batches=4,
        #   limit_val_batches=4,
        #   limit_test_batches=4,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None,
        logger=pl_loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=LOG_INTERVAL,
        enable_progress_bar=False)

    # record hyperparameters
    # hyperparams = dict(num_of_naive_net_layers=num_of_naive_net_layers, naive_net_output_dims=naive_net_output_dims)
    # trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(pipeline_module, data_module)

    return trainer.callback_metrics["val_epoch/total_loss"].item()
