import os

import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import init_train  # noqa: F401
from src.models.ecg_step_module import BasicCnn, BasicCnnPipeline
from src.basic.constants import TRAIN_LOG_PATH
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import set_cuda_env, flatten_dict, get_basic_cnn_hparams, get_common_trainer_params, get_optim_hparams, get_trainer_callbacks, tune, visualize_study  # noqa: E501

# not-tuned Parameters
SEED = 5

N_WORKERS = 4
USE_QMC = False
N_TRIALS = 4
TIMEOUT = 39600
MAX_EPOCHS = 20
SAVE_TOP_K = 3
USE_MPAV = False
USE_LATTICE = False
SAVE_DIR = os.path.join(TRAIN_LOG_PATH, "basic_cnn/")


# Define hyperparameters here
def get_hparams(trial: optuna.Trial) -> dict:
    pipeline_hparams = {
        'feat_loss_weight': 0,
        'delta_loss_weight': 0,
        'is_agg_mid_output': False,
        'is_using_hard_rule': False
    }

    return {**pipeline_hparams, 'optim': get_optim_hparams(trial), BasicCnn.__name__: get_basic_cnn_hparams(trial)}


def objective(trial: optuna.Trial, datamodule: EcgDataModule, save_dir: str):
    hparams = get_hparams(trial)
    model = BasicCnnPipeline(hparams)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, SAVE_TOP_K),
        logger=TensorBoardLogger(save_dir=save_dir),
        max_epochs=MAX_EPOCHS,
        # auto_scale_batch_size='power',  # Use Tuner for pytorch_lightning >= 2.0
        auto_scale_batch_size=False,
        limit_train_batches=32,
        limit_val_batches=32,
        # limit_train_batches=2,
        # limit_val_batches=2,
        **get_common_trainer_params())

    # record hyperparameters
    trainer.logger.log_hyperparams(flatten_dict(hparams))

    trainer.tune(model, datamodule=datamodule)
    # tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
    if datamodule.hparams.batch_size <= 16:
        raise torch.cuda.OutOfMemoryError("Batch size <= 16, it's likely that OOM Error has occur")
    # if len(datamodule.train_ds) % datamodule.hparams.batch_size == 1:
    #     datamodule.hparams.batch_size -= 1

    # datamodule.hparams.batch_size = 2048

    print('Using batch size: ', datamodule.hparams.batch_size)

    trainer.fit(model=model, datamodule=datamodule)

    return trainer.callback_metrics["val_metrics/auroc"].item()


if __name__ == '__main__':
    set_cuda_env(gpu_ids="0,1,2,3")
    study = tune(objective,
                 n_trials=N_TRIALS,
                 timeout=TIMEOUT,
                 save_dir=SAVE_DIR,
                 use_qmc_sampler=USE_QMC,
                 num_workers=N_WORKERS,
                 seed=SEED)
    visualize_study(study, SAVE_DIR, USE_LATTICE, False)
