import os
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import init_train  # noqa: F401
from src.models.ecg_step_module import EcgPipeline
from src.basic.constants import TRAIN_LOG_PATH
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import flatten_dict, get_dummy_hparams, get_common_trainer_params, get_trainer_callbacks, set_cuda_env, tune, visualize_study

# not-tuned Parameters
SEED = 66

N_WORKERS = 4
USE_QMC = False
N_TRIALS = 1
TIMEOUT = 39600
MAX_EPOCHS = 20
SAVE_TOP_K = 3
USE_MPAV = False
USE_LATTICE = False
BATCH_SIZE = 256
SAVE_DIR = os.path.join(TRAIN_LOG_PATH, "hard_rule/")


def objective(trial: optuna.Trial, datamodule: EcgDataModule, save_dir: str):
    hparams = get_dummy_hparams()
    hparams['is_using_hard_rule'] = True
    model = EcgPipeline(hparams)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, SAVE_TOP_K),
        logger=TensorBoardLogger(save_dir=save_dir),
        max_epochs=MAX_EPOCHS,
        auto_scale_batch_size='power',
        # limit_train_batches=2,
        # limit_val_batches=2,
        **get_common_trainer_params())

    # record hyperparameters
    trainer.logger.log_hyperparams(flatten_dict(hparams))

    # trainer.tune(model, datamodule=datamodule)
    # if datamodule.hparams.batch_size <= 16:
    #     raise torch.cuda.OutOfMemoryError("Batch size <= 16, it's likely that OOM Error has occur")
    # if len(datamodule.train_ds) % datamodule.hparams.batch_size == 1:
    #     datamodule.hparams.batch_size -= 1

    datamodule.hparams.batch_size = BATCH_SIZE
    print('Using batch size: ', datamodule.hparams.batch_size)

    trainer.fit(model, datamodule)

    return trainer.callback_metrics["val_metrics/auroc"].item()


if __name__ == '__main__':
    set_cuda_env(gpu_ids='6')
    study = tune(objective,
                 n_trials=N_TRIALS,
                 timeout=TIMEOUT,
                 save_dir=SAVE_DIR,
                 use_qmc_sampler=USE_QMC,
                 num_workers=N_WORKERS,
                 seed=SEED)
    visualize_study(study, SAVE_DIR, USE_LATTICE)
