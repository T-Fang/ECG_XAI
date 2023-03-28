import os
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import init_train  # noqa: F401
from src.models.ecg_step_module import EcgPipeline
from src.basic.constants import TRAIN_LOG_PATH
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import flatten_dict, get_hparams, get_common_trainer_params, get_trainer_callbacks, tune, visualize_study

# not-tuned Parameters
N_WORKERS = 8
USE_QMC = False
N_TRIALS = 4
TIMEOUT = 86400
MAX_EPOCHS = 30
SAVE_TOP_K = 3
USE_MPAV = True
USE_LATTICE = False
SAVE_DIR = os.path.join(TRAIN_LOG_PATH, "soft_rule_mpav/")


def objective(trial: optuna.Trial, datamodule: EcgDataModule, save_dir: str):
    hparams = get_hparams(trial, USE_MPAV, USE_LATTICE)
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

    trainer.tune(model, datamodule=datamodule)
    print('Using batch size: ', datamodule.hparams.batch_size)
    if len(datamodule.train_ds) % datamodule.hparams.batch_size == 1:
        datamodule.hparams.batch_size -= 1

    trainer.fit(model, datamodule)

    return trainer.callback_metrics["val_metrics/auroc"].item()


if __name__ == '__main__':
    study = tune(objective,
                 n_trials=N_TRIALS,
                 timeout=TIMEOUT,
                 save_dir=SAVE_DIR,
                 use_qmc_sampler=USE_QMC,
                 num_workers=N_WORKERS)
    visualize_study(study, SAVE_DIR, USE_LATTICE)
