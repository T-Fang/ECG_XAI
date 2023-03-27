import os
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import init_train  # noqa: F401
from src.models.ecg_step_module import EcgEmbed, RhythmModule, BlockModule, WPWModule, STModule, QRModule, PModule, VHModule, TModule, AxisModule, EcgPipeline  # noqa: E501
from src.basic.constants import TRAIN_LOG_PATH
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import get_axis_hparams, get_block_hparams, get_common_trainer_params, get_ecg_embed_hparams, get_optim_hparams, get_p_hparams, get_pipeline_hparams, get_qr_hparams, get_rhythm_hparams, get_st_hparams, get_t_hparams, get_trainer_callbacks, get_vh_hparams, get_wpw_hparams, tune, visualize_study  # noqa: E501

# not-tuned Parameters
USE_QMC = True
N_TRIALS = 64
TIMEOUT = 86400
MAX_EPOCHS = 50
SAVE_TOP_K = 5
USE_MPAV = False
USE_LATTICE = True
SAVE_DIR = os.path.join(TRAIN_LOG_PATH, "soft_rule_lattice/")


# Define hyperparameters here
def get_hparams(trial: optuna.Trial) -> dict:
    ecg_step_hparams = {
        EcgEmbed.__name__: get_ecg_embed_hparams(trial),
        RhythmModule.__name__: get_rhythm_hparams(trial, USE_MPAV, USE_LATTICE),
        BlockModule.__name__: get_block_hparams(trial, USE_MPAV, USE_LATTICE),
        WPWModule.__name__: get_wpw_hparams(trial, USE_MPAV, USE_LATTICE),
        STModule.__name__: get_st_hparams(trial, USE_MPAV, USE_LATTICE),
        QRModule.__name__: get_qr_hparams(trial, USE_MPAV, USE_LATTICE),
        PModule.__name__: get_p_hparams(trial, USE_MPAV, USE_LATTICE),
        VHModule.__name__: get_vh_hparams(trial, USE_MPAV, USE_LATTICE),
        TModule.__name__: get_t_hparams(trial, USE_MPAV, USE_LATTICE),
        AxisModule.__name__: get_axis_hparams(trial, USE_MPAV, USE_LATTICE)
    }

    return {**get_pipeline_hparams(trial), 'optim': get_optim_hparams(trial), **ecg_step_hparams}


def objective_soft_rule_mpav(trial: optuna.Trial, datamodule: EcgDataModule, save_dir: str):
    hparams = get_hparams(trial)
    model = EcgPipeline(hparams)

    trainer = Trainer(
        callbacks=get_trainer_callbacks(trial, SAVE_TOP_K),
        logger=TensorBoardLogger(save_dir=save_dir),
        max_epochs=MAX_EPOCHS,
        # auto_scale_batch_size='binsearch',  # Use Tuner for pytorch_lightning >= 2.0
        # limit_train_batches=2,
        # limit_val_batches=2,
        **get_common_trainer_params())

    # record hyperparameters
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model, datamodule)

    # tuner.scale_batch_size(model, datamodule=datamodule, mode="binsearch")
    print('Using batch size: ', datamodule.hparams.batch_size)

    return trainer.callback_metrics["val_loss/total_loss"].item()


if __name__ == '__main__':
    study = tune(objective_soft_rule_mpav,
                 n_trials=N_TRIALS,
                 timeout=TIMEOUT,
                 save_dir=SAVE_DIR,
                 use_qmc_sampler=USE_QMC)
    visualize_study(study, SAVE_DIR, USE_LATTICE)
