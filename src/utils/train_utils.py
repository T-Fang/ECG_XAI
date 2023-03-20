import os
from pathlib import Path
import torch
import pickle
# import random
from pytorch_lightning import seed_everything
from src.utils.data_utils import EcgDataModule
from src.basic.constants import MANUAL_SEED, TRAIN_LOG_PATH

import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def seed_all():
    """
    Automatically seeds across all dataloader workers and processes for torch, numpy and stdlib random number generators.
    """
    # random.seed(MANUAL_SEED)
    # torch.manual_seed(MANUAL_SEED)
    # np.random.seed(MANUAL_SEED)
    seed_everything(MANUAL_SEED)


def set_gpu_device(gpu_number=0):
    device = torch.device('cpu')
    run_on_gpu = False
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_number}')
        run_on_gpu = True
        # torch.cuda.set_device(gpu_number)
        torch.cuda.current_device()
    print(f'Number of cuda devices: {torch.cuda.device_count()}')
    print(f'My device: {device}')
    return device, run_on_gpu


def tune(objective, n_trials=100, timeout=36000, save_dir=TRAIN_LOG_PATH):
    ecg_data_module = EcgDataModule()
    study_file_path = os.path.join(save_dir, "study.pkl")
    if os.path.isfile(study_file_path):
        study = pickle.load(open(study_file_path, "rb"))
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        sampler = TPESampler(seed=MANUAL_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, ecg_data_module, save_dir), n_trials=n_trials, timeout=timeout)

    print_best_trial(study)
    pickle.dump(study, open(study_file_path, "wb"))


def print_best_trial(study: optuna.Trial):
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Trial number: {trial.number}")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def get_trainer_callbacks(trial, save_top_k: int):
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss/total_loss")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # early_stop_callback = EarlyStopping(monitor="val_loss/total_loss", min_delta=0.000001, patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=save_top_k,
                                          monitor="val_loss/total_loss",
                                          mode="min",
                                          save_last=True,
                                          filename="epoch={epoch}-step={step}-val_loss={val_loss/total_loss:.7f}",
                                          auto_insert_metric_name=False)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "epoch={epoch}-step={step}-last"

    return [checkpoint_callback, lr_monitor, optuna_pruning]
