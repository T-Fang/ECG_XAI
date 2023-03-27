import os
from pathlib import Path
import torch
import pickle
from pytorch_lightning import seed_everything
from src.utils.data_utils import EcgDataModule
from src.basic.constants import CHECK_VAL_EVERY_N_EPOCH, LOG_INTERVAL, MANUAL_SEED, TRAIN_LOG_PATH

import optuna
from optuna.samplers import TPESampler, QMCSampler
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import plot_slice, plot_param_importances, plot_optimization_history
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor


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


def calc_output_shape(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the shape of the output from a convolutional/maxpooling layer
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def tune(objective, n_trials=100, timeout=36000, save_dir=TRAIN_LOG_PATH, use_qmc_sampler=False):
    ecg_datamodule = EcgDataModule(batch_size=512)
    study_file_path = os.path.join(save_dir, "study.pkl")
    if os.path.isfile(study_file_path):
        study = pickle.load(open(study_file_path, "rb"))
    else:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # For QMC sampler, better to set n_trials to power of 2
        sampler = QMCSampler(scramble=True, seed=MANUAL_SEED) if use_qmc_sampler else TPESampler(seed=MANUAL_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(lambda trial: objective(trial, ecg_datamodule, save_dir), n_trials=n_trials, timeout=timeout)

    print_best_trial(study)
    pickle.dump(study, open(study_file_path, "wb"))
    return study


def visualize_study(study, save_dir: str, use_lattice: bool, use_rule: bool = True):
    hparams_to_check = [
        'Embed_n_conv_layers', 'Embed_n_fc_layers', 'Embed_conv_kernel_size', 'Embed_conv_stride',
        'Embed_pool_kernel_size', 'Embed_pool_stride'
    ]
    if use_rule:
        all_imply_module_name = ['Rhythm', 'Block', 'WPW', 'ST', 'QR', 'P', 'VH', 'T', 'Axis']
        for module_name in all_imply_module_name:
            hparams_to_check.append(f'{module_name}_Imply_n_fc_layers')
            if use_lattice:
                hparams_to_check.append(f'{module_name}_Imply_lattice_size')

    # these Figure are from plotly.graph_objects
    slice_plot = plot_slice(study, params=hparams_to_check)
    slice_plot.write_image(os.path.join(save_dir, "hparams_slice_plot.png"))
    importance_plot = plot_param_importances(study, params=hparams_to_check)
    importance_plot.write_image(os.path.join(save_dir, "hparams_importance_plot.png"))
    history_plot = plot_optimization_history(study)
    history_plot.write_image(os.path.join(save_dir, "hparams_optimization_history.png"))


def print_best_trial(study: optuna.Trial):
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Trial number: {trial.number}")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def get_common_trainer_params() -> dict:
    return {
        'accelerator': 'auto',
        'precision': 'bf16',
        'log_every_n_steps': LOG_INTERVAL,
        'check_val_every_n_epoch': CHECK_VAL_EVERY_N_EPOCH,
        'enable_progress_bar': False,
        'profiler': 'simple'
    }


def get_trainer_callbacks(trial, save_top_k: int):
    optuna_pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss/total_loss")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    device_monitor = DeviceStatsMonitor(cpu_stats=True)
    # early_stop_callback = EarlyStopping(monitor="val_loss/total_loss", min_delta=0.000001, patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=save_top_k,
                                          monitor="val_loss/total_loss",
                                          mode="min",
                                          save_last=True,
                                          filename="epoch={epoch}-step={step}-val_loss={val_loss/total_loss:.7f}",
                                          auto_insert_metric_name=False)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "epoch={epoch}-step={step}-last"

    return [checkpoint_callback, lr_monitor, optuna_pruning, device_monitor]


####################################
# Hyperparameter tuning
####################################


def get_dummy_hparams() -> dict:
    USE_MPAV = True
    USE_LATTICE = False
    optim_hparams = {'lr': 1e-3, 'beta1': 0.9, 'eps': 1e-8, 'beta2': 0.999, 'exp_lr_gamma': 0.98}
    pipeline_hparams = {
        'feat_loss_weight': 1,
        'delta_loss_weight': 1,
        'is_agg_mid_output': True,
        'is_using_hard_rule': False
    }
    ecg_step_hparams = {
        'EcgEmbed': get_dummy_ecg_embed_hparams(),
        'RhythmModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'BlockModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'WPWModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'STModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'QRModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'PModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'VHModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'TModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
        'AxisModule': {
            'Imply': get_dummy_imply_hparams(USE_MPAV, USE_LATTICE)
        },
    }
    return {**pipeline_hparams, 'optim': optim_hparams, **ecg_step_hparams}


def get_optim_hparams(trial: optuna.Trial) -> dict:
    return {
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'beta1': trial.suggest_float('beta1', 0.5, 0.99),
        'eps': trial.suggest_float('eps', 1e-8, 1e-4, log=True),
        'beta2': trial.suggest_float('beta2', 0.8, 0.99),
        'exp_lr_gamma': trial.suggest_float('exp_lr_gamma', 0.1, 1)
    }


def get_pipeline_hparams(trial: optuna.Trial) -> dict:
    return {
        'feat_loss_weight': trial.suggest_float('feat_loss_weight', 1e-2, 1e2, log=True),
        'delta_loss_weight': trial.suggest_float('delta_loss_weight', 1e-2, 1e4, log=True),
        'is_agg_mid_output': True,
        'is_using_hard_rule': False
    }


def get_basic_cnn_hparams(trial: optuna.Trial) -> dict:
    embed_n_conv_layers = trial.suggest_int('Embed_n_conv_layers', 1, 5)
    embed_conv_out_channels = [
        trial.suggest_int(f"Embed_conv_out_ch_l{i}", 4, 256, log=True) for i in range(embed_n_conv_layers)
    ]

    embed_conv_kernel_size = trial.suggest_int('Embed_conv_kernel_size', 1, 12)
    embed_conv_stride = trial.suggest_int('Embed_conv_stride', 1, 2)
    embed_pool_kernel_size = trial.suggest_int('Embed_pool_kernel_size', 1, 2)
    embed_pool_stride = trial.suggest_int('Embed_pool_stride', 1, embed_pool_kernel_size)

    embed_n_fc_layers = trial.suggest_int('Embed_n_fc_layers', 1, 5)
    embed_fc_out_dims = [
        trial.suggest_int(f"Embed_fc_out_dim_l{i}", 4, 256, log=True) for i in range(embed_n_fc_layers)
    ]

    return {
        'conv_out_channels': embed_conv_out_channels,
        'fc_out_dims': embed_fc_out_dims,
        'conv_kernel_size': embed_conv_kernel_size,
        'conv_stride': embed_conv_stride,
        'pool_kernel_size': embed_pool_kernel_size,
        'pool_stride': embed_pool_stride
    }


def get_dummy_ecg_embed_hparams() -> dict:
    return {
        'conv_out_channels': [2],
        'fc_out_dims': [32, 16],
        'conv_kernel_size': 1000,
        'conv_stride': 500,
        'pool_kernel_size': 2,
        'pool_stride': 2
    }


def get_ecg_embed_hparams(trial: optuna.Trial) -> dict:
    embed_n_conv_layers = trial.suggest_int('Embed_n_conv_layers', 1, 5)
    embed_conv_out_channels = [
        trial.suggest_int(f"Embed_conv_out_ch_l{i}", 4, 256, log=True) for i in range(embed_n_conv_layers)
    ]
    embed_conv_kernel_size = trial.suggest_int('Embed_conv_kernel_size', 1, 12)
    embed_conv_stride = trial.suggest_int('Embed_conv_stride', 1, 2)
    embed_pool_kernel_size = trial.suggest_int('Embed_pool_kernel_size', 1, 2)
    embed_pool_stride = trial.suggest_int('Embed_pool_stride', 1, embed_pool_kernel_size)

    embed_n_fc_layers = trial.suggest_int('Embed_n_fc_layers', 1, 5)
    embed_fc_out_dims = [
        trial.suggest_int(f"Embed_fc_out_dim_l{i}", 4, 256, log=True) for i in range(embed_n_fc_layers - 1)
    ] + [trial.suggest_int(f"Embed_fc_out_dim_l{embed_n_fc_layers - 1}", 4, 64, log=True)]

    return {
        'conv_out_channels': embed_conv_out_channels,
        'fc_out_dims': embed_fc_out_dims,
        'conv_kernel_size': embed_conv_kernel_size,
        'conv_stride': embed_conv_stride,
        'pool_kernel_size': embed_pool_kernel_size,
        'pool_stride': embed_pool_stride
    }


def get_dummy_imply_hparams(use_mpav: bool, use_lattice: bool) -> dict:
    return {'output_dims': [16, 8], 'use_mpav': use_mpav, 'lattice_sizes': [3] if use_lattice else []}


def get_imply_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool, prefix: str) -> dict:
    imply_n_fc_layers = trial.suggest_int(f"{prefix}_Imply_n_fc_layers", 1, 3)
    imply_fc_out_dims = [
        trial.suggest_int(f"{prefix}_Imply_fc_out_dim_l{i}", 4, 256, log=True) for i in range(imply_n_fc_layers)
    ]
    lattice_sizes = [trial.suggest_int(f"{prefix}_Imply_lattice_size", 2, 16, log=True)] if use_lattice else []

    return {'output_dims': imply_fc_out_dims, 'use_mpav': use_mpav, 'lattice_sizes': lattice_sizes}


def get_rhythm_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'Rhythm')}


def get_block_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'Block')}


def get_wpw_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'WPW')}


def get_st_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'ST')}


def get_qr_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'QR')}


def get_p_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'P')}


def get_vh_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'VH')}


def get_t_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'T')}


def get_axis_hparams(trial: optuna.Trial, use_mpav: bool, use_lattice: bool) -> dict:
    return {'Imply': get_imply_hparams(trial, use_mpav, use_lattice, 'Axis')}
