import torch
# import random
from pytorch_lightning import seed_everything
from src.basic.constants import MANUAL_SEED


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
