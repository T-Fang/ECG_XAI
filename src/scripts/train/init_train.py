import os
import sys
import platform
import torch

CURRENT_OS = platform.system()
PROJECT_PATH = '/home/ftian/storage/ECG_XAI/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/FYP/ECG_XAI/'
sys.path.insert(1, PROJECT_PATH)

from src.utils.train_utils import seed_all  # noqa: E402

seed_all()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('medium')
