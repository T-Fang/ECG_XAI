import sys

sys.path.insert(1, '/home/ftian/storage/projects/ECG_XAI/')
from src.utils.data_utils import EcgDataModule  # noqa: E402

if __name__ == '__main__':
    ecg_data_module = EcgDataModule()
    ecg_data_module.prepare_data()
