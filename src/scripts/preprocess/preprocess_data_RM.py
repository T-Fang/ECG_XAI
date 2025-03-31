import sys
import pandas as pd

# sys.path.insert(1, '/data2/haoyang/ECG_XAI')
sys.path.insert(1,"/Users/haoyangchen/Desktop/ChenHaoyang/coding/python/ECG_XAI")

from src.utils.data_utils import EcgDataModule  # noqa: E402

# if __name__ == '__main__':
def rm_main():
    ecg_data_module = EcgDataModule()
    # ecg_data_module.prepare_data()
    ecg_data_module.preprocess_data()

# prepare_data()->load_ds_with_feat()->load_processed_ds()->load_ds_from_raw()
# load data,        calc feat,                                  process
#                     train_feat            train               raw
