import sys
import pandas as pd

# sys.path.insert(1, '/data2/haoyang/ECG_XAI')
sys.path.insert(1,"/Users/haoyangchen/Desktop/ChenHaoyang/coding/python/ECG_XAI")

from src.utils.data_utils import EcgDataModule
from src.basic.dx_and_feat import Feature# noqa: E402

# if __name__ == '__main__':
def rm_main():
    ecg_data_module = EcgDataModule()
    train_ds, _,_ = ecg_data_module.load_ds_with_feat(False)
    train_signals=train_ds.signals
    # print(train_signals[0].get_feat_with_name())
    df= pd.DataFrame(columns=[feat.name for feat in Feature])
    for signal in train_signals:
        feat_dict = signal.get_feat_with_name()
        df.loc[len(df)]=feat_dict

    print(df)
    return df

# prepare_data()->load_ds_with_feat()->load_processed_ds()->load_ds_from_raw()
# load data,        calc feat,                                  process
#                     train_feat            train               raw
