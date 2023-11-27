import sys
import pandas as pd

# sys.path.insert(1, '/data2/haoyang/ECG_XAI')
sys.path.insert(1,"/Users/haoyangchen/Desktop/ChenHaoyang/coding/python/ECG_XAI")

from src.utils.data_utils import EcgDataModule  # noqa: E402

# if __name__ == '__main__':
def rm_main(data):
    data = {'features': [
        ['SINUS', 'HR', 'RR_DIFF', 'QRS_DUR', 'PR_DUR', 'ST_AMP', 'Q_DUR', 'Q_AMP', 'PRWP', 'P_DUR', 'P_AMP', 'AGE',
         'MALE', 'R_AMP', 'S_AMP', 'RS_RATIO', 'RAD', 'T_AMP', 'QRS_SUM']]}
    df = pd.DataFrame(data)

    # 提取 features 列的元素
    features_list = df['features'].iloc[0]

    # 打印结果
    print(features_list)
    ecg_data_module = EcgDataModule()
    # # ecg_data_module.prepare_data()
    # # ecg_data_module.preprocess_data()
    ecg_data_module.calc_feat_data(features_list)

# prepare_data()->load_ds_with_feat()->load_processed_ds()->load_ds_from_raw()
# load data,        calc feat,                                  process
#                     train_feat            train               raw
