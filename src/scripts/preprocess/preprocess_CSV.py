import sys
import pandas as pd
import numpy as np
import ast
import os

import wfdb

from src.basic.constants import *
from src.basic.ecg_dataset import EcgDataset
from src.basic.dx_and_feat import Feature

sys.path.insert(1,"/Users/haoyangchen/Coding/python/ECG_XAI")

from src.utils.data_utils import EcgDataModule, add_labels_to_db, load_database, load_corresponding_ecg  # noqa: E402


def load_ds_from_raw(self):
    database = pd.read_csv(os.path.join('/Users/haoyangchen/Coding/python/ECG_XAI/data/ptbxl', 'ptbxl_database.csv'), index_col='ecg_id')
    database = database.loc[:, CHOSEN_METADATA]
    database.scp_codes = database.scp_codes.apply(lambda x: ast.literal_eval(x))
    database.age = database.age.clip(upper=90)  # patients older than 90 are considered 90
    database = add_labels_to_db(database, '/Users/haoyangchen/Coding/python/ECG_XAI/data/ptbxl')

    X = [wfdb.rdsamp(os.path.join('/Users/haoyangchen/Coding/python/ECG_XAI/data/ptbxl', f)) for f in database.filename_hr]
    X = np.array([signal.transpose() for signal, meta in X], dtype=np.float32)
    X = load_corresponding_ecg(database, '/Users/haoyangchen/Coding/python/ECG_XAI/data/ptbxl')

    sampled_indices = np.random.choice(len(database), size=len(database)//1000, replace=False)

    sampled_database = database.iloc[sampled_indices]
    sampled_X = X[sampled_indices]

    # train-val-test split: 60-20-20
    # Train

    train_mask = sampled_database.strat_fold.isin(TRAIN_FOLDS)
    train_ds = EcgDataset('train', sampled_X[train_mask], sampled_database[train_mask])

    # Validation
    val_mask = sampled_database.strat_fold.isin(VAL_FOLDS)
    val_ds = EcgDataset('val', sampled_X[val_mask], sampled_database[val_mask])

    # Test
    test_mask = sampled_database.strat_fold.isin(TEST_FOLDS)
    test_ds = EcgDataset('test', sampled_X[test_mask], sampled_database[test_mask])

    return train_ds, val_ds, test_ds


def preprocess_data(train_ds, val_ds, test_ds):
    print('Calculating features...')
    train_ds.calc_feat()
    val_ds.calc_feat()
    test_ds.calc_feat()
    return train_ds, val_ds, test_ds


def save_to_csv(dataset, output_path):
    print(f'Saving dataset to {output_path}...')

    # Assuming dataset.signal[0].get_feat() is a numpy array of features for one signal
    data = pd.DataFrame([signal.get_feat() for signal in dataset.signals], columns=[feat.name for feat in Feature])

    # Optionally add labels or metadata
    data['diagnosis'] = [signal.str_diagnoses for signal in dataset.signals]

    # if hasattr(dataset, 'metadata'):
    #     metadata_df = pd.DataFrame(dataset.metadata.reset_index(drop=True))
    #     data = pd.concat([metadata_df, data], axis=1)

    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f'Dataset saved to {output_path}')

if __name__ == '__main__':
    # Load and preprocess data
    train_ds, val_ds, test_ds = load_ds_from_raw(EcgDataModule)
    train_ds, val_ds, test_ds = preprocess_data(train_ds, val_ds, test_ds)

    # Save datasets to CSV files
    save_to_csv(train_ds, '/Users/haoyangchen/Coding/python/ECG_XAI/data/train_data.csv')
    save_to_csv(val_ds, '/Users/haoyangchen/Coding/python/ECG_XAI/data/val_data.csv')
    save_to_csv(test_ds, '/Users/haoyangchen/Coding/python/ECG_XAI/data/test_data.csv')