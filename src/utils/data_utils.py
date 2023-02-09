import ast
import os
import wfdb
import numpy as np
import pandas as pd
from src.basic.constants import PTBXL_PATH, SAMPLING_RATE, CHOSEN_METADATA, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS
from src.basic.ecg_dataset import EcgDataset


def load_database():
    db = pd.read_csv(PTBXL_PATH + 'ptbxl_database.csv', index_col='ecg_id')
    db = db.loc[:, CHOSEN_METADATA]
    db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))
    db.age = db.age.clip(upper=90)  # patients older than 90 are considered 90
    return db


def load_corresponding_ecg(database_df: pd.DataFrame):
    if SAMPLING_RATE == 100:
        data = [wfdb.rdsamp(os.path.join(PTBXL_PATH, f)) for f in database_df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(PTBXL_PATH, f)) for f in database_df.filename_hr]
    data = np.array([signal.transpose() for signal, meta in data])
    return data


def load_scp():
    scp_df = pd.read_csv(PTBXL_PATH + 'scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df.chosen == 1]
    return scp_df


def add_labels_to_db(database_df: pd.DataFrame):
    scp_df = load_scp()

    def aggregate_superclass(scp_codes):
        corresponding_superclass = []
        for key in scp_codes.keys():
            if key in scp_df.index:
                corresponding_superclass.append(scp_df.loc[key].superclass)
        return list(set(corresponding_superclass))

    def aggregate_subclass(scp_codes):
        corresponding_subclass = []
        for key in scp_codes.keys():
            if key in scp_df.index:
                corresponding_subclass.extend(scp_df.loc[key].subclass.split())
        return list(set(corresponding_subclass))

    # Add labels
    database_df['superclass'] = database_df.scp_codes.apply(aggregate_superclass)
    database_df['subclass'] = database_df.scp_codes.apply(aggregate_subclass)

    return database_df


def load_datasets():
    database = load_database()
    database = add_labels_to_db(database)
    X = load_corresponding_ecg(database)

    # Train
    train_mask = database.strat_fold.isin(TRAIN_FOLDS)
    train_ds = EcgDataset('train', X[train_mask], database[train_mask])
    # y_train = database[train_mask].diagnostic_subclass

    # Validation
    val_mask = database.strat_fold.isin(VAL_FOLDS)
    val_ds = EcgDataset('val', X[val_mask], database[val_mask])
    # y_val = database[val_mask].diagnostic_subclass

    # Test
    test_mask = database.strat_fold.isin(TEST_FOLDS)
    test_ds = EcgDataset('test', X[test_mask], database[test_mask])
    # y_test = database[test_mask].diagnostic_subclass

    return train_ds, val_ds, test_ds
