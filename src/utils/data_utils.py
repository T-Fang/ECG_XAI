import ast
import os
import wfdb
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.basic.constants import PTBXL_PATH, PROCESSED_DATA_PATH, SAMPLING_RATE, CHOSEN_METADATA, TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS  # noqa E501
from src.basic.ecg_dataset import EcgDataset


def load_database(data_dir: str):
    db = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    db = db.loc[:, CHOSEN_METADATA]
    db.scp_codes = db.scp_codes.apply(lambda x: ast.literal_eval(x))
    db.age = db.age.clip(upper=90)  # patients older than 90 are considered 90
    db = add_labels_to_db(db, data_dir)
    return db


def load_corresponding_ecg(database_df: pd.DataFrame, data_dir: str):
    if SAMPLING_RATE == 100:
        data = [wfdb.rdsamp(os.path.join(data_dir, f)) for f in database_df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(data_dir, f)) for f in database_df.filename_hr]
    data = np.array([signal.transpose() for signal, meta in data], dtype=np.float32)
    return data


def load_scp(data_dir: str):
    scp_df = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.chosen == 1]
    return scp_df


def add_labels_to_db(database_df: pd.DataFrame, data_dir: str):
    scp_df = load_scp(data_dir)

    def aggregate_superclass(scp_codes):
        corresponding_superclass = []
        for key in scp_codes.keys():
            if key in scp_df.index:
                corresponding_superclass.append(scp_df.loc[key].superclass)
        return list(set(corresponding_superclass))

    def aggregate_diagnoses(scp_codes):
        corresponding_diagnoses = []
        for key in scp_codes.keys():
            if key in scp_df.index and scp_df.loc[key].superclass != 'FORM':
                corresponding_diagnoses.extend(scp_df.loc[key].subclass.split())
        return list(set(corresponding_diagnoses))

    def aggregate_form(scp_codes):
        corresponding_form = []
        for key in scp_codes.keys():
            if key in scp_df.index and scp_df.loc[key].superclass == 'FORM':
                corresponding_form.append(scp_df.loc[key].subclass)
        return list(set(corresponding_form))

    # Add labels
    database_df['superclass'] = database_df.scp_codes.apply(aggregate_superclass)
    database_df['diagnoses'] = database_df.scp_codes.apply(aggregate_diagnoses)
    database_df['form'] = database_df.scp_codes.apply(aggregate_form)

    return database_df


class SignalDataModule(pl.LightningDataModule):
    """
    Base class for concrete Signal data modules, which are used to load/cache data for training, validation, and testing.

    At bare minimum, subclass of SignalDataModule should implement the following methods:
        - load_ds_from_raw()
    """

    def __init__(self,
                 data_dir: str,
                 processed_dir: str,
                 re_calc_feat: bool = False,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()
        # create processed directory if not exist
        if not os.path.exists(self.hparams.processed_dir):
            os.makedirs(self.hparams.processed_dir)

    def preprocess_data(self):
        # load preprocessed dataset [get_cycle,delineated] without calc features
        self.load_processed_ds()

    def calc_feat_data(self,feat_list):
        if not self.ds_path_exists('train', with_feat=True):
            self.load_ds_with_feat_NEW(feat_list,re_calc_feat=self.hparams.re_calc_feat)

    def prepare_data(self):
        # download, etc.

        # Make sure relevant dataset files exists
        if not self.ds_path_exists('train', with_feat=True):
            self.load_ds_with_feat(re_calc_feat=self.hparams.re_calc_feat)


    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = self.load_processed_sample('train', with_feat=True)
            self.val_ds = self.load_processed_sample('val', with_feat=True)

        if stage == "test":
            self.test_ds = self.load_processed_sample('test', with_feat=True)

        if stage == "predict":
            self.predict_ds = self.load_processed_sample('test', with_feat=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def load_ds_from_raw(self):
        """
        Load train, val, test datasets from raw data.
        Note that the dataset should subclass src.basic.rule_ml.SignalDataset
        """
        raise NotImplementedError

    def save_datasets(self, train_ds, val_ds, test_ds, postfix=''):
        pickle.dump(train_ds, open(os.path.join(self.hparams.processed_dir, f'train_ds{postfix}.pickle'), "wb"))
        pickle.dump(val_ds, open(os.path.join(self.hparams.processed_dir, f'val_ds{postfix}.pickle'), "wb"))
        pickle.dump(test_ds, open(os.path.join(self.hparams.processed_dir, f'test_ds{postfix}.pickle'), "wb"))

    def ds_path_exists(self, sample_name: str, with_feat: bool = False):
        """
        Check if the dataset file exists

        sample_name: one of 'train', 'val', and 'test'
        """
        file_postfix = '_with_feat' if with_feat else ''
        dataset_path = os.path.join(self.hparams.processed_dir, f'{sample_name}_ds{file_postfix}.pickle')
        return os.path.exists(dataset_path)

    def load_processed_ds(self):
        """
        Load all processed datasets.
        If the corresponding dataset files do not exist, process the datasets and save them to disk.
        """
        if not self.ds_path_exists('train', with_feat=False):
            print('Datasets not processed, processing now...')
            train_ds, val_ds, test_ds = self.load_ds_from_raw()
            self.save_datasets(train_ds, val_ds, test_ds)
        else:
            train_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'train_ds.pickle'), "rb"))
            val_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'val_ds.pickle'), "rb"))
            test_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'test_ds.pickle'), "rb"))
        return train_ds, val_ds, test_ds

    def load_processed_ds_NEW(self):
        """
        Load all processed datasets.
        If the corresponding dataset files do not exist, process the datasets and save them to disk.
        """
        if not self.ds_path_exists('train', with_feat=False):
            print('Datasets not processed')
            exit(0)
        else:
            train_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'train_ds.pickle'), "rb"))
            val_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'val_ds.pickle'), "rb"))
            test_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'test_ds.pickle'), "rb"))
        return train_ds, val_ds, test_ds

    def load_ds_with_feat(self, re_calc_feat: bool = False):
        """
        Load all processed datasets with features.

        If the corresponding dataset files do not exist,
        calculate features of each instances in the datasets and save them to disk.
        """
        if re_calc_feat or not self.ds_path_exists('train', with_feat=True):
            print('Loading datasets without features...')
            train_ds, val_ds, test_ds = self.load_processed_ds()
            print('Calculating features...')
            train_ds.calc_feat()
            val_ds.calc_feat()
            test_ds.calc_feat()
            self.save_datasets(train_ds, val_ds, test_ds, postfix='_with_feat')
        else:
            train_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'train_ds_with_feat.pickle'), "rb"))
            val_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'val_ds_with_feat.pickle'), "rb"))
            test_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'test_ds_with_feat.pickle'), "rb"))
        return train_ds, val_ds, test_ds

    def load_ds_with_feat_NEW(self,feat_list, re_calc_feat: bool = False):
        """
        Load all processed datasets with features.

        If the corresponding dataset files do not exist,
        calculate features of each instances in the datasets and save them to disk.
        """
        if re_calc_feat or not self.ds_path_exists('train', with_feat=True):
            print('Loading datasets without features...')
            train_ds, val_ds, test_ds = self.load_processed_ds_NEW()
            print('Calculating features...')
            train_ds.calc_feat_NEW(feat_list)
            val_ds.calc_feat_NEW(feat_list)
            test_ds.calc_feat_NEW(feat_list)
            self.save_datasets(train_ds, val_ds, test_ds, postfix='_with_feat')
        else:
            train_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'train_ds_with_feat.pickle'), "rb"))
            val_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'val_ds_with_feat.pickle'), "rb"))
            test_ds = pickle.load(open(os.path.join(self.hparams.processed_dir, 'test_ds_with_feat.pickle'), "rb"))
        return train_ds, val_ds, test_ds

    def load_processed_sample(self, sample_name: str, with_feat: bool = True):
        """
        Loads the processed dataset specified by the ``sample_name`` and ``with_feat``.
        If the corresponding dataset file does not exist yet, process the dataset save it to disk.

        sample_name: one of 'train', 'val' and 'test'
        """
        assert self.ds_path_exists(sample_name,
                                   with_feat), f'No dataset file found for {sample_name} with_feat={with_feat}'
        file_postfix = '_with_feat' if with_feat else ''
        dataset_path = os.path.join(self.hparams.processed_dir, f'{sample_name}_ds{file_postfix}.pickle')
        dataset = pickle.load(open(dataset_path, "rb"))
        print(sample_name, 'dataset loaded!')
        return dataset


class EcgDataModule(SignalDataModule):

    def __init__(self,
                 data_dir: str = PTBXL_PATH,
                 processed_dir: str = PROCESSED_DATA_PATH,
                 re_calc_feat: bool = False,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = True):
        super().__init__(data_dir, processed_dir, re_calc_feat, batch_size, num_workers, pin_memory)


    def load_ds_from_raw(self):
        database = load_database(self.hparams.data_dir)
        X = load_corresponding_ecg(database, self.hparams.data_dir)

        sampled_indices = np.random.choice(len(database), size=len(database) // 1000, replace=False)

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


        # train_mask = database.strat_fold.isin(TRAIN_FOLDS)
        # train_ds = EcgDataset('train', X[train_mask], database[train_mask])
        #
        # # Validation
        # val_mask = database.strat_fold.isin(VAL_FOLDS)
        # val_ds = EcgDataset('val', X[val_mask], database[val_mask])
        #
        # # Test
        # test_mask = database.strat_fold.isin(TEST_FOLDS)
        # test_ds = EcgDataset('test', X[test_mask], database[test_mask])

        return train_ds, val_ds, test_ds

