import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.basic.ecg import Ecg


class EcgDataset(Dataset):

    def __init__(self, sample_name: str, recordings, database_df: pd.DataFrame):
        """
        sample_name: one of 'train', 'val' and 'test'
        """
        self.sample_name = sample_name

        if isinstance(recordings, np.ndarray):
            recordings = torch.from_numpy(recordings)
        self.recordings = recordings
        self.database_df = database_df
        self.ecgs = []
        self.labels = []

        for i in range(len(self.recordings)):
            ecg = Ecg(self.recordings[i], self.database_df.iloc[i])
            self.ecgs.append(ecg)
            self.labels.append(ecg.labels)

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        return (self.recordings[index], self.labels[index])

    def find_ecg_with_subclass(self, class_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.ecgs if class_name in ecg.subclass.__str__()]
        return filtered_ecg[index]

    def find_ecg_with_superclass(self, class_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.ecgs if class_name in ecg.superclass.__str__()]
        return filtered_ecg[index]
