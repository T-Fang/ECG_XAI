import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.basic.ecg import Ecg


class EcgDataset(Dataset):

    def __init__(self, sample_name: str, raw_recordings: np.ndarray, database_df: pd.DataFrame):
        """
        sample_name: one of 'train', 'val' and 'test'
        """
        self.sample_name: str = sample_name

        self.raw_recordings: np.ndarray = raw_recordings
        self.database_df: pd.DataFrame = database_df
        self.ecgs: list[Ecg] = []
        self.labels: list[torch.Tensor] = []

        for i in range(len(self.raw_recordings)):
            ecg = Ecg(self.raw_recordings[i], self.database_df.iloc[i])
            self.ecgs.append(ecg)
            self.labels.append(ecg.labels)

    def __len__(self):
        return len(self.raw_recordings)

    def __getitem__(self, index):
        return (torch.from_numpy(self.ecgs[index].cleaned), self.labels[index])

    def find_ecg_with_subclass(self, class_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.ecgs if class_name in ecg.subclass.__str__()]
        return filtered_ecg[index]

    def find_ecg_with_superclass(self, class_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.ecgs if class_name in ecg.superclass.__str__()]
        return filtered_ecg[index]
