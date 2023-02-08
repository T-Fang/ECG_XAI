import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.basic.ecg import Ecg


class EcgDataset(Dataset):

    def __init__(self, sample_name: str, recordings: np.ndarray | torch.Tensor, database_df: pd.DataFrame):
        """
        sample_name: one of 'train', 'val' and 'test'
        """
        self.sample_name = sample_name

        if isinstance(recordings, np.ndarray):
            recordings = torch.from_numpy(recordings)
        self.recordings = recordings
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
