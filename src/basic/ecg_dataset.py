import pandas as pd
import numpy as np
from numpy.typing import NDArray

from src.basic.ecg import Ecg
from src.basic.rule_ml import SignalDataset


class EcgDataset(SignalDataset):

    def __init__(self, sample_name: str, raw_recordings: NDArray[np.float32], database_df: pd.DataFrame):
        """
        sample_name: one of 'train', 'val' and 'test'
        """
        self.sample_name: str = sample_name

        self.raw_recordings: NDArray[np.float32] = raw_recordings
        self.database_df: pd.DataFrame = database_df
        self.signals: list[Ecg] = []
        self.is_ecg_used: list[bool] = []
        for i in range(len(self.raw_recordings)):
            # TODO: remove print
            print('***********************************\n**processing ecg', i, 'of', len(self.raw_recordings), '**')
            ecg = Ecg(self.raw_recordings[i], self.database_df.iloc[i])
            self.is_ecg_used.append(ecg.is_used)
            if ecg.is_used:
                self.signals.append(ecg)

    def find_ecg_with_diagnosis(self, diagnosis_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.signals if diagnosis_name in ecg.str_diagnoses.__str__()]
        return filtered_ecg[index]

    def find_ecg_with_superclass(self, class_name: str, index: int = 0):
        filtered_ecg = [ecg for ecg in self.signals if class_name in ecg.superclass.__str__()]
        return filtered_ecg[index]
