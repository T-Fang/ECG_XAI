import os
import ecg_plot
import numpy as np
import pandas as pd
import neurokit2 as nk
import torch

from src.basic.constants import SAMPLING_RATE, LEAD_TO_INDEX
from src.utils.ecg_utils import custom_ecg_delineate_plot


class Ecg:

    def __init__(self, raw: torch.Tensor, metadata: pd.Series, preprocess: bool = True):
        """
        raw: 12x5000 matrix (500Hz sampling rate).
                    Order of leads in dataset: I,II,III,aVR,aVL,aVF,V1–V6
                    Voltage Unit: mV;
                    Time Unit: second
        metadata: metadata describing the recording and the corresponding patient
        """
        self.raw = raw
        self.metadata = metadata
        
        if preprocess:
            self._preprocess()

    def _preprocess(self):
        self.clean()

    def clean(self):
        """
        Clean the ECG signal
        """
        self.cleaned = [nk.ecg_clean(lead, sampling_rate=SAMPLING_RATE) for lead in self.raw]
        self.cleaned = torch.stack(self.cleaned)
        return self.cleaned

    @property
    def labels(self):
        return self.metadata.subclass

    @property
    def age(self):
        return self.metadata.age

    @property
    def gender(self):
        """
        gender of the patient (male 0, female 1)
        """
        return self.metadata.sex

    @property
    def heart_axis(self):
        return self.metadata.heart_axis

    @property
    def MI_stage(self):
        return self.metadata.infarction_stadium1

    @property
    def superclass(self):
        return self.metadata.superclass

    @property
    def subclass(self):
        return self.metadata.subclass

    def to_csv(self, dir, filename):
        os.makedirs(dir, exist_ok=True)
        np.savetxt(os.path.join(dir, filename + '.csv'), self.recording.numpy(), delimiter=',')

    def clean(self):
        
    def process_and_plot(self, lead='II'):
        ecg_df, info = nk.ecg_process(self.recording[LEAD_TO_INDEX[lead]], sampling_rate=SAMPLING_RATE)
        nk.ecg_plot(ecg_df, rpeaks=None, sampling_rate=None, show_type='default')
        return ecg_df, info

    def delineate(self, lead='II'):
        cleaned_lead = nk.ecg_clean(self.recording[LEAD_TO_INDEX[lead]], sampling_rate=SAMPLING_RATE)
        # Extract R-peaks locations
        _, rpeaks = nk.ecg_peaks(cleaned_lead, sampling_rate=SAMPLING_RATE)
        # Delineate the ECG signal
        signal_dwt, waves_dwt = nk.ecg_delineate(cleaned_lead, rpeaks, sampling_rate=SAMPLING_RATE, method="dwt")
        custom_ecg_delineate_plot(cleaned_lead, rpeaks=rpeaks, ecg_feat=signal_dwt, window_range=(0.7, 1.6))

    def show_with_grid(self):
        ecg_plot.plot(self.recording, sample_rate=SAMPLING_RATE, title='ECG')
        ecg_plot.show()
