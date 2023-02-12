import os
import ecg_plot
import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.signal import signal_rate

from src.basic.constants import SAMPLING_RATE, LEAD_TO_INDEX, N_LEADS
from src.utils.ecg_utils import get_all_rpeaks, get_all_delineations, custom_ecg_delineate_plot, analyze_hrv, check_inverted_T, check_all_inverted_T, check_inverted_P, check_all_inverted_P  # noqa: E501


class Ecg:

    def __init__(self, raw: np.ndarray, metadata: pd.Series, preprocess: bool = False):
        """
        raw: 12x5000 numpy matrix (500Hz sampling rate).
                    Order of leads in dataset: I,II,III,aVR,aVL,aVF,V1–V6
                    Voltage Unit: mV;
                    Time Unit: second
        metadata: metadata describing the recording and the corresponding patient
        """
        self.raw = raw
        self.metadata = metadata

        if preprocess:
            self._preprocess()

    ##############################
    # Preprocessing
    ##############################

    def _preprocess(self):
        # self.clean()
        self.delineate()

    def clean(self):
        """
        Clean the ECG signal
        """
        self.cleaned = [nk.ecg_clean(lead, sampling_rate=SAMPLING_RATE) for lead in self.raw]
        self.cleaned = np.stack(self.cleaned)
        return self.cleaned

    @property
    def is_cleaned(self):
        return hasattr(self, 'cleaned')

    def find_rpeaks(self):
        """
        # Extract R-peaks locations: self.all_rpeaks has 12 elements,
        # each element is a NDArray of R-peaks locations
        """
        if not self.is_cleaned:
            self.clean()

        self.all_rpeaks = get_all_rpeaks(self.cleaned)
        return self.all_rpeaks

    @property
    def has_found_rpeaks(self):
        return hasattr(self, 'all_rpeaks')

    def delineate(self, method='dwt', check_P_inversion=True, check_T_inversion=True):
        if not self.has_found_rpeaks:
            self.find_rpeaks()

        self.delineations = get_all_delineations(self.cleaned, self.all_rpeaks, method=method)

        are_T_inverted = [False] * N_LEADS
        are_P_inverted = [False] * N_LEADS

        if check_T_inversion or check_P_inversion:
            inverted_ecg: Ecg = self.invert()
            inverted_ecg_delineations, _, _ = inverted_ecg.delineate(method=method,
                                                                     check_T_inversion=False,
                                                                     check_P_inversion=False)

        if check_T_inversion:
            are_T_inverted = check_all_inverted_T(self.cleaned, self.delineations)
            for i in are_T_inverted.nonzero()[0]:
                inverted_ecg_delineation = inverted_ecg_delineations[i]
                if not check_inverted_T(inverted_ecg.cleaned[i], inverted_ecg_delineation):
                    self.delineations[i]['ECG_T_Peaks'] = inverted_ecg_delineation['ECG_T_Peaks']
                    self.delineations[i]['ECG_T_Onsets'] = inverted_ecg_delineation['ECG_T_Onsets']
                    self.delineations[i]['ECG_T_Offsets'] = inverted_ecg_delineation['ECG_T_Offsets']

        if check_P_inversion:
            are_P_inverted = check_all_inverted_P(self.cleaned, self.delineations)
            for i in are_P_inverted.nonzero()[0]:
                inverted_ecg_delineation = inverted_ecg_delineations[i]
                if not check_inverted_P(inverted_ecg.cleaned[i], inverted_ecg_delineation):
                    self.delineations[i]['ECG_P_Peaks'] = inverted_ecg_delineation['ECG_P_Peaks']
                    self.delineations[i]['ECG_P_Onsets'] = inverted_ecg_delineation['ECG_P_Onsets']
                    self.delineations[i]['ECG_P_Offsets'] = inverted_ecg_delineation['ECG_P_Offsets']

        return self.delineations, are_P_inverted, are_T_inverted

    @property
    def has_delineated(self):
        return hasattr(self, 'delineations')

    ##############################
    # Additional Preprocessing
    ##############################
    def calc_heart_rate(self):
        """
        Calculate heart rate for each lead
        """
        if not self.has_found_rpeaks:
            self.find_rpeaks()
        self.all_heart_rates = []
        for i in range(N_LEADS):
            rpeaks = self.all_rpeaks[i]
            self.all_heart_rates.append(
                signal_rate(rpeaks, sampling_rate=SAMPLING_RATE, desired_length=self.raw.shape[1]))

        self.all_heart_rates = np.stack(self.all_heart_rates)
        self.heart_rate_mean = np.mean(self.all_heart_rates)
        self.heart_rate_std = np.std(self.all_heart_rates)
        return self.all_heart_rates

    def calc_hrv(self):
        """
        Calculate heart rate variability for each lead
        """
        if not self.has_found_rpeaks:
            self.find_rpeaks()
        self.all_hrv = []
        for i in range(N_LEADS):
            rpeaks = self.all_rpeaks[i]
            hrv_analysis = analyze_hrv(rpeaks)
            self.all_hrv.append(hrv_analysis['RMSSD (ms)'])

        return self.all_hrv

    ##############################
    # Properties
    ##############################
    @property
    def labels(self):
        # TODO: return a torch.Tensor
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

    ##############################
    # Plotting
    ##############################
    def process_and_plot(self, lead='II'):
        ecg_df, info = nk.ecg_process(self.raw[LEAD_TO_INDEX[lead]], sampling_rate=SAMPLING_RATE)
        nk.ecg_plot(ecg_df, rpeaks=None, sampling_rate=None, show_type='default')
        return ecg_df, info

    def plot_delineation(self, lead='II', method='dwt', window_range=(0.7, 1.6)):
        if not self.has_delineated:
            self.delineate()

        idx = LEAD_TO_INDEX[lead]
        cleaned_lead = self.cleaned[idx]
        rpeaks = self.all_rpeaks[idx]
        delineation = self.delineations[idx]

        custom_ecg_delineate_plot(ecg_signal=cleaned_lead,
                                  rpeaks=rpeaks,
                                  delineation=delineation,
                                  window_range=window_range)

    def show_with_grid(self, show_cleaned=False):
        if show_cleaned and not self.is_cleaned:
            self.clean()
        recording = self.cleaned if show_cleaned else self.raw
        ecg_plot.plot(recording, sample_rate=SAMPLING_RATE, title='ECG')
        ecg_plot.show()

    ##############################
    # TODO: To be deleted
    ##############################
    def invert(self):
        new_ecg = Ecg(self.raw * -1, self.metadata)
        return new_ecg
