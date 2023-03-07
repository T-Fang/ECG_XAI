import os
import ecg_plot
import numpy as np
import pandas as pd
import neurokit2 as nk
import torch
from typing import Callable
from numpy.typing import NDArray
from neurokit2.signal import signal_rate

from src.basic.rule_ml import Signal
from src.basic.dx_and_feat import Diagnosis, Feature, keys_to_vector, zero_mid_output, fill_mid_output
from src.basic.constants import SAMPLING_RATE, LEAD_TO_INDEX, N_LEADS, ALL_LEADS
from src.utils.ecg_utils import get_all_rpeaks, get_all_delineations, custom_ecg_delineate_plot, analyze_hrv, check_inverted_wave, check_all_inverted_waves  # noqa: E501
from src.basic.cardiac_cycle import CardiacCycle, get_all_cycles

VERBOSE = True  # for debugging purpose only


class Ecg(Signal):

    def __init__(self, raw: NDArray[np.float32], metadata: pd.Series, preprocess: bool = True):
        """
        raw: 12x5000 numpy matrix (500Hz sampling rate).
                    Order of leads in dataset: I,II,III,aVR,aVL,aVF,V1–V6
                    Voltage Unit: mV;
                    Time Unit: second
        metadata: metadata describing the recording and the corresponding patient
        """
        self.raw: NDArray[np.float32] = raw
        self.metadata: pd.Series = metadata
        self.diagnoses: torch.Tensor = keys_to_vector(self.metadata.diagnoses, enum_type=Diagnosis)
        self.is_used: bool = True
        if preprocess:
            self._preprocess()

    def get_data(self):
        return torch.from_numpy(self.cleaned)

    def get_feat(self):
        mid_output = zero_mid_output()
        fill_mid_output(mid_output, torch.tensor([self.LPR, self.SHORTPR, self.PRDUR]), ['LPR', 'SHORTPR', 'PRDUR'],
                        Feature)
        fill_mid_output(mid_output, torch.tensor([self.LQRS, self.QRSDUR]), ['LQRS', 'QRSDUR'], Feature)

        return mid_output

    def get_diagnoses(self):
        return self.diagnoses

    ##############################
    # Preprocessing
    ##############################

    def _preprocess(self):
        self.get_cycles()
        # self.delineate()

    def clean(self):
        """
        Clean the ECG signal
        """
        if VERBOSE:
            print('Cleaning ECG signal...')
        cleaned = [nk.ecg_clean(lead, sampling_rate=SAMPLING_RATE) for lead in self.raw]
        self.cleaned: NDArray[np.float32] = np.stack(cleaned)
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

        if VERBOSE:
            print('Finding R-peaks...')

        self.all_rpeaks: list[NDArray[np.int64]] = []
        try:
            self.all_rpeaks = get_all_rpeaks(self.cleaned)
        except:  # noqa: E722
            if VERBOSE:
                print('--> Cannot find R-peaks')
            self.is_used = False

        return self.all_rpeaks

    @property
    def has_found_rpeaks(self):
        return hasattr(self, 'all_rpeaks')

    def delineate(self, method='dwt', check_P_inversion=True, check_T_inversion=True):
        if not self.has_found_rpeaks:
            self.find_rpeaks()
        if VERBOSE:
            print('Delineating ECG signal...')

        self.delineations = []
        are_T_inverted = [False] * N_LEADS
        are_P_inverted = [False] * N_LEADS

        if not self.is_used:
            return self.delineations, are_P_inverted, are_T_inverted

        # Delineate the ECG signal
        try:
            self.delineations: list[dict[str, list]] = get_all_delineations(self.cleaned,
                                                                            self.all_rpeaks,
                                                                            method=method)
        except:  # noqa: E722
            if VERBOSE:
                print('--> Cannot delineate the ECG signal')
            self.is_used = False
            return self.delineations, are_P_inverted, are_T_inverted

        # check whether in each lead, all features have the same number of feature points
        for i in range(N_LEADS):
            all_n_feat = [len(feat_indices) for feat_indices in self.delineations[i].values()]
            if not all(n_feat == all_n_feat[0] for n_feat in all_n_feat):
                if VERBOSE:
                    print(
                        '--> In at least one lead, not all features (such as P peaks) have the same number of feature points'
                    )
                self.is_used = False
                return self.delineations, are_P_inverted, are_T_inverted

        # Refine delineation if wave inversions occur
        if not check_T_inversion and not check_P_inversion:
            return self.delineations, are_P_inverted, are_T_inverted

        inverted_ecg: Ecg = self.invert()
        if VERBOSE:
            print('* Delineating inverted ECG signal... *')
        inverted_ecg_delineations, _, _ = inverted_ecg.delineate(method=method,
                                                                 check_T_inversion=False,
                                                                 check_P_inversion=False)

        if not inverted_ecg_delineations:
            return self.delineations, are_P_inverted, are_T_inverted

        def refine_inversion(wave_name: str):

            are_waves_inverted = check_all_inverted_waves(wave_name, self.cleaned, self.delineations)
            for i in are_waves_inverted.nonzero()[0]:
                inverted_delineation = inverted_ecg_delineations[i]
                if len(self.delineations[i]['ECG_R_Peaks']) != len(inverted_delineation['ECG_R_Peaks']):
                    continue
                if not check_inverted_wave(wave_name, self.cleaned[i], inverted_delineation):
                    self.delineations[i][f'ECG_{wave_name}_Peaks'] = inverted_delineation[f'ECG_{wave_name}_Peaks']
                    self.delineations[i][f'ECG_{wave_name}_Onsets'] = inverted_delineation[f'ECG_{wave_name}_Onsets']
                    self.delineations[i][f'ECG_{wave_name}_Offsets'] = inverted_delineation[f'ECG_{wave_name}_Offsets']

            return are_waves_inverted

        are_P_inverted = refine_inversion('P') if check_P_inversion else are_P_inverted
        are_T_inverted = refine_inversion('T') if check_T_inversion else are_T_inverted

        return self.delineations, are_P_inverted, are_T_inverted

    @property
    def has_delineated(self):
        return hasattr(self, 'delineations')

    def get_cycles(self):
        if not self.has_delineated:
            self.delineate()
        if VERBOSE:
            print('Getting cardiac cycles...')
        if not self.is_used:
            self.all_cycles = []
            return self.all_cycles

        self.all_cycles: list[list[CardiacCycle]] = get_all_cycles(self.delineations)
        # check all leads have at least one cardiac cycle
        if not all(self.all_cycles[i] for i in range(N_LEADS)):
            if VERBOSE:
                print('--> Not all leads have at least one cardiac cycle')
            self.is_used = False
        return self.all_cycles

    @property
    def has_cycles(self):
        return hasattr(self, 'all_cycles')

    ##############################
    # Additional Processing to get ECG features
    ##############################
    def calc_feat(self):
        """
        Calculate all features
        """
        # if not self.has_cycles:
        #     self.get_cycles()
        # if VERBOSE:
        #     print('Calculating features...')
        # self.calc_PR()
        # self.calc_QRS()
        pass

    def agg_feat_across_leads(self, cycle_feat_func: Callable, leads: list[str] = ALL_LEADS, use_mean: bool = True):
        """
        Aggregate averaged features (both boolean and numerical ones specified by the ``cycle_feat_func``) across the given leads

        cycle_feat_func: a function that takes a CardiacCycle and returns a tuple of boolean features and numerical features
        leads: a list of leads to aggregate features from
        use_mean: whether to average across leads.
            If not, the features will be return as a dict of lead -> tuple(bool feature, numerical feature)
        """
        if use_mean:
            lead_features = [self.calc_lead_feature(lead, cycle_feat_func) for lead in leads]
            return np.nanmean(lead_features, axis=0)
        else:
            return {lead: self.calc_lead_feature(lead, cycle_feat_func) for lead in leads}

    def calc_lead_feature(self, lead: str, cycle_feat_func: Callable):
        """
        Calculate an averaged features (both boolean and numerical ones specified by the ``cycle_feat_func``) for the given lead
        cycle_feat_func: a function that takes a CardiacCycle and returns a tuple of boolean features and numerical features
        """
        lead_cycles = self.all_cycles[LEAD_TO_INDEX[lead]]
        lead_features = np.array([cycle_feat_func(cycle) for cycle in lead_cycles])
        return np.nanmean(lead_features, axis=0)

    def calc_PR(self):
        self.LPR, self.SHORTPR, self.PRDUR = self.agg_feat_across_leads(lambda cycle: cycle.get_PR())

    def calc_QRS(self):
        self.LQRS, self.QRSDUR = self.agg_feat_across_leads(lambda cycle: cycle.get_QRS())

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
    def str_diagnoses(self):
        return self.metadata.diagnoses

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
        delineation = self.delineations[idx]

        custom_ecg_delineate_plot(ecg_signal=cleaned_lead, delineation=delineation, window_range=window_range)

    def show_with_grid(self, show_cleaned=False):
        if show_cleaned and not self.is_cleaned:
            self.clean()
        recording = self.cleaned if show_cleaned else self.raw
        ecg_plot.plot(recording, sample_rate=SAMPLING_RATE, title='ECG')
        ecg_plot.show()

    def invert(self):
        new_ecg = Ecg(self.raw * -1, self.metadata, preprocess=False)
        return new_ecg

    def check_cycle(self):
        # TODO: remove
        if not self.has_delineated:
            self.delineate()
        for key, value in self.delineations[3].items():
            print(f"{key} has length {len(value)}")
