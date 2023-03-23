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
from src.basic.dx_and_feat import Diagnosis, keys_to_vector, get_feat_vector
from src.basic.constants import AGE_OLD_THRESH, LVH_L1_OLD_THRESH, LVH_L1_YOUNG_THRESH, LVH_L2_FEMALE_THRESH, LVH_L2_MALE_THRESH, MS_PER_INDEX, P_LEADS, PRWP_LEADS, SAMPLING_RATE, LEAD_TO_INDEX, N_LEADS, ALL_LEADS, DURATION, T_LEADS  # noqa: E501
from src.utils.ecg_utils import get_all_rpeaks, get_all_delineations, custom_ecg_delineate_plot, check_inverted_wave, check_all_inverted_waves  # noqa: E501
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
        return get_feat_vector(self)

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
                if not check_inverted_wave(wave_name, inverted_ecg.cleaned[i], inverted_delineation):
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
        if not self.has_cycles:
            self.get_cycles()
        if VERBOSE:
            print('Calculating features...')
        self.add_extra_info()

        # * calculate objective features for RhythmModule
        self.calc_heart_rate()
        self.calc_sinus()
        self.calc_RR_DIFF()

        # * calculate objective features for BlockModule
        self.calc_PR()
        self.calc_QRS()

        # * calculate objective features for WPWModule
        # Already calculated when getting objective features for BlockModule

        # * calculate objective features for STModule
        self.calc_ST()

        # * calculate objective features for QRModule
        self.calc_PRWP()
        self.calc_PATH_Q()

        # * calculate objective features for PModule
        self.calc_P()

        # * calculate objective features for VHModule
        self.calc_VH_related()

        # * calculate objective features for TModule
        self.calc_T()

        # * calculate objective features for AxisModule
        self.calc_axis()

    def add_extra_info(self):
        """
        Add extra information to each cardiac cycle's ``extra_info`` dict,
        such as reference to the corresponding lead signal.
        """
        for lead in ALL_LEADS:
            lead_idx = LEAD_TO_INDEX[lead]
            for cycle in self.all_cycles[lead_idx]:
                cycle.extra_info['age_range'] = self.age_range
                cycle.extra_info['lead'] = lead
                cycle.extra_info['signal'] = self.cleaned[lead_idx]
                cycle.get_Q_offset()

    def agg_feat_across_leads(self, cycle_feat_func: Callable, leads: list[str] = ALL_LEADS, use_mean: bool = True):
        """
        Aggregate averaged features (both boolean and numerical ones specified by the ``cycle_feat_func``) across the given leads

        cycle_feat_func: a function that takes a CardiacCycle and returns a tuple of boolean features and numerical features
        leads: a list of leads to aggregate features from
        use_mean: whether to average across leads.
            If not, the features will be return as a dict of lead -> tuple(bool feature, numerical feature)
        """
        if use_mean:
            lead_features = [self.calc_lead_feat(lead, cycle_feat_func) for lead in leads]
            return np.mean(lead_features, axis=0)
        else:
            return {lead: self.calc_lead_feat(lead, cycle_feat_func) for lead in leads}

    def calc_lead_feat(self, lead: str, cycle_feat_func: Callable):
        """
        Calculate an averaged features (both boolean and numerical ones specified by the ``cycle_feat_func``) for the given lead
        cycle_feat_func: a function that takes a CardiacCycle and returns a tuple of boolean features and numerical features
        """
        lead_cycles = self.all_cycles[LEAD_TO_INDEX[lead]]
        lead_features = np.array([cycle_feat_func(cycle) for cycle in lead_cycles])

        mean_lead_feat: np.ndarray = np.nanmean(lead_features, axis=0)
        if np.isnan(mean_lead_feat).any():
            mean_lead_feat = np.zeros(mean_lead_feat.shape, dtype=int)
        return mean_lead_feat

    # * Objective features for RhythmModule
    def calc_heart_rate(self):
        all_heart_rates = []
        for i in range(N_LEADS):
            rpeaks = self.all_rpeaks[i]
            all_heart_rates.append(signal_rate(rpeaks, sampling_rate=SAMPLING_RATE, desired_length=self.raw.shape[1]))

        all_heart_rates = np.stack(all_heart_rates)
        self.HR = np.nanmean(all_heart_rates)
        self.HR_STD = np.std(all_heart_rates)
        self.expected_n_cycles = int(self.HR * DURATION / 60)
        self.BRAD = self.HR < 60
        self.TACH = self.HR > 100

    def calc_sinus(self):
        """
        Examine lead II and check whether each P wave is +ve AND followed by a QRS
        """
        lead_II_cycles = self.all_cycles[LEAD_TO_INDEX['II']]
        n_sinus_cycles = 0
        for cycle in lead_II_cycles:
            if cycle.is_sinus():
                n_sinus_cycles += 1
        self.SINUS = n_sinus_cycles / self.expected_n_cycles

    def calc_RR_DIFF(self):
        """
        Calculate the difference between max RR interval and min RR interval
        """
        all_RR_DIFF = []
        for i in range(N_LEADS):
            rpeaks = self.all_rpeaks[i]
            if len(rpeaks) < 2:
                continue
            rr_intervals = np.diff(rpeaks)
            all_RR_DIFF.append(np.max(rr_intervals) - np.min(rr_intervals))

        if not all_RR_DIFF:
            self.RR_DIFF = 0
        else:
            self.RR_DIFF = np.mean(all_RR_DIFF) * MS_PER_INDEX

    # * Objective features for BlockModule
    def calc_PR(self):
        self.LPR, self.SPR, self.PR_DUR = self.agg_feat_across_leads(lambda cycle: cycle.get_PR_dur())

    def calc_QRS(self):
        self.LQRS, self.LQRS_WPW, self.QRS_DUR = self.agg_feat_across_leads(lambda cycle: cycle.get_QRS_dur())

    # * Objective features for WPWModule
    # Already calculated when getting objective features for BlockModule

    # * Objective features for STModule
    def calc_ST(self):
        for lead in ALL_LEADS:
            ST_feat = self.calc_lead_feat(lead, lambda cycle: cycle.get_ST_amp())
            setattr(self, f'STE_{lead}', ST_feat[0])
            setattr(self, f'STD_{lead}', ST_feat[1])
            setattr(self, f'ST_AMP_{lead}', ST_feat[2])

    # * Objective features for QRModule
    def calc_PRWP(self):
        all_PRWP = []
        for lead in PRWP_LEADS:
            PRWP = self.calc_lead_feat(lead, lambda cycle: cycle.get_PRWP())
            all_PRWP.append(PRWP)
        self.PRWP = min(1, sum(all_PRWP))

    def calc_PATH_Q(self):
        for lead in ALL_LEADS:
            Q_feat = self.calc_lead_feat(lead, lambda cycle: cycle.get_PATH_Q())
            setattr(self, f'Q_DUR_{lead}', Q_feat[0])
            setattr(self, f'Q_AMP_{lead}', Q_feat[1])
            setattr(self, f'PATH_Q_{lead}', Q_feat[2])

    # * Objective features for PModule
    def calc_P(self):
        self.LP_II, self.P_DUR_II = self.calc_lead_feat('II', lambda cycle: cycle.get_P_dur())
        for lead in P_LEADS:
            P_feat = self.calc_lead_feat(lead, lambda cycle: cycle.get_P_amp())
            setattr(self, f'PEAK_P_{lead}', P_feat[0])
            setattr(self, f'P_AMP_{lead}', P_feat[1])

    # * Objective features for VHModule
    def calc_VH_related(self):
        self.AGE = self.age
        self.AGE_OLD = int(self.AGE > AGE_OLD_THRESH)
        self.MALE = int(not self.sex)
        self.R_AMP_V1, self.S_AMP_V1, self.PEAK_R_V1, _, self.DOM_R_V1, _, self.RS_RATIO_V1 = self.calc_lead_feat(
            'V1', lambda cycle: cycle.get_RS_ratio())
        self.R_AMP_V5, self.S_AMP_V5, _, self.DEEP_S_V5, _, self.DOM_S_V5, self.RS_RATIO_V5 = self.calc_lead_feat(
            'V5', lambda cycle: cycle.get_RS_ratio())
        self.R_AMP_V6, self.S_AMP_V6, _, self.DEEP_S_V6, _, self.DOM_S_V5, self.RS_RATIO_V6 = self.calc_lead_feat(
            'V6', lambda cycle: cycle.get_RS_ratio())
        self.R_AMP_aVL = self.calc_lead_feat('aVL', lambda cycle: cycle.R_amp)
        self.S_AMP_V3 = self.calc_lead_feat('V3', lambda cycle: cycle.S_amp)

        LVH_L1_left_term = self.S_AMP_V1 + self.R_AMP_V6
        self.LVH_L1_OLD = int(LVH_L1_left_term > LVH_L1_OLD_THRESH)
        self.LVH_L1_YOUNG = int(LVH_L1_left_term > LVH_L1_YOUNG_THRESH)

        LVH_L2_left_term = self.R_AMP_aVL + self.S_AMP_V3
        self.LVH_L2_MALE = int(LVH_L2_left_term > LVH_L2_MALE_THRESH)
        self.LVH_L2_FEMALE = int(LVH_L2_left_term > LVH_L2_FEMALE_THRESH)

    # * Objective features for TModule
    def calc_T(self):
        for lead in T_LEADS:
            T_feat = self.calc_lead_feat(lead, lambda cycle: cycle.get_T_amp())
            setattr(self, f'INVT_{lead}', T_feat[0])
            setattr(self, f'T_AMP_{lead}', T_feat[1])

    # * Objective features for TModule
    def calc_axis(self):
        self.POS_QRS_I, self.QRS_SUM_I = self.calc_lead_feat('I', lambda cycle: cycle.get_QRS_sum())
        self.POS_QRS_aVF, self.QRS_SUM_aVF = self.calc_lead_feat('aVF', lambda cycle: cycle.get_QRS_sum())
        self.NORM_AXIS = max(0, self.POS_QRS_I + self.POS_QRS_aVF - 1)
        self.LAD = max(0, self.POS_QRS_I - self.POS_QRS_aVF)
        self.RAD = max(0, self.POS_QRS_aVF - self.POS_QRS_I)

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
    def age_range(self):
        if self.age < 20:
            return 'young'
        elif self.age < 30:
            return 'middle'
        else:
            return 'old'

    @property
    def sex(self):
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
