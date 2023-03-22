from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from src.basic.constants import DEEP_S_THRESH, DOM_R_THRESH, DOM_S_THRESH, INVT_THRESH, LP_THRESH_II, LQRS_WPW_THRESH, MS_PER_INDEX, LPR_THRESH, PEAK_P_THRESH_II, PEAK_P_THRESH_V1, PEAK_R_THRESH, POS_QRS_THRESH, SPR_THRESH, LQRS_THRESH, Q_AMP_THRESH, Q_DUR_THRESH, R_AMP_THRESH  # noqa: E501


@dataclass
class CardiacCycle():
    P_onset: int
    P_peak: int
    P_offset: int
    QRS_onset: int
    Q_peak: int
    R_peak: int
    S_peak: int
    QRS_offset: int
    T_onset: int
    T_peak: int
    T_offset: int
    extra_info: dict = {}

    @property
    def signal(self):
        if 'signal' in self.extra_info:
            return self.extra_info['signal']
        else:
            raise AttributeError('No signal found in CardiacCycle\'s extra_info')

    def is_sinus(self):
        return self.signal[self.P_peak] > 0

    def get_PR_dur(self):
        if self.QRS_onset < self.P_onset:
            return np.nan, np.nan, np.nan
        pr_dur = (self.QRS_onset - self.P_onset) * MS_PER_INDEX
        return int(pr_dur > LPR_THRESH), int(pr_dur < SPR_THRESH), pr_dur

    def get_QRS_dur(self):
        if self.QRS_offset < self.QRS_onset:
            return np.nan, np.nan, np.nan
        qrs_dur = (self.QRS_offset - self.QRS_onset) * MS_PER_INDEX
        return int(qrs_dur > LQRS_THRESH), int(qrs_dur > LQRS_WPW_THRESH), qrs_dur

    def get_ST_amp(self):
        if self.T_onset < self.QRS_offset:
            return np.nan, np.nan, np.nan
        ST_AMP = np.mean(self.signal[self.QRS_offset:self.T_onset + 1])
        return int(ST_AMP > 0), int(ST_AMP < 0), int(ST_AMP)

    def get_PRWP(self):
        R_AMP = self.signal[self.R_peak]
        low_thresh, high_thresh = R_AMP_THRESH[self.extra_info['lead']][
            self.extra_info['age_range']] if self.extra_info['lead'] == 'I' else R_AMP_THRESH[self.extra_info['lead']]

        PRWP = min(1, (high_thresh - R_AMP) / (high_thresh - low_thresh))
        return PRWP

    def get_PATH_Q(self):
        Q_AMP = self.signal[self.Q_peak]
        Q_DUR = max(0, (self.extra_info['Q_offset'] - self.QRS_onset) * MS_PER_INDEX)
        PATH_Q = int((Q_DUR > Q_DUR_THRESH) and (Q_AMP < Q_AMP_THRESH[self.extra_info['lead']]))
        return Q_DUR, Q_AMP, PATH_Q

    def get_Q_offset(self):
        """
        Get the Q offset using binary search to search for the index
        between Q_peak and R_peak, whose value (i.e., ``self.signal[index]``) is closest to 0.

        (Note that ``self.signal[Q_peak]`` should be -ve and ``self.signal[R_peak]`` should be +ve)
        """

        def calc_Q_offset():
            if self.signal[self.Q_peak] > 0 or self.signal[self.R_peak] < 0:
                return None
            if self.signal[self.Q_peak] == 0:
                return self.Q_peak
            if self.signal[self.R_peak] == 0:
                return self.R_peak

            left = self.Q_peak
            right = self.R_peak
            while left < right:
                mid = (left + right) // 2
                if self.signal[mid] == 0:
                    return mid
                elif self.signal[mid] > 0:
                    right = mid
                else:
                    left = mid + 1
            return left

        self.extra_info['Q_offset'] = calc_Q_offset()
        return self.extra_info['Q_offset']

    def get_P_dur(self):
        P_DUR = max(0, (self.P_offset - self.P_onset) * MS_PER_INDEX)
        if self.extra_info['lead'] == 'II':
            return int(P_DUR > LP_THRESH_II), P_DUR
        else:
            return 0, P_DUR

    def get_P_amp(self):
        P_AMP = self.signal[self.P_peak]
        lead = self.extra_info['lead']
        if lead == 'II':
            return int(P_AMP > PEAK_P_THRESH_II), P_AMP
        elif lead == 'V1':
            return int(P_AMP > PEAK_P_THRESH_V1), P_AMP
        else:
            return 0, P_AMP

    def get_RS_ratio(self):
        R_AMP = np.abs(self.signal[self.R_peak])
        S_AMP = np.abs(self.signal[self.S_peak])
        PEAK_R = int(R_AMP > PEAK_R_THRESH)
        DEEP_S = int(S_AMP > DEEP_S_THRESH)
        RS_RATIO = R_AMP / S_AMP if S_AMP > 0 else 10000  # very unlikly that S amplitude is exactly 0
        DOM_R = int(RS_RATIO > DOM_R_THRESH)
        DOM_S = int(RS_RATIO < DOM_S_THRESH)
        return R_AMP, S_AMP, PEAK_R, DEEP_S, DOM_R, DOM_S, RS_RATIO

    def get_T_amp(self):
        T_AMP = self.signal[self.T_peak]
        return int(T_AMP < INVT_THRESH), T_AMP

    def get_QRS_sum(self):
        # sum q, r, s amplitudes
        QRS_SUM = np.sum(self.signal[self.Q_peak:self.S_peak + 1])
        return int(QRS_SUM > POS_QRS_THRESH), QRS_SUM

    @property
    def R_amp(self):
        return np.abs(self.signal[self.R_peak])

    @property
    def S_amp(self):
        return np.abs(self.signal[self.S_peak])


def check_delineation_dict_values_have_same_length(delineation: dict[str, NDArray[np.int64]]):
    lengths = [len(value) for value in delineation.values()]
    return len(set(lengths)) == 1


def get_cycles_in_lead(delineation: dict[str, NDArray[np.int64]]) -> list[CardiacCycle]:
    """
    Get cardiac cycles in a single lead given its delineation
    """
    # print('inside get_cycles_in_lead')
    # assert check_delineation_dict_values_have_same_length(delineation)
    feature_indices = np.stack([
        delineation['ECG_P_Onsets'], delineation['ECG_P_Peaks'], delineation['ECG_P_Offsets'],
        delineation['ECG_R_Onsets'], delineation['ECG_Q_Peaks'], delineation['ECG_R_Peaks'], delineation['ECG_S_Peaks'],
        delineation['ECG_R_Offsets'], delineation['ECG_T_Onsets'], delineation['ECG_T_Peaks'],
        delineation['ECG_T_Offsets']
    ],
                               axis=1)
    # remove cardiac cycle with nan
    feature_indices = feature_indices[~np.isnan(feature_indices).any(axis=1)]
    return [indices2cycle(indices) for indices in feature_indices]


def indices2cycle(feature_indices: NDArray[np.int64]):
    assert len(feature_indices) == 11
    return CardiacCycle(P_onset=feature_indices[0],
                        P_peak=feature_indices[1],
                        P_offset=feature_indices[2],
                        QRS_onset=feature_indices[3],
                        Q_peak=feature_indices[4],
                        R_peak=feature_indices[5],
                        S_peak=feature_indices[6],
                        QRS_offset=feature_indices[7],
                        T_onset=feature_indices[8],
                        T_peak=feature_indices[9],
                        T_offset=feature_indices[10])


def get_all_cycles(delineations: list[dict[str, NDArray[np.int64]]]) -> list[list[CardiacCycle]]:
    """
    Group the cardiac cycles of all leads into a list of lists
    """
    all_cycles = [get_cycles_in_lead(lead_delineation) for lead_delineation in delineations]
    return all_cycles
