from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from src.basic.constants import MS_PER_INDEX, LPR_THRESH, SPR_THRESH, LQRS_THRESH


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

    def get_PR_dur(self):
        pr_dur = (self.QRS_onset - self.P_onset) * MS_PER_INDEX
        return int(pr_dur > LPR_THRESH), int(pr_dur < SPR_THRESH), pr_dur

    def get_QRS_dur(self):
        qrs_dur = (self.QRS_offset - self.QRS_onset) * MS_PER_INDEX
        return int(qrs_dur > LQRS_THRESH), qrs_dur


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
