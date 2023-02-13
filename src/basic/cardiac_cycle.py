from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


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


def get_cycle_around(rpeak_idx: np.int32, delineation: dict[str, NDArray[np.int32]]) -> CardiacCycle:
    """
    Given the delineation of a lead, get the cardiac cycle around the given R peak
    """

    def find_feat_point_before(feat_point_name: str) -> int:
        for i in range(rpeak_idx, -1, -1):
            if delineation[feat_point_name][i]:
                return i
        return None

    def find_feat_point_after(feat_point_name: str) -> int:
        for i in range(rpeak_idx, len(delineation[feat_point_name])):
            if delineation[feat_point_name][i]:
                return i
        return None

    P_onset = find_feat_point_before('ECG_P_Onsets')
    P_peak = find_feat_point_before('ECG_P_Peaks')
    P_offset = find_feat_point_before('ECG_P_Offsets')
    QRS_onset = find_feat_point_before('ECG_R_Onsets')
    Q_peak = find_feat_point_before('ECG_Q_Peaks')
    R_peak = rpeak_idx
    S_peak = find_feat_point_after('ECG_S_Peaks')
    QRS_offset = find_feat_point_after('ECG_R_Offsets')
    T_onset = find_feat_point_after('ECG_T_Onsets')
    T_peak = find_feat_point_after('ECG_T_Peaks')
    T_offset = find_feat_point_after('ECG_T_Offsets')

    if None in [P_onset, P_peak, P_offset, QRS_onset, Q_peak, R_peak, S_peak, QRS_offset, T_onset, T_peak, T_offset]:
        return None

    return CardiacCycle(P_onset=P_onset,
                        P_peak=P_peak,
                        P_offset=P_offset,
                        QRS_onset=QRS_onset,
                        Q_peak=Q_peak,
                        R_peak=R_peak,
                        S_peak=S_peak,
                        QRS_offset=QRS_offset,
                        T_onset=T_onset,
                        T_peak=T_peak,
                        T_offset=T_offset)


def get_cycles_in_lead(rpeaks: NDArray[np.int32], delineation: dict[str, NDArray[np.int32]]) -> list[CardiacCycle]:
    """
    Get cardiac cycles in a single lead given its delineation
    """
    all_lead_cycles = []
    for rpeak_idx in rpeaks:
        cycle = get_cycle_around(rpeak_idx, delineation)
        if cycle:
            all_lead_cycles.append(cycle)
    return all_lead_cycles


def get_all_cycles(all_rpeaks: list[NDArray[np.int32]],
                   delineations: list[dict[str, NDArray[np.int32]]]) -> list[list[CardiacCycle]]:
    """
    Group the cardiac cycles of all leads into a list of lists
    """
    all_cycles = []
    for lead_rpeaks, lead_delineation in zip(all_rpeaks, delineations):
        all_cycles.append(get_cycles_in_lead(lead_rpeaks, lead_delineation))
    return all_cycles
