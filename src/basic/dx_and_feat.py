import torch
import pandas as pd
from enum import Enum


class Diagnosis(Enum):
    NORM = 0
    AFIB = 1
    AFLT = 2
    SARRH = 3
    SBRAD = 4
    SR = 5
    STACH = 6
    AVB = 7  # First degree AV block
    IVCD = 8
    LAFB = 9
    LBBB = 10
    LPFB = 11
    RBBB = 12
    WPW = 13
    LAE = 14
    LVH = 15
    RAE = 16
    RVH = 17
    AMI = 18
    IMI = 19
    LMI = 20


class Feature(Enum):
    # * Objective features for RhythmModule
    HR = 0  # heart rate
    BRAD = 1  # bradycardia: hr < 60 bpm
    TACH = 2  # tachycardia: hr > 100 bpm
    SINUS = 3  # whether the rhythm is sinus: Each P should be +ve AND followed by a QRS
    RR_DIFF = 4  # RRmax - RRmin where RR is the RR interval

    # * Objective features for BlockModule
    PR_DUR = 5  # PR interval duration
    LPR = 6  # prolonged PR interval
    QRS_DUR = 7  # QRS duration
    LQRS = 8  # prolonged QRS duration

    # * Objective features for WPWModule
    LQRS_WPW = 9  # prolonged QRS duration for WPWModule
    SPR = 10  # short PR interval

    # * Objective features for STModule
    ST_AMP_I = 11  # average ST amplitude in lead I
    ST_AMP_II = 12  # average ST amplitude in lead II
    ST_AMP_III = 13  # average ST amplitude in lead III
    ST_AMP_AVR = 14  # average ST amplitude in lead aVR
    ST_AMP_AVL = 15  # average ST amplitude in lead aVL
    ST_AMP_AVF = 16  # average ST amplitude in lead aVF
    ST_AMP_V1 = 17  # average ST amplitude in lead V1
    ST_AMP_V2 = 18  # average ST amplitude in lead V2
    ST_AMP_V3 = 19  # average ST amplitude in lead V3
    ST_AMP_V4 = 20  # average ST amplitude in lead V4
    ST_AMP_V5 = 21  # average ST amplitude in lead V5
    ST_AMP_V6 = 22  # average ST amplitude in lead V6
    STE_I = 23  # ST elevation in lead I
    STE_II = 24  # ST elevation in lead II
    STE_III = 25  # ST elevation in lead III
    STE_AVR = 26  # ST elevation in lead aVR
    STE_AVL = 27  # ST elevation in lead aVL
    STE_AVF = 28  # ST elevation in lead aVF
    STE_V1 = 29  # ST elevation in lead V1
    STE_V2 = 30  # ST elevation in lead V2
    STE_V3 = 31  # ST elevation in lead V3
    STE_V4 = 32  # ST elevation in lead V4
    STE_V5 = 33  # ST elevation in lead V5
    STE_V6 = 34  # ST elevation in lead V6
    STD_I = 35  # ST depression in lead I
    STD_II = 36  # ST depression in lead II
    STD_III = 37  # ST depression in lead III
    STD_AVR = 38  # ST depression in lead aVR
    STD_AVL = 39  # ST depression in lead aVL
    STD_AVF = 40  # ST depression in lead aVF
    STD_V1 = 41  # ST depression in lead V1
    STD_V2 = 42  # ST depression in lead V2
    STD_V3 = 43  # ST depression in lead V3
    STD_V4 = 44  # ST depression in lead V4
    STD_V5 = 45  # ST depression in lead V5
    STD_V6 = 46  # ST depression in lead V6

    # * Objective features for QRModule
    PRWP = 47  # Poor R wave progression
    Q_DUR_I = 48  # Q wave duration in lead I
    Q_DUR_II = 49  # Q wave duration in lead II
    Q_DUR_III = 50  # Q wave duration in lead III
    Q_DUR_AVR = 51  # Q wave duration in lead aVR
    Q_DUR_AVL = 52  # Q wave duration in lead aVL
    Q_DUR_AVF = 53  # Q wave duration in lead aVF
    Q_DUR_V1 = 54  # Q wave duration in lead V1
    Q_DUR_V2 = 55  # Q wave duration in lead V2
    Q_DUR_V3 = 56  # Q wave duration in lead V3
    Q_DUR_V4 = 57  # Q wave duration in lead V4
    Q_DUR_V5 = 58  # Q wave duration in lead V5
    Q_DUR_V6 = 59  # Q wave duration in lead V6
    Q_AMP_I = 60  # Q wave amplitude in lead I
    Q_AMP_II = 61  # Q wave amplitude in lead II
    Q_AMP_III = 62  # Q wave amplitude in lead III
    Q_AMP_AVR = 63  # Q wave amplitude in lead aVR
    Q_AMP_AVL = 64  # Q wave amplitude in lead aVL
    Q_AMP_AVF = 65  # Q wave amplitude in lead aVF
    Q_AMP_V1 = 66  # Q wave amplitude in lead V1
    Q_AMP_V2 = 67  # Q wave amplitude in lead V2
    Q_AMP_V3 = 68  # Q wave amplitude in lead V3
    Q_AMP_V4 = 69  # Q wave amplitude in lead V4
    Q_AMP_V5 = 70  # Q wave amplitude in lead V5
    Q_AMP_V6 = 71  # Q wave amplitude in lead V6
    PATH_Q_I = 72  # Pathological Q wave in lead I
    PATH_Q_II = 73  # Pathological Q wave in lead II
    PATH_Q_III = 74  # Pathological Q wave in lead III
    PATH_Q_AVR = 75  # Pathological Q wave in lead aVR
    PATH_Q_AVL = 76  # Pathological Q wave in lead aVL
    PATH_Q_AVF = 77  # Pathological Q wave in lead aVF
    PATH_Q_V1 = 78  # Pathological Q wave in lead V1
    PATH_Q_V2 = 79  # Pathological Q wave in lead V2
    PATH_Q_V3 = 80  # Pathological Q wave in lead V3
    PATH_Q_V4 = 81  # Pathological Q wave in lead V4
    PATH_Q_V5 = 82  # Pathological Q wave in lead V5
    PATH_Q_V6 = 83  # Pathological Q wave in lead V6

    # * Objective features for PModule
    P_DUR_II = 84  # P wave duration in lead II
    P_AMP_II = 85  # P wave amplitude in lead II
    P_AMP_V1 = 86  # P wave amplitude in lead V1
    LP_II = 87  # P wave prolonged in lead II
    PEAK_P_II = 88  # large P wave amplitude in lead II
    PEAK_P_V1 = 89  # large P wave amplitude in lead V1

    # * Objective features for VHModule
    AGE = 90
    AGE_OLD = 91  # age > 30
    MALE = 92
    R_AMP_aVL = 93  # R wave amplitude in lead aVL
    R_AMP_V1 = 94  # R wave amplitude in lead V1
    R_AMP_V6 = 95  # R wave amplitude in lead V6
    S_AMP_V1 = 96  # S wave's absolute amplitude in lead V1
    S_AMP_V3 = 97  # S wave's absolute amplitude in lead V2
    S_AMP_V5 = 98  # S wave's absolute amplitude in lead V3
    S_AMP_V6 = 99  # S wave's absolute amplitude in lead V6
    RS_RATIO_V1 = 100  # R/S ratio in lead V1
    RS_RATIO_V5 = 102  # R/S ratio in lead V5
    RS_RATIO_V6 = 103  # R/S ratio in lead V6
    LVH_L1_OLD = 104  # LVH L1 criteria for "old" people (age > 30)
    LVH_L1_YOUNG = 105  # LVH L1 criteria for "young" people (age <= 30)
    LVH_L2_MALE = 106  # LVH L2 criteria for male
    LVH_L2_FEMALE = 107  # LVH L2 criteria for female
    PEAK_R_V1 = 108  # large R wave amplitude in lead V1
    DEEP_S_V5 = 109  # deep S wave in lead V5
    DEEP_S_V6 = 110  # deep S wave in lead V6
    DOM_R_V1 = 111  # dominant R wave in lead V1
    DOM_S_V5 = 112  # dominant S wave in lead V5
    DOM_S_V6 = 113  # dominant S wave in lead V6

    # * Objective features for TModule
    T_AMP_I = 114  # T wave amplitude in lead I
    T_AMP_II = 115  # T wave amplitude in lead II
    T_AMP_V1 = 116  # T wave amplitude in lead V1
    T_AMP_V2 = 117  # T wave amplitude in lead V2
    T_AMP_V3 = 118  # T wave amplitude in lead V3
    T_AMP_V4 = 119  # T wave amplitude in lead V4
    T_AMP_V5 = 120  # T wave amplitude in lead V5
    T_AMP_V6 = 121  # T wave amplitude in lead V6
    INVT_I = 122  # inverted T wave in lead I
    INVT_II = 123  # inverted T wave in lead II
    INVT_V1 = 124  # inverted T wave in lead V1
    INVT_V2 = 125  # inverted T wave in lead V2
    INVT_V3 = 126  # inverted T wave in lead V3
    INVT_V4 = 127  # inverted T wave in lead V4
    INVT_V5 = 128  # inverted T wave in lead V5
    INVT_V6 = 129  # inverted T wave in lead V6

    # * Objective features for AxisModule
    QRS_SUM_I = 130  # sum/integral of QRS complex in lead I
    QRS_SUM_aVF = 131  # sum/integral of QRS complex in lead aVF
    POS_QRS_I = 132  # QRS complex is net positive in lead I
    POS_QRS_aVF = 133  # QRS complex is net positive in lead aVF
    NORM_AXIS = 134  # normal axis
    LAD = 135  # left axis deviation
    RAD = 136  # right axis deviation


def keys_to_vector(keys: list[str] | pd.Series, enum_type: Diagnosis | Feature) -> torch.Tensor:
    vector = torch.zeros(len(enum_type))
    for label in keys:
        vector[enum_type[label].value] = 1
    return vector


def zero_vec(vec_type: Diagnosis | Feature, extra_dim: list[int] | None = None):
    # vec = torch.zeros(len(Feature) + len(Diagnosis), dtype=torch.float32)
    if not extra_dim:
        vec = torch.zeros(len(vec_type), dtype=torch.float32)
    else:
        vec = torch.zeros([*extra_dim, len(vec_type)], dtype=torch.float32)
    return vec


def fill_vec(vec: torch.Tensor, values: torch.Tensor, keys: list[str], vec_type: Diagnosis | Feature):
    """
    Fill the vector with the given values (values' corresponding keys are also given)
    """
    if vec.ndim == 1:
        vec[[vec_type[key].value for key in keys]] = values.type(torch.float32)
    elif vec.ndim == 2:
        vec[:, [vec_type[key].value for key in keys]] = values.type(torch.float32)
    else:
        raise ValueError(f"vec must be 1 or 2 dimensional, got {vec.ndim}")
    return vec


def pad_vector(values: torch.Tensor, keys: list[str], enum_type: Diagnosis | Feature):
    """
    Pad the diagnoses/feature values with zeros
    such that the they become corresponding vectors,
    whose last dimension match the total number of elements in the ``enum_type``
    """

    extra_dim = list(values.shape)[:-1]
    empty_vec = zero_vec(enum_type, extra_dim)
    padded_vec = fill_vec(empty_vec, values, keys, enum_type)
    return padded_vec


def get_by_str(vec: torch.Tensor, keys: list[str], vec_type: Diagnosis | Feature):
    """
    Get features/diagnoses by the given list of string keys
    """
    if not keys:
        return None

    idx = vec_type[keys[0]].value if len(keys) == 1 else [vec_type[key].value for key in keys]
    if vec.ndim == 1:
        return vec[idx]

    # if it is batched vectors
    return vec[:, idx]
