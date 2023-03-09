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
    #######################
    # Boolean/Probability features #
    #######################
    NORMAXIS = 0
    LAD = 1
    RAD = 2
    HVOLT = 3
    INVT = 4
    LOWT = 5
    LPR = 6  # LPR: prolonged PR
    LVOLT = 7
    QWAVE = 8
    STD = 9
    STE = 10
    VCLVH = 11
    # below are unlabelled boolean features
    LQRS = 12  # prolonged QRS
    LRWPT = 13  # prolonged R-wave peak time
    WIDES = 14  # wide S wave
    HYPT = 15  # hyper-acute T wave
    SHORTPR = 16  # short PR interval
    HPVOLT = 17  # high P wave voltage
    LP = 18  # prolonged P wave duration

    #######################
    # Numerical features #
    #######################
    HR = 19  # heart rate
    PRDUR = 20  # PR interval duration
    QRSDUR = 21  # QRS duration


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
