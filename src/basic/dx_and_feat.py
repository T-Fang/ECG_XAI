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


def pad_vector(vec: torch.Tensor, keys: list[str], enum_type: Diagnosis | Feature):
    """
    Pad the prediction/feature vector with zeros
    such that the last dimension match the total number of elements in the ``enum_type``

    vec: a single prediction/feature vector or batched prediction/feature vectors
    keys: list of str showing corresponding labels of the ``vec``
    """

    output_shape = list(vec.shape)
    if vec.dim() == 1:
        output_shape[0] = len(enum_type)
        output = torch.zeros(output_shape)
        output.to(vec)
        output[[enum_type[label].value for label in keys]] = vec
    elif vec.dim() == 2:
        output_shape[1] = len(enum_type)
        output = torch.zeros(output_shape)
        output.to(vec)
        output[:, [enum_type[label].value for label in keys]] = vec
    else:
        raise ValueError(f"vec must be 1 or 2 dimensional, got {vec.dim()}")
    return output


def zero_mid_output():
    return torch.zeros(len(Feature) + len(Diagnosis))


# TODO: allow batched version mid_output
def fill_mid_output(mid_output: torch.Tensor, values: torch.Tensor, keys: list[str], enum_type: Diagnosis | Feature):
    """
    Fill the middle output with the given values (values' corresponding keys are also given)
    """
    offset = 0 if enum_type == Diagnosis else len(Diagnosis)
    mid_output[[enum_type[key].value for key in keys] + offset] = values
    return mid_output


def get_mid_output_by_str(mid_output: torch.Tensor, keys: list[str], enum_type: Diagnosis | Feature):
    """
    Get the middle output by the given list of string keys
    """
    offset = 0 if enum_type == Diagnosis else len(Diagnosis)
    return mid_output[[enum_type[key].value for key in keys] + offset]
