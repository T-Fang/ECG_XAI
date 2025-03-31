import torch
import pandas as pd
from enum import Enum
class Diagnosis(Enum):
    NORM = 0
    AFIB = 1
    AFLT = 2
    AMI = 3
    AVB = 4
    IMI = 5
    IVCD = 6
    LAD = 7
    LAE = 8
    LAFB = 9
    LBBB = 10
    LMI = 11
    LPFB = 12
    LVH = 13
    NORM_AXIS = 14
    RAD = 15
    RAE = 16
    RBBB = 17
    RVH = 18
    SARRH = 19
    SBRAD = 20
    SR = 21
    STACH = 22
    WPW = 23
