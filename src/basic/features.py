import torch
import pandas as pd
from enum import Enum
class Feature(Enum):
    AGE = 0
    AGE_OLD = 1
    ARRH = 2
    BRAD = 3
    DEEP_S = 4
    DOM_R_V1 = 5
    DOM_S = 6
    HR = 7
    INVT = 8
    INVT_V1 = 9
    INVT_V2 = 10
    INVT_V3 = 11
    INVT_V5 = 12
    INVT_V6 = 13
    LBBB = 14
    LMI_II = 15
    LMI_V1 = 16
    LPR = 17
    LP_II = 18
    LQRS = 19
    LQRS = 20
    LVH_L1_OLD = 21
    LVH_L1_YOUNG = 22
    LVH_L2_FEMALE = 23
    LVH_L2_MALE = 24
    MALE = 25
    PATH_Q_I = 26
    PATH_Q_II = 27
    PATH_Q_III = 28
    PATH_Q_V1 = 29
    PATH_Q_V2 = 30
    PATH_Q_V3 = 31
    PATH_Q_V4 = 32
    PATH_Q_V5 = 33
    PATH_Q_V6 = 34
    PATH_Q_aVF = 35
    PATH_Q_aVL = 36
    PEAK_P_II = 37
    PEAK_P_V1 = 38
    PEAK_R_V1 = 39
    POS_QRS_I = 40
    POS_QRS_aVF = 41
    PR_DUR = 42
    PR_DUR = 43
    QRS_DUR = 44
    QRS_DUR = 45
    QRS_SUM_I = 46
    QRS_SUM_aVF = 47
    Q_AMP_I = 48
    Q_AMP_II = 49
    Q_AMP_III = 50
    Q_AMP_V1 = 51
    Q_AMP_V2 = 52
    Q_AMP_V3 = 53
    Q_AMP_V4 = 54
    Q_AMP_V5 = 55
    Q_AMP_V6 = 56
    Q_AMP_aVF = 57
    Q_AMP_aVL = 58
    Q_DUR_I = 59
    Q_DUR_II = 60
    Q_DUR_III = 61
    Q_DUR_V1 = 62
    Q_DUR_V2 = 63
    Q_DUR_V3 = 64
    Q_DUR_V4 = 65
    Q_DUR_V5 = 66
    Q_DUR_V6 = 67
    Q_DUR_aVF = 68
    Q_DUR_aVL = 69
    RAD = 70
    RBBB = 71
    RR_DIFF = 72
    RS_RATIO_V1 = 73
    RS_RATIO_V5 = 74
    RS_RATIO_V6 = 75
    R_AMP_V = 76
    R_AMP_V1 = 77
    R_AMP_aV = 78
    SPR = 79
    STD_I = 80
    STD_II = 81
    STD_II = 82
    STD_III = 83
    STD_III = 84
    STD_V1 = 85
    STD_V1 = 86
    STD_V2 = 87
    STD_V2 = 88
    STD_V3 = 89
    STD_V3 = 90
    STD_V4 = 91
    STD_V5 = 92
    STD_V5 = 93
    STD_V6 = 94
    STD_V6 = 95
    STD_aVF = 96
    STD_aVF = 97
    STD_aVL = 98
    STD_aVL = 99
    STD_aVR = 100
    STE_I = 101
    STE_I = 102
    STE_II = 103
    STE_II = 104
    STE_III = 105
    STE_III = 106
    STE_V1 = 107
    STE_V1 = 108
    STE_V2 = 109
    STE_V2 = 110
    STE_V3 = 111
    STE_V3 = 112
    STE_V4 = 113
    STE_V4 = 114
    STE_V5 = 115
    STE_V5 = 116
    STE_V6 = 117
    STE_V6 = 118
    STE_aVF = 119
    STE_aVF = 120
    STE_aVL = 121
    STE_aVL = 122
    STE_aVR = 123
    ST_AMP_I = 124
    ST_AMP_II = 125
    ST_AMP_III = 126
    ST_AMP_V1 = 127
    ST_AMP_V2 = 128
    ST_AMP_V3 = 129
    ST_AMP_V4 = 130
    ST_AMP_V5 = 131
    ST_AMP_V6 = 132
    ST_AMP_aVF = 133
    ST_AMP_aVL = 134
    S_AMP_V5 = 135
    S_AMP_V6 = 136
    TACH = 137
    T_AMP_I = 138
    T_AMP_II = 139
    T_AMP_III = 140
    T_AMP_V1 = 141
    T_AMP_V2 = 142
    T_AMP_V3 = 143
    T_AMP_V4 = 144
    T_AMP_V5 = 145
    T_AMP_V6 = 146
    T_AMP_aVF = 147
    T_AMP_aVL = 148
    T_AMP_aVR = 149
    _AMP_V1 = 150
    _AMP_V3 = 151
