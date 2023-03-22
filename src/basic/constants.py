import os
import platform

# File Names
CURRENT_OS = platform.system()
PROJECT_PATH = '/home/ftian/storage/ECG_XAI/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer Science/Archive/FYP/ECG_XAI/'
PTBXL_PATH = os.path.join(PROJECT_PATH, 'data/ptbxl/')
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed/')
TRAIN_LOG_PATH = os.path.join(PROJECT_PATH, 'logs/train/')

# General constants
MANUAL_SEED = 5

# Training, Validating and Testing
PROB_THRESHOLD = 0.5
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]
TEST_FOLDS = [9, 10]
FEAT_LOSS_WEIGHT = 1
DELTA_LOSS_WEIGHT = 1

LOG_INTERVAL = 50
CHECK_VAL_EVERY_N_EPOCH = 1

# ECG rule related constants
MIN_RULE_SATISFACTION_PERCENTAGE = 0.8
DEVIATION_TOL = 0.05
T_AMP_THRESH = 0.1
P_AMP_THRESH = 0.07

# ECG thresholds for RhythmModule
RHYTHM_LEADS = ['II', 'V1']
BRAD_THRESH = 60
TACH_THRESH = 100
SARRH_THRESH = 120

# ECG thresholds for BlockModule
BLOCK_LEADS = ['V1', 'V2', 'V5', 'V6']
LPR_THRESH = 200
LQRS_THRESH = 120

# ECG thresholds for WPWModule
LQRS_WPW_THRESH = 110
SPR_THRESH = 120

# ECG thresholds for STModule
STE_THRESH = 0.1
STD_THRESH = -0.1

# ECG thresholds for QRModule
Q_DUR_THRESH = 40
Q_AMP_THRESH = {
    'I': -0.15,
    'III': -0.7,
    'aVL': -0.7,
    'II': -0.3,
    'aVF': -0.3,
    'aVR': -0.3,
    'V1': -0.3,
    'V2': -0.3,
    'V3': -0.3,
    'V4': -0.3,
    'V5': -0.3,
    'V6': -0.3
}
PRWP_LEADS = [
    'V1',
    'V2',
    'V3',
    'V4',
]
R_AMP_THRESH = {
    'V1': {
        'young': (0, 1.5),
        'middle': (0, 0.8),
        'old': (0, 0.6)
    },
    'V2': (0.02, 1.2),
    'V3': (0.1, 2),
    'V4': (0.1, 2)
}

# ECG thresholds for PModule
P_LEADS = ['II', 'V1']
LP_THRESH_II = 110
PEAK_P_THRESH_II = 0.25
PEAK_P_THRESH_V1 = 0.15

# ECG thresholds for VHModule
VH_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVL']
AGE_OLD_THRESH = 30
LVH_L1_OLD_THRESH = 3.5
LVH_L1_YOUNG_THRESH = 4
LVH_L2_MALE_THRESH = 2.4
LVH_L2_FEMALE_THRESH = 1.8
PEAK_R_THRESH = 0.7
DEEP_S_THRESH = 0.7
DOM_R_THRESH = 1
DOM_S_THRESH = 1

# ECG thresholds for TModule
T_LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
INVT_THRESH = 0

# ECG thresholds for AxisModule
AXIS_LEADS = ['I', 'aVF']
POS_QRS_THRESH = 0

# ECG metadata related constants
SAMPLING_RATE = 500
DURATION = 10
SIGNAL_LEN = 5000  # 10 seconds x 500 Hz
MS_PER_INDEX = 2
N_LEADS = 12
CHOSEN_METADATA = [
    'age', 'sex', 'scp_codes', 'heart_axis', 'infarction_stadium1', 'strat_fold', 'filename_lr', 'filename_hr'
]
ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
STD_LEADS = ['II', 'III', 'aVL', 'aVF']
LEAD_TO_INDEX = {
    'I': 0,
    'II': 1,
    'III': 2,
    'aVR': 3,
    'aVL': 4,
    'aVF': 5,
    'V1': 6,
    'V2': 7,
    'V3': 8,
    'V4': 9,
    'V5': 10,
    'V6': 11
}
