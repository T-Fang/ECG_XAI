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
REG_LOSS_WEIGHT = 1

# ECG rule related constants
MIN_RULE_SATISFACTION_PERCENTAGE = 0.8
DEVIATION_TOL = 0.05

# ECG thresholds
T_AMP_THRESH = 0.1
P_AMP_THRESH = 0.07
LPR_THRESH = 200
SHORTPR_THRESH = 120
LQRS_THRESH = 120

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
