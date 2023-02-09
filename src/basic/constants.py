import os

# File Names
# PROJECT_PATH = '/home/ftian/storage/ECG_XAI/'
PROJECT_PATH = '/Users/tf/Computer Science/Archive/FYP/ECG_XAI/'
PTBXL_PATH = os.path.join(PROJECT_PATH, 'data/ptbxl/')

# General constants
MANUAL_SEED = 5

# Training, Validating and Testing
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]
TEST_FOLDS = [9, 10]

# ECG related constants
SAMPLING_RATE = 500
CHOSEN_METADATA = [
    'age', 'sex', 'scp_codes', 'heart_axis', 'infarction_stadium1', 'strat_fold', 'filename_lr', 'filename_hr'
]
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
