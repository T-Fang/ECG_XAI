import os

# File Names
PROJECT_PATH = '/home/ftian/storage/ECG_XAI/'
PTBXL_PATH = os.path.join(PROJECT_PATH, 'data/ptbxl/')

# General constants
MANUAL_SEED = 5

# ECG related constants
SAMPLING_RATE = 500
CHOSEN_METADATA = [
    'age', 'sex', 'scp_codes', 'heart_axis', 'infarction_stadium1', 'strat_fold', 'filename_lr', 'filename_hr'
]

# Training, Validating and Testing
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
VAL_FOLDS = [7, 8]
TEST_FOLDS = [9, 10]
