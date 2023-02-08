import os
import wfdb
import numpy as np
from src.basic.constants import SAMPLING_RATE, PTBXL_PATH


def load_raw_data(df, path):
    if SAMPLING_RATE == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
