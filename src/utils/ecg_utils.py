import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.basic.constants import SAMPLING_RATE


def custom_ecg_delineate_plot(
        ecg_signal,
        rpeaks=None,
        ecg_feat=None,
        sampling_rate=SAMPLING_RATE,
        window_range=(0.5, 1.5),
):

    window_start = int(window_range[0] * sampling_rate)
    window_end = int(window_range[1] * sampling_rate)
    time = pd.DataFrame({"Time": np.arange(0, ecg_signal.shape[0]) / sampling_rate})

    if rpeaks is not None:
        rpeak_loc = rpeaks['ECG_R_Peaks']
        rpeak_feat = np.zeros(ecg_signal.shape[0])
        rpeak_feat[rpeak_loc] = 1
        rpeak_feat = pd.DataFrame({"R_Peaks": rpeak_feat})
        ecg_feat = pd.concat([ecg_feat, rpeak_feat], axis=1)

    ecg_feat.rename(
        {
            'ECG_P_Peaks': 'P_Peaks',
            'ECG_P_Onsets': 'P_Onsets',
            'ECG_P_Offsets': 'P_Offsets',
            'ECG_Q_Peaks': 'Q_Peaks',
            'ECG_R_Onsets': 'QRS_Onsets',
            'ECG_R_Offsets': 'QRS_Offsets',
            'ECG_S_Peaks': 'S_Peaks',
            'ECG_T_Peaks': 'T_Peaks',
            'ECG_T_Onsets': 'T_Onsets',
            'ECG_T_Offsets': 'T_Offsets'
        },
        axis='columns',
        inplace=True)

    signal = pd.DataFrame({"Signal": list(ecg_signal)})
    data = pd.concat([signal, ecg_feat, time], axis=1)
    data = data.iloc[window_start:window_end]

    fig, ax = plt.subplots()
    ax.plot(data.Time, data.Signal, color="grey", alpha=0.2)

    colors = cm.rainbow(np.linspace(0, 1, len(ecg_feat.columns.values)))
    for i, feature_type in enumerate(ecg_feat.columns.values):
        event_data = data[data[feature_type] == 1.0]
        ax.scatter(event_data.Time, event_data.Signal, label=feature_type, alpha=0.5, s=50, color=colors[i])
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    return fig
