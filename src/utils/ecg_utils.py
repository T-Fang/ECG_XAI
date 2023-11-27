import warnings
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
import neurokit2 as nk
from src.basic.constants import SAMPLING_RATE, DURATION, MIN_RULE_SATISFACTION_PERCENTAGE, SIGNAL_LEN, T_AMP_THRESH, P_AMP_THRESH


def get_rpeaks(cleaned: NDArray[np.float32]) -> NDArray[np.int64]:
    """
    Get R peaks from the cleaned ECG lead

    Parameters
    ----------
    cleaned :
        The cleaned ECG lead.

    Returns
    -------
    rpeaks :
        The R peaks of the lead.
    """
    _, rpeaks_dict = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE)
    # print(rpeaks_dict['ECG_R_Peaks'])
    return rpeaks_dict['ECG_R_Peaks']


def get_all_rpeaks(cleaned: NDArray[np.float32]) -> list[NDArray[np.int32]]:
    """
    Get all R peaks from the cleaned ECG signal

    Parameters
    ----------
    cleaned :
        The cleaned ECG signal (might be multi-leads).

    Returns
    -------
    all_rpeaks :
        A list of R peaks of all leads.
    """
    if cleaned.ndim == 1:
        return [get_rpeaks(cleaned)]

    all_rpeaks = [get_rpeaks(cleaned_lead) for cleaned_lead in cleaned]
    return all_rpeaks


def get_delineation(cleaned: NDArray[np.float32], rpeaks: NDArray[np.int64], method: str = 'dwt') -> dict[str, list]:
    """
    Get the delineation of the cleaned ECG lead

    Parameters
    ----------
    cleaned :
        The cleaned ECG lead.
    rpeaks :
        The R peaks of the lead.
    method :
        The method to use for delineation. Default is 'dwt'.

    Returns
    -------
    delineation :
        The delineation of the lead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # np.set_printoptions(threshold=np.inf)
        # print(cleaned)
        _, delineation = nk.ecg_delineate(ecg_cleaned=cleaned,rpeaks=rpeaks, sampling_rate=SAMPLING_RATE, method=method)


    for feat_name, feat_indices in delineation.items():
        # print(feat_indices)
        delineation[feat_name] = np.array(feat_indices)
    # print(delineation['ECG_Q_Peaks'])
    # Add rpeaks info to the delineation dict
    delineation['ECG_R_Peaks'] = rpeaks
    return delineation


def get_all_delineations(cleaned: NDArray[np.float32],
                         all_rpeaks: list[NDArray[np.int32]],
                         method: str = 'dwt') -> list[dict[str, NDArray[np.int32]]]:
    """
    Get all delineations from the cleaned ECG signal

    Parameters
    ----------
    cleaned : NDArray[np.float32]
        The cleaned ECG signal (might be multi-leads).
    all_rpeaks : list[NDArray[np.int32]]
        A list of R peaks of all leads.
    method : str
        The method to use for delineation. Default is 'dwt'.

    Returns
    -------
    all_delineations :
        A list of delineations of all leads.
    """
    if cleaned.ndim == 1:
        return [get_delineation(cleaned, all_rpeaks[0], method=method)]

    all_delineations = []
    for i in range(cleaned.shape[0]):
        # print(get_delineation(cleaned[i], all_rpeaks[i], method=method))
        all_delineations.append(get_delineation(cleaned[i], all_rpeaks[i], method=method))

    return all_delineations


def check_inverted_wave(wave_name: str, lead_signal: NDArray[np.float32], delineation: dict[str,
                                                                                            NDArray[np.int32]]) -> bool:
    """
    Check if the wave is inverted in the lead

    Parameters
    ----------
    wave_name :
        The name of the wave to check. can be either 'P' or 'T'
    lead_signal :
        The ECG signal of one lead.
    delineation :
        A dict containing the delineation information.

    Returns
    -------
    is_wave_inverted : bool
        whether the lead has inverted wave.
    """

    wave_onsets = delineation[f'ECG_{wave_name}_Onsets']
    wave_peaks = delineation[f'ECG_{wave_name}_Peaks']
    wave_offsets = delineation[f'ECG_{wave_name}_Offsets']

    assert len(wave_onsets) == len(wave_peaks) == len(wave_offsets)
    n_wave = len(wave_onsets)
    if len(wave_onsets) == 0:
        return False

    wave_amp_thresh = T_AMP_THRESH if wave_name == 'T' else P_AMP_THRESH

    def check_all_wave_peaks():
        peaks_without_nan = wave_peaks[~np.isnan(wave_peaks)].astype(int)
        all_checks = lead_signal[peaks_without_nan] > 0
        if len(all_checks) == 0:
            return False
        return all_checks.mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    def check_onset_offset():
        # the detected wave cannot be too flat
        all_checks = []
        for i in range(n_wave):
            wave_peak = wave_peaks[i]
            wave_onset = wave_onsets[i]
            wave_offset = wave_offsets[i]
            if not (np.isnan(wave_onset) or np.isnan(wave_peak) or np.isnan(wave_offset)):
                peak_minus_thresh = lead_signal[np.int(wave_peak)] - wave_amp_thresh
                check = peak_minus_thresh > lead_signal[np.int(wave_onset)] and peak_minus_thresh > lead_signal[np.int(
                    wave_offset)]

                all_checks.append(check)
        if len(all_checks) == 0:
            return False
        return np.array(all_checks).mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    is_inverted = not (check_all_wave_peaks() and check_onset_offset())
    return is_inverted


def check_all_inverted_waves(wave_name: str, cleaned: NDArray[np.float32],
                             delineations: list[dict[str, NDArray[np.int32]]]) -> NDArray[np.bool_]:
    """
    For all leads, check if the wave is inverted

    Parameters
    ----------
    wave_name :
        The name of the wave to check. can be either 'P' or 'T'.
    cleaned : NDArray[np.float32]
        The cleaned ECG signal of all leads.
    delineations : list[dict[str, NDArray[np.int32]]]
        A list of DataFrames of the same length as a cleaned lead containing the delineation information.

    Returns
    -------
    are_waves_inverted : NDArray[np.bool_]
        a boolean array showing which leads have inverted waves.
    """
    are_waves_inverted = []
    for i in range(cleaned.shape[0]):
        are_waves_inverted.append(check_inverted_wave(wave_name, cleaned[i], delineations[i]))
    return np.array(are_waves_inverted)


def delineation2signal(delineation: dict[str, NDArray[np.int32]]):
    """
    Convert the delineation to signals where the occurrences of peaks, onsets and offsets marked as “1” in a list of zeros.
    The signals are then combined into a DataFrame
    """
    delineation_signals = {}

    for feature_name in delineation.keys():

        signal = np.zeros(SIGNAL_LEN)
        feature_without_nan = delineation[feature_name][~np.isnan(delineation[feature_name])].astype(int)
        signal[feature_without_nan] = 1
        delineation_signals[feature_name] = signal
    delineation_signals = pd.DataFrame(delineation_signals)
    return delineation_signals


# def all_delineations2signal_df(delineations: list[dict[str, NDArray[np.int32]]]):
#     """
#     Convert the delineations to signals where the occurrences of peaks, onsets and offsets marked as “1” in a list of zeros.
#     The signals for each are then combined into a dataframe
#     The dataframes are then put into a list
#     """
#     delineation_signals = []
#     for i in range(len(delineations)):
#         delineation_signals.append(delineation2signal_df(delineations[i]))
#     return delineation_signals


########################################
# Visualization
########################################
def custom_ecg_delineate_plot(
        ecg_signal,
        delineation,
        sampling_rate=SAMPLING_RATE,
        window_range=(0.5, 1.5),
) -> Figure:

    window_start = int(window_range[0] * sampling_rate)
    window_end = int(window_range[1] * sampling_rate)

    delineation = delineation2signal(delineation)
    delineation.rename(
        {
            'ECG_P_Peaks': 'P_Peaks',
            'ECG_P_Onsets': 'P_Onsets',
            'ECG_P_Offsets': 'P_Offsets',
            'ECG_Q_Peaks': 'Q_Peaks',
            'ECG_R_Onsets': 'QRS_Onsets',
            'ECG_R_Peaks': 'R_Peaks',
            'ECG_R_Offsets': 'QRS_Offsets',
            'ECG_S_Peaks': 'S_Peaks',
            'ECG_T_Peaks': 'T_Peaks',
            'ECG_T_Onsets': 'T_Onsets',
            'ECG_T_Offsets': 'T_Offsets'
        },
        axis='columns',
        inplace=True)

    signal = pd.DataFrame({"Signal": ecg_signal})
    time = pd.DataFrame({"Time": np.arange(0, ecg_signal.shape[0]) / sampling_rate})
    data = pd.concat([signal, delineation, time], axis=1)
    data = data.iloc[window_start:window_end]

    fig, ax = plt.subplots()
    ax.plot(data.Time, data.Signal, color="grey", alpha=0.2)

    colors = cm.rainbow(np.linspace(0, 1, len(delineation.columns.values)))

    def get_marker(feature_type: str):
        if 'Onsets' in feature_type:
            return ">"
        elif 'Offsets' in feature_type:
            return "<"
        else:
            return "o"

    for i, feature_type in enumerate(delineation.columns.values):
        event_data = data[data[feature_type] == 1.0]
        ax.scatter(event_data.Time,
                   event_data.Signal,
                   label=feature_type,
                   alpha=0.5,
                   s=50,
                   marker=get_marker(feature_type),
                   color=colors[i])
        ax.legend()

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    return fig


# Plot heart rate.
def plot_heart_rate(heart_rates):
    x_axis = np.linspace(0, DURATION, SAMPLING_RATE * DURATION)

    fig, ax = plt.subplots()
    ax.set_title("Heart Rate")
    ax.set_ylabel("Beats per minute (bpm)")

    ax.plot(x_axis, heart_rates, color="#FF5722", label="Rate", linewidth=1.5)
    rate_mean = heart_rates.mean()
    ax.axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")

    ax.legend(loc="upper right")


def analyze_hrv(rpeaks):
    rr = np.diff(rpeaks) / SAMPLING_RATE * 1000
    hr = 60 * SAMPLING_RATE / rr

    hrv_analysis = {}
    hrv_analysis['Mean RR (ms)'] = np.mean(rr)
    hrv_analysis['STD RR/SDNN (ms)'] = np.std(rr)
    hrv_analysis['Mean HR (Kubios\' style) (beats/min)'] = 30000 / np.mean(rr)
    hrv_analysis['Mean HR (beats/min)'] = np.mean(hr)
    hrv_analysis['STD HR (beats/min)'] = np.std(hr)
    hrv_analysis['Min HR (beats/min)'] = np.min(hr)
    hrv_analysis['Max HR (beats/min)'] = np.max(hr)
    hrv_analysis['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))  # Normal method to get HRV
    hrv_analysis['NNxx'] = np.sum(np.abs(np.diff(rr)) > 50) * 1
    hrv_analysis['pNNxx (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50) * 1) / len(rr)
    return hrv_analysis
