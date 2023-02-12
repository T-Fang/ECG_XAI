import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import neurokit2 as nk
from src.basic.constants import SAMPLING_RATE, DURATION, MIN_RULE_SATISFACTION_PERCENTAGE, T_THRESH, P_THRESH


def get_rpeaks(cleaned: np.ndarray) -> np.ndarray:
    """
    Get R peaks from the cleaned ECG lead

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG lead.

    Returns
    -------
    rpeaks : np.ndarray
        The R peaks of the lead.
    """
    _, rpeaks_dict = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE)
    return rpeaks_dict['ECG_R_Peaks']


def get_all_rpeaks(cleaned: np.ndarray) -> list:
    """
    Get all R peaks from the cleaned ECG signal

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG signal (might be multi-leads).

    Returns
    -------
    all_rpeaks : list
        A list of R peaks of all leads.
    """
    if cleaned.ndim == 1:
        return [get_rpeaks(cleaned)]

    all_rpeaks = [get_rpeaks(cleaned_lead) for cleaned_lead in cleaned]
    return all_rpeaks


def get_delineation(cleaned: np.ndarray, rpeaks: np.ndarray, method: str = 'dwt') -> dict:
    """
    Get the delineation of the cleaned ECG lead

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG lead.
    rpeaks : np.ndarray
        The R peaks of the lead.
    method : str
        The method to use for delineation. Default is 'dwt'.

    Returns
    -------
    delineation : dict
        The delineation of the lead.
    """
    delineation, _ = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=SAMPLING_RATE, method=method)
    return delineation


def get_all_delineations(cleaned: np.ndarray, all_rpeaks: list[np.ndarray], method: str = 'dwt') -> list[dict]:
    """
    Get all delineations from the cleaned ECG signal

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG signal (might be multi-leads).
    all_rpeaks : list
        A list of R peaks of all leads.
    method : str
        The method to use for delineation. Default is 'dwt'.

    Returns
    -------
    all_delineations : list[dict]
        A list of delineations of all leads.
    """
    if cleaned.ndim == 1:
        return [get_delineation(cleaned, all_rpeaks[0], method=method)]

    all_delineations = []
    for i in range(cleaned.shape[0]):
        all_delineations.append(get_delineation(cleaned[i], all_rpeaks[i], method=method))

    return all_delineations


def check_inverted_T(lead_signal: np.ndarray, delineation: pd.DataFrame) -> bool:
    """
    Check if the T wave is inverted in the lead

    Parameters
    ----------
    lead_signal :
        The ECG signal of one lead.
    delineation : DataFrame
        A DataFrame of the same length as the ``lead_signal`` containing the delineation information.

    Returns
    -------
    is_T_inverted : bool
        whether the lead has inverted T wave.
    """
    T_peaks = delineation['ECG_T_Peaks'].to_numpy().nonzero()[0]
    T_onsets = delineation['ECG_T_Onsets'].to_numpy().nonzero()[0]
    T_offsets = delineation['ECG_T_Offsets'].to_numpy().nonzero()[0]
    num_T = min(len(T_peaks), len(T_onsets), len(T_offsets))
    if num_T == 0:
        return False

    def check_all_T_peaks():
        all_checks = lead_signal[T_peaks] > 0
        return all_checks.mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    def check_onset_offset():
        # the detected T wave cannot be too flat
        all_checks = []
        for i in range(num_T):
            t_peak = T_peaks[i]
            t_onset = T_onsets[i]
            t_offset = T_offsets[i]

            flag1 = lead_signal[t_peak] - T_THRESH > lead_signal[t_onset]
            flag2 = lead_signal[t_peak] - T_THRESH > lead_signal[t_offset]

            all_checks.append(flag1 and flag2)
        return np.array(all_checks).mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    # TODO: order different check to utilize short-circuit evaluation
    is_T_inverted = not (check_all_T_peaks() and check_onset_offset())
    return is_T_inverted


def check_all_inverted_T(cleaned: np.ndarray, delineations: list) -> np.ndarray:
    """
    Check if the T wave is inverted in all leads

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG signal of all leads.
    delineations : list
        A list of DataFrames of the same length as a cleaned lead containing the delineation information.

    Returns
    -------
    are_T_inverted : np.ndarray
        a boolean array showing which leads have inverted T waves.
    """
    are_T_inverted = []
    for i in range(cleaned.shape[0]):
        are_T_inverted.append(check_inverted_T(cleaned[i], delineations[i]))
    return np.array(are_T_inverted)


def check_inverted_P(lead_signal: np.ndarray, delineation: pd.DataFrame) -> bool:
    """
    Check if the P wave is inverted in the lead

    Parameters
    ----------
    lead_signal :
        The ECG signal of one lead.
    delineation : DataFrame
        A DataFrame of the same length as the ``lead_signal`` containing the delineation information.

    Returns
    -------
    is_P_inverted : bool
        whether the lead has inverted P wave.
    """
    P_peaks = delineation['ECG_P_Peaks'].to_numpy().nonzero()[0]
    P_onsets = delineation['ECG_P_Onsets'].to_numpy().nonzero()[0]
    P_offsets = delineation['ECG_P_Offsets'].to_numpy().nonzero()[0]
    num_P = min(len(P_peaks), len(P_onsets), len(P_offsets))
    if num_P == 0:
        return False

    def check_all_P_peaks():
        all_checks = lead_signal[P_peaks] > 0
        return all_checks.mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    def check_onset_offset():
        # the detected P wave cannot be too flat
        all_checks = []
        for i in range(num_P):
            p_peak = P_peaks[i]
            p_onset = P_onsets[i]
            p_offset = P_offsets[i]

            flag1 = lead_signal[p_peak] - P_THRESH > lead_signal[p_onset]
            flag2 = lead_signal[p_peak] - P_THRESH > lead_signal[p_offset]

            all_checks.append(flag1 and flag2)
        return np.array(all_checks).mean() >= MIN_RULE_SATISFACTION_PERCENTAGE

    is_P_inverted = not (check_all_P_peaks() and check_onset_offset())
    return is_P_inverted


def check_all_inverted_P(cleaned: np.ndarray, delineations: list) -> np.ndarray:
    """
    Check if the P wave is inverted in all leads

    Parameters
    ----------
    cleaned : np.ndarray
        The cleaned ECG signal of all leads.
    delineations : list
        A list of DataFrames of the same length as a cleaned lead containing the delineation information.

    Returns
    -------
    are_P_inverted : np.ndarray
        a boolean array showing which leads have inverted P waves.
    """
    are_P_inverted = []
    for i in range(cleaned.shape[0]):
        are_P_inverted.append(check_inverted_P(cleaned[i], delineations[i]))
    return np.array(are_P_inverted)


########################################
# Visualization
########################################
def custom_ecg_delineate_plot(
        ecg_signal,
        rpeaks=None,
        delineation=None,
        sampling_rate=SAMPLING_RATE,
        window_range=(0.5, 1.5),
):

    window_start = int(window_range[0] * sampling_rate)
    window_end = int(window_range[1] * sampling_rate)

    if rpeaks is not None:
        rpeak_feat = np.zeros(ecg_signal.shape[0])
        rpeak_feat[rpeaks] = 1
        rpeak_feat = pd.DataFrame({"R_Peaks": rpeak_feat})
        delineation = pd.concat([delineation, rpeak_feat], axis=1)

    delineation.rename(
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
    time = pd.DataFrame({"Time": np.arange(0, ecg_signal.shape[0]) / sampling_rate})
    data = pd.concat([signal, delineation, time], axis=1)
    data = data.iloc[window_start:window_end]

    fig, ax = plt.subplots()
    ax.plot(data.Time, data.Signal, color="grey", alpha=0.2)

    colors = cm.rainbow(np.linspace(0, 1, len(delineation.columns.values)))
    for i, feature_type in enumerate(delineation.columns.values):
        event_data = data[data[feature_type] == 1.0]
        ax.scatter(event_data.Time, event_data.Signal, label=feature_type, alpha=0.5, s=50, color=colors[i])
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
    rr = np.diff(rpeaks)
    hr = 60 * SAMPLING_RATE / rr

    hrv_analysis = {}
    hrv_analysis['Mean RR (ms)'] = np.mean(rr)
    hrv_analysis['STD RR/SDNN (ms)'] = np.std(rr)
    hrv_analysis['Mean HR (Kubios\' style) (beats/min)'] = 60000 / np.mean(rr)
    hrv_analysis['Mean HR (beats/min)'] = np.mean(hr)
    hrv_analysis['STD HR (beats/min)'] = np.std(hr)
    hrv_analysis['Min HR (beats/min)'] = np.min(hr)
    hrv_analysis['Max HR (beats/min)'] = np.max(hr)
    hrv_analysis['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))  # Normal method to get HRV
    hrv_analysis['NNxx'] = np.sum(np.abs(np.diff(rr)) > 50) * 1
    hrv_analysis['pNNxx (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50) * 1) / len(rr)
    return hrv_analysis
