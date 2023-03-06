from typing import Any
import numpy as np
from neurokit2.signal import signal_rate
from src.basic.constants import N_LEADS, SAMPLING_RATE, DEVIATION_TOL, LEAD_TO_INDEX
from src.basic.ecg import Ecg
from src.basic.cardiac_cycle import CardiacCycle
from src.utils.ecg_utils import analyze_hrv


##################################
# TODO: Delete this file
##################################
def remove_outliers(values: np.ndarray):
    """
    Remove outliers from a numpy array
    """
    # Get the 1st and 3rd quartiles
    q1, q3 = np.percentile(values, [25, 75])
    # Get the interquartile range
    iqr = q3 - q1
    # Get the lower and upper bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # Remove outliers
    return values[(values > lower_bound) & (values < upper_bound)]


def get_mean_without_outliers(values: np.ndarray):
    """
    Get the mean of a numpy array without outliers
    """
    # return np.mean(remove_outliers(values))
    return np.nanmean(values)


def get_std_without_outliers(values: np.ndarray):
    """
    Get the standard deviation of a numpy array without outliers
    """
    return np.std(remove_outliers(values))


class RuleModule():

    def __init__(self, ecg: Ecg, is_processed: bool = False):
        self.ecg: Ecg = ecg
        self.is_processed: bool = is_processed

    def process(self):
        """
        Process the ECG
        """
        self.is_processed = True

    def apply(self) -> Any:
        """
        Apply the rule to the ECG (and possibly predict CVD)
        """
        if not self.is_processed:
            self.process()


class RhythmRule(RuleModule):

    def process(self):
        RhythmRule.calc_heart_rate(self.ecg)
        # RhythmRule.calc_hrv(self.ecg)
        RhythmRule.calc_RR_intervals(self.ecg)
        RhythmRule.calc_PP_intervals(self.ecg)
        self.is_processed = True

    def apply(self) -> Any:
        if not self.is_processed:
            self.process()
        is_sinus = self.check_sinus_rhythm()
        # TODO: consider age and gender
        is_tachy = self.ecg.heart_rate_mean > 100
        is_brady = self.ecg.heart_rate_mean < 60
        print(f"HR: {self.ecg.heart_rate_mean}, is_tachy: {is_tachy}, is_brady: {is_brady}")
        if is_sinus:
            if is_tachy:
                return "STACH"
            elif is_brady:
                return "SBRAD"
            else:
                return "SR"

    def check_sinus_rhythm(self) -> bool:
        """
        Check if the ECG has sinus rhythm
        """
        # check lead II only
        idx = LEAD_TO_INDEX['II']
        are_RR_equidistant = np.allclose(np.diff(self.ecg.all_RR_intervals[idx] / self.ecg.RR_interval_mean),
                                         0,
                                         atol=DEVIATION_TOL)
        are_PP_equidistant = np.allclose(np.diff(self.ecg.all_PP_intervals[idx] / self.ecg.PP_interval_mean),
                                         0,
                                         atol=DEVIATION_TOL)
        are_PP_same_as_RR = np.allclose(self.ecg.all_PP_intervals[idx],
                                        self.ecg.all_RR_intervals[idx],
                                        rtol=DEVIATION_TOL)
        print(
            f'are_RR_equidistant: {are_RR_equidistant}, are_PP_equidistant: {are_PP_equidistant}, are_PP_same_as_RR: {are_PP_same_as_RR}'  # noqa E501
        )
        return are_RR_equidistant and are_PP_equidistant and are_PP_same_as_RR

    @staticmethod
    def calc_RR_intervals(ecg: Ecg):
        """
        Calculate RR intervals for each lead
        """
        if not ecg.has_cycles:
            ecg.get_cycles()

        all_cycles = ecg.all_cycles
        ecg.all_RR_intervals = []
        for i in range(N_LEADS):
            cycles: list[CardiacCycle] = all_cycles[i]
            rpeaks: list[int] = [cycle.R_peak for cycle in cycles]
            ecg.all_RR_intervals.append(np.diff(rpeaks) / SAMPLING_RATE * 1000)

        ecg.RR_interval_mean = np.nanmean(
            np.array([get_mean_without_outliers(RR_intervals) for RR_intervals in ecg.all_RR_intervals]))
        # ecg.RR_interval_std = np.std(
        #     np.array([get_std_without_outliers(RR_intervals) for RR_intervals in ecg.all_RR_intervals]))

        return ecg.all_RR_intervals

    @staticmethod
    def calc_PP_intervals(ecg: Ecg):
        """
        Calculate PP intervals for each lead
        """
        if not ecg.has_cycles:
            ecg.get_cycles()

        all_cycles = ecg.all_cycles
        ecg.all_PP_intervals = []
        for i in range(N_LEADS):
            cycles: list[CardiacCycle] = all_cycles[i]
            ppeaks: list[int] = [cycle.P_peak for cycle in cycles]
            ecg.all_PP_intervals.append(np.diff(ppeaks) / SAMPLING_RATE * 1000)  # Convert to ms

        ecg.PP_interval_mean = np.nanmean(
            np.array([get_mean_without_outliers(PP_intervals) for PP_intervals in ecg.all_PP_intervals]))
        # ecg.PP_interval_std = np.std(ecg.all_PP_intervals)
        return ecg.all_PP_intervals

    @staticmethod
    def calc_heart_rate(ecg: Ecg):
        """
        Calculate heart rate for each lead
        """
        if not ecg.has_found_rpeaks:
            ecg.find_rpeaks()
        ecg.all_heart_rates = []
        for i in range(N_LEADS):
            rpeaks = ecg.all_rpeaks[i]
            ecg.all_heart_rates.append(signal_rate(rpeaks, sampling_rate=SAMPLING_RATE,
                                                   desired_length=ecg.raw.shape[1]))

        ecg.all_heart_rates = np.stack(ecg.all_heart_rates)
        ecg.heart_rate_mean = get_mean_without_outliers(ecg.all_heart_rates)
        ecg.heart_rate_std = get_std_without_outliers(ecg.all_heart_rates)
        return ecg.all_heart_rates

    @staticmethod
    def calc_hrv(ecg: Ecg):
        """
        Calculate heart rate variability for each lead
        """
        if not ecg.has_found_rpeaks:
            ecg.find_rpeaks()
        ecg.all_hrv = []
        for i in range(N_LEADS):
            rpeaks = ecg.all_rpeaks[i]
            hrv_analysis = analyze_hrv(rpeaks)
            ecg.all_hrv.append(hrv_analysis['RMSSD (ms)'])

        return ecg.all_hrv


class IntervalRule(RuleModule):

    def apply(self, ecg: Ecg) -> Any:
        pass
