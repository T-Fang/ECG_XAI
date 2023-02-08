import pandas as pd
import torch


class Ecg:

    def __init__(self, recording: torch.Tensor, metadata: pd.Series):
        """
        recording: 12x5000 matrix (500Hz sampling rate).
                    Order of leads in dataset: I,II,III,aVR,aVL,aVF,V1–V6
                    Voltage Unit: mV;
                    Time Unit: second
        metadata: metadata describing the recording and the corresponding patient
        """
        self.recording = recording
        self.metadata = metadata

    @property
    def labels(self):
        return self.metadata.subclass

    @property
    def age(self):
        return self.metadata.age

    @property
    def gender(self):
        """
        gender of the patient (male 0, female 1)
        """
        return self.metadata.sex

    @property
    def heart_axis(self):
        return self.metadata.heart_axis

    @property
    def MI_stage(self):
        return self.metadata.infarction_stadium1

    @property
    def superclass(self):
        return self.metadata.superclass

    @property
    def subclass(self):
        return self.metadata.subclass
