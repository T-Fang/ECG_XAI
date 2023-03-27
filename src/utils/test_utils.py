import pandas as pd


def generate_report(path_to_mid_output, patient_idx: int = 0):
    all_mid_output: pd.DataFrame = pd.read_csv('file.csv', header=0, index_col=0)
    mid_output: pd.Series = all_mid_output.loc[patient_idx]
    return mid_output
