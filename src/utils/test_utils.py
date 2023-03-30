import os
from matplotlib import pyplot as plt
import pandas as pd
import torch


def compare_agg_via_scatter(path_to_mid_output: str, x_name, y_name):
    """
        Compare the aggregated mid output via scatter plot
        """
    all_mid_output: pd.DataFrame = pd.read_csv('file.csv', header=0, index_col=0)
    fig_path = os.path.join(os.path.splitext(path_to_mid_output)[0], 'png')
    # fig, ax = plt.subplots()
    # for xlabel, ylabel in self.compared_agg:
    #     x = self.mid_output_agg[xlabel]
    #     y = self.mid_output_agg[ylabel]
    #     corr = torch.corrcoef(torch.stack([x, y], dim=0))[0, 1]
    #     ax.set_title(f'{ylabel} vs {xlabel} (r={corr:.4f})')
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.scatter(x, y)
    #     fig.savefig(fig_path)
    #     ax.clear()
    # plt.close(fig)


def generate_report(path_to_mid_output, patient_idx: int = 0):
    all_mid_output: pd.DataFrame = pd.read_csv('file.csv', header=0, index_col=0)
    mid_output: pd.Series = all_mid_output.loc[patient_idx]
    return mid_output
