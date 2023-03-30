import os
from matplotlib import pyplot as plt
import pandas as pd


def scatter_two_agg(path_to_agg: str, x_name, y_name):
    """
        Compare the aggregated mid output via scatter plot
        """
    all_mid_output: pd.DataFrame = pd.read_csv(path_to_agg, header=0, index_col=0)
    fig_path = os.path.splitext(path_to_agg)[0] + '.png'
    # extract column x_name and y_name
    x = all_mid_output.loc[:, x_name]
    y = all_mid_output.loc[:, y_name]
    fig, ax = plt.subplots()
    corr = x.corr(y)
    ax.set_title(f'{y_name} vs {x_name} (r={corr:.4f})')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.scatter(x, y)
    plt.show(fig)
    fig.savefig(fig_path)
    ax.clear()
    plt.close(fig)


def generate_report(path_to_agg, patient_idx: int = 0):
    all_mid_output: pd.DataFrame = pd.read_csv(path_to_agg, header=0, index_col=0)
    mid_output: pd.Series = all_mid_output.loc[patient_idx]
    return mid_output
