
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from collections import defaultdict
from itertools import cycle

from typing import List, Optional

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'

def histogram(errors:np.ndarray) -> plt.figure:
    err = errors.flatten()
    sigma, mu = np.std(err), np.mean(err)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    fig = plt.figure(figsize=(4.8, 3.2))
    plt.axvspan(mu - 4 * sigma, mu + 4 * sigma, color='lightblue', alpha=0.5)
    plt.plot(x, pdf, color='blue', lw=2.0, alpha=0.8)
    plt.hist(err, bins=30, color='gray', density=True, alpha=1)
    plt.xlabel('Residues (V)')
    plt.ylabel('Probability density')
    plt.xlim([-1, 1])
    plt.ylim([0, 3])
    plt.grid(alpha=0.2)
    plt.tight_layout()
    return fig



def boxplot(errors:np.ndarray) -> plt.Figure:
    data_resn = defaultdict(list)
    step = cycle(np.arange(0, 24, 0.25))
    for res in errors:
        t = next(step)
        data_resn[t] += list(res)
    fig = plt.figure(figsize=(6.2, 2.3))
    plt.fill_between(x=np.arange(0, 24, 0.25), y1=4, y2=-4, color='lime', alpha=0.40)
    for key, val in data_resn.items():
        plt.boxplot(
            val, 
            positions=[key], 
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2),
            flierprops=dict(marker='.', markerfacecolor='k', markersize=4, linestyle='none')
        )
    plt.xticks(np.arange(0, 26, 2.0), np.arange(0, 26, 2.0))
    plt.yticks(np.arange(-6, 18, 6))
    plt.ylim([-6, 20])
    plt.grid(alpha=0.2)
    plt.ylabel('Normalized Residues')
    plt.xlabel('Time (h)')
    plt.tight_layout()
    return fig



def plot_outputs(columns:List[str], y_pred:np.ndarray, y_true:np.ndarray, y_optim:Optional[np.ndarray]=None) -> plt.figure:
    y_pred_series = pd.Series(data=y_pred, index=columns)
    y_true_series = pd.Series(data=y_true, index=columns)
    if y_optim is not None:
        y_optim_series = pd.Series(data=y_optim, index=columns)
    cols_a = [x for x in y_pred_series.index if '1' in x.split('.')]
    cols_b = [x for x in y_pred_series.index if '2' in x.split('.')]
    cols_c = [x for x in y_pred_series.index if '3' in x.split('.')]
    xa = np.arange(0, len(cols_a)) + 1
    xb = np.arange(len(cols_a), len(cols_a) + len(cols_b)) + 1
    xc = np.arange(len(cols_a) + len(cols_b), len(cols_a) + len(cols_b) + len(cols_c)) + 1
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))
    axes[0].plot(xa, y_true_series[cols_a].values, linestyle='-', color='blue', label='Measurements')
    axes[0].plot(xa, y_pred_series[cols_a].values, linestyle='--', color='red')
    if y_optim is not None:
        axes[0].plot(xa, y_optim_series[cols_a].values, linestyle='-.', lw=1.5, color='orange')
    axes[1].plot(xb, y_true_series[cols_b].values, linestyle='-', color='blue')
    axes[1].plot(xb, y_pred_series[cols_b].values, linestyle='--', color='red', label='Predictions')
    if y_optim is not None:
        axes[1].plot(xb, y_optim_series[cols_b].values, linestyle='-.', lw=1.5, color='orange')
    axes[2].plot(xc, y_true_series[cols_c].values, linestyle='-', color='blue')
    axes[2].plot(xc, y_pred_series[cols_c].values, linestyle='--', color='red')
    if y_optim is not None:
        axes[2].plot(xc, y_optim_series[cols_c].values, linestyle='-.', lw=1.5, color='orange', label='Adjusted')
    y_min = np.min([
        y_pred_series[cols_a].min(), y_true_series[cols_a].min(),
        y_pred_series[cols_b].min(), y_true_series[cols_b].min(),
        y_pred_series[cols_c].min(), y_true_series[cols_c].min()
    ]) - 2.0
    y_max = np.max([
        y_pred_series[cols_a].max(), y_true_series[cols_a].max(),
        y_pred_series[cols_b].max(), y_true_series[cols_b].max(),
        y_pred_series[cols_c].max(), y_true_series[cols_c].max()
    ]) + 0.5
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Voltage magnitudes (V)')
        ax.set_xlabel('Monitored nodes')
        ax.legend(loc='lower center').get_frame().set_edgecolor('none')
        ax.grid(alpha=0.25)
    axes[0].set_xticks(np.arange(0, max(xa) + 1, 5))
    axes[1].set_xticks(np.arange(min(xb), max(xb) + 1, 5))
    axes[2].set_xticks(np.arange(min(xc), max(xc) + 1, 5))
    plt.tight_layout()
    return fig



def plot_inputs(
    columns = List[str], 
    x_input = np.ndarray, 
    x_optim = np.ndarray,
    ):

    x_series = pd.Series(data=x_optim, index=columns)
    y_series = pd.Series(data=x_input, index=columns)

    cols_a = [x for x in x_series.index if '1' in x.split('.') and 'kW' in x.split('.')]
    cols_b = [x for x in x_series.index if '2' in x.split('.') and 'kW' in x.split('.')]
    cols_c = [x for x in x_series.index if '3' in x.split('.') and 'kW' in x.split('.')]

    xa = np.arange(0, len(cols_a)) + 1
    xb = np.arange(len(cols_a), len(cols_a) + len(cols_b)) + 1
    xc = np.arange(len(cols_a) + len(cols_b), len(cols_a) + len(cols_b) + len(cols_c)) + 1

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))

    axes[0].step(xa, y_series[cols_a].values, linestyle='-', color='blue', lw=1.5, where="mid", label='Measurements')
    axes[0].step(xa, x_series[cols_a].values, linestyle='-.', color='orange', lw=1.5, where="mid")
    axes[0].fill_between(x=xa - 0.5, y1=y_series[cols_a], step='post', color='blue', alpha=0.15)
    axes[0].fill_between(x=xa - 0.5, y1=x_series[cols_a], step='post', color='orange', alpha=0.15)
    axes[0].grid(alpha=0.2)

    axes[1].step(xb, y_series[cols_b].values, linestyle='-', color='blue', lw=1.5, where="mid")
    axes[1].step(xb, x_series[cols_b].values, linestyle='-.', color='orange', lw=1.5, where="mid", label='Adjusted')
    axes[1].fill_between(x=xb - 0.5, y1=y_series[cols_b], step='post', color='blue', alpha=0.15)
    axes[1].fill_between(x=xb - 0.5, y1=x_series[cols_b], step='post', color='orange', alpha=0.15)
    axes[1].grid(alpha=0.2)

    axes[2].step(xc, y_series[cols_c].values, linestyle='-', color='blue', lw=1.5, where="mid")
    axes[2].step(xc, x_series[cols_c].values, linestyle='-.', color='orange', lw=1.5, where="mid")
    axes[2].fill_between(x=xc - 0.5, y1=y_series[cols_c], step='post', color='blue', alpha=0.15)
    axes[2].fill_between(x=xc - 0.5, y1=x_series[cols_c], step='post', color='orange', alpha=0.15)
    axes[2].grid(alpha=0.2)

    axes[0].set_ylabel('Active powers (kW)')
    axes[1].set_ylabel('Active powers (kW)')
    axes[2].set_ylabel('Active powers (kW)')
    axes[0].set_xlabel('Monitored nodes')
    axes[1].set_xlabel('Monitored nodes')
    axes[2].set_xlabel('Monitored nodes')

    y_max = np.max(
        [x_series[cols_a].values.max(), y_series[cols_a].values.max(),
        x_series[cols_b].values.max(), y_series[cols_b].values.max(),
        x_series[cols_c].values.max(), y_series[cols_c].values.max(),
        ])+1

    for ax in axes:
        ax.tick_params(axis='both')
        ax.set_ylim(0, y_max)
        ax.legend(loc='upper left').get_frame().set_edgecolor('none')

    axes[0].set_xticks(np.arange(0, max(xa) + 1, 5))
    axes[1].set_xticks(np.arange(min(xb), max(xb) + 1, 5))
    axes[2].set_xticks(np.arange(min(xc), max(xc) + 1, 5))
    plt.tight_layout()
    return fig