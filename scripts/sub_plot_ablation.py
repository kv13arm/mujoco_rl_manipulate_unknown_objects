import argparse
import os
import os.path as osp
from pathlib import Path
from glob import glob
from typing import Callable, Tuple, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn
from matplotlib.ticker import FuncFormatter
from stable_baselines3.common.monitor import LoadMonitorResultsError
from stable_baselines3.common.results_plotter import load_results, ts2xy, X_TIMESTEPS, X_WALLTIME


def millions(x, pos):
    """
    Formatter for matplotlib
    The two args are the value and tick position
    :param x: (float)
    :param pos: (int) tick position (not used here
    :return: (str)
    """
    return '{:.1f}M'.format(x * 1e-6)

def thousands(x, pos):
    """
    Formatter for matplotlib
    The two args are the value and tick position
    :param x: (float)
    :param pos: (int) tick position (not used here
    :return: (str)
    """
    return '{:.0f}'.format(x * 1e-3)

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def smooth(xy, window=50):
    x, y = xy
    if y.shape[0] < window:
        return x, y

    original_y = y.copy()
    y = moving_average(y, window)

    if len(y) == 0:
        return x, original_y

    # Truncate x
    x = x[len(x) - len(y):]
    return x, y


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: the input Array
    :param window: length of the rolling window
    :return: rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: variable 1
    :param var_2: variable 2
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2 (such as np.mean)
    :return:  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1 :], function_on_var2


def get_all_subdirs(all_logdirs):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           list all subdirectories of the entry and add them to the list
    """
    logdirs = []

    for logdir in all_logdirs:
        if osp.isdir(logdir):
            logdirs += [logdir]
            logdirs += [osp.join(logdir, subdir) for subdir in os.listdir(logdir) if osp.isdir(osp.join(logdir, subdir))]
        else :
            print("{} is not a directory".format(logdir))
            break

    all_sub_dirs = []
    for folder_idx, folder_log_path in enumerate(logdirs):
        subdirs = [osp.join(folder_log_path, subdir) for subdir in os.listdir(folder_log_path) if (osp.isdir(osp.join(folder_log_path, subdir)) and any(fname.endswith('.csv') for fname in os.listdir(osp.join(folder_log_path, subdir))))]
        all_sub_dirs += subdirs

    return all_sub_dirs


def get_monitor_files_csv(path: str) -> List[str]:
    """
    get all the monitor files in the given path

    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, "*.csv" ))


def load_results_csv(path: str) -> pandas.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files_csv(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *.csv found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            # first_line = file_handler.readline()
            # assert first_line[0] == "#"
            # header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            # headers.append(header)
            # data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("wall_time", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame = data_frame[data_frame['metric'] == "eval/mean_reward"]
    # data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame


def ts2xy_csv(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = data_frame.epoch.values
        y_var = data_frame.value.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.wall_time.values / 3600.0
        y_var = data_frame.value.values
    else:
        raise NotImplementedError
    return x_var, y_var


# Init seaborn
seaborn.set()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', type=str)
parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Smooth Learning Curve')
args = parser.parse_args()

################################## TRAINING DIR #####################################
args.log_dirs = [Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/ablation/sugar_cube_baseline_best_model",
                 Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/ablation/sugar_cube_no_contact_best_model",
                 Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/ablation/sugar_cube_no_pheromone_best_model",
                 Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/ablation/sugar_cube_no_roll_best_model",
                 Path(__file__).resolve().parent.parent.as_posix() + "/models/trained_models/ablation/sugar_cube_no_depth_best_model"]

plot_title = ["baseline", "baseline", "contact information", "contact information", "pheromone information", "pheromone information", "roll movement", "roll movement", "visual observation", "visual observation"]


args.smooth = True
args.title = ""
plot_train = True

results = []
algos = []

for folder in args.log_dirs:
    if plot_train:
        timesteps = load_results(folder)
    else:
        timesteps = load_results_csv(folder)
    results.append(timesteps)
    if folder.endswith('\\'):
        folder = folder[:-1]
    algos.append(folder.split('\\')[-1])

min_timesteps = np.inf

plot_color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# 'walltime_hrs', 'episodes'
for plot_type in ['timesteps']:
    xy_list = []
    color_list = []
    algo_legend = []
    y_min_max_to_consider = []
    choice_legend = []
    for idx, result in enumerate(results):
        if plot_train:
            x, y = ts2xy(result, plot_type)
        else:
            x, y = ts2xy_csv(result, plot_type)

        color_list.append(matplotlib.colors.to_rgba(plot_color_list[idx], alpha=0.3))
        xy_list.append((x, y))
        choice_legend = ["baseline", "no contact", "no pheromone", "no roll", "no depth"]
        if 'baseline' in algos[idx]:
            algo_legend.append(choice_legend[0])
        elif 'contact' in algos[idx]:
            algo_legend.append(choice_legend[1])
        elif 'pheromone' in algos[idx]:
            algo_legend.append(choice_legend[2])
        elif 'roll' in algos[idx]:
            algo_legend.append(choice_legend[3])
        elif 'depth' in algos[idx]:
            algo_legend.append(choice_legend[4])

        # algo_legend.append(algos[idx])
        y_min_max_to_consider.append(False) #do not consider Ymin or max in plot for raw data
        if args.smooth:
            if plot_train:
                window_size = 600
            else:
                window_size = 100
            smooth_type = 2
            if smooth_type == 1:
                x, y = window_func(x, y, window_size, np.mean)
                x1, y1 = smooth((x, y), window=window_size)
                xy_list.append((x1, y1))
                # algo_legend.append("Smoothed type 1 ")
                color_list.append(matplotlib.colors.to_rgba(plot_color_list[idx]))
                y_min_max_to_consider.append(True)
                choice_legend = ["baseline", "no contact", "no pheromone", "no roll", "no depth"]
                if 'baseline' in algos[idx]:
                    algo_legend.append(choice_legend[0])
                elif 'contact' in algos[idx]:
                    algo_legend.append(choice_legend[1])
                elif 'pheromone' in algos[idx]:
                    algo_legend.append(choice_legend[2])
                elif 'roll' in algos[idx]:
                    algo_legend.append(choice_legend[3])
                elif 'depth' in algos[idx]:
                    algo_legend.append(choice_legend[4])
            elif smooth_type == 2:
                """
                smooth data with moving window average.
                that is,
                    smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
                where the "smooth" param is width of that window (2k+1)
                """
                y_smooth = np.ones(window_size)

                x_smooth= np.asarray(y.copy())
                z_smooth = np.ones(len(x_smooth))
                smoothed_rew = np.convolve(x_smooth, y_smooth, 'same') / np.convolve(z_smooth, y_smooth, 'same')

                xy_list.append((x, smoothed_rew))
                color_list.append(matplotlib.colors.to_rgba(plot_color_list[idx]))
                # algo_legend.append("smoothed " + algos[idx])
                y_min_max_to_consider.append(True)
                choice_legend = ["baseline", "no contact", "no pheromone", "no roll", "no depth"]
                if 'baseline' in algos[idx]:
                    algo_legend.append(choice_legend[0])
                elif 'contact' in algos[idx]:
                    algo_legend.append(choice_legend[1])
                elif 'pheromone' in algos[idx]:
                    algo_legend.append(choice_legend[2])
                elif 'roll' in algos[idx]:
                    algo_legend.append(choice_legend[3])
                elif 'depth' in algos[idx]:
                    algo_legend.append(choice_legend[4])

        n_timesteps = x[-1]
        if n_timesteps < min_timesteps:
            min_timesteps = n_timesteps


    fig = plt.figure(args.title)
    y_min_max_init = [0, 100]
    y_min_max = y_min_max_init # min and max of y axis for each subplot that have True in y_min_max_to_consider
    for i, (x, y) in enumerate(xy_list):
        # skip baseline as it is present in every other plot
        if i == 0 or i == 1:
            continue
        plt.subplot(2, 2, int((i-2)/2)+1)
        if y_min_max_to_consider[i]:
            plt.plot(x[:min_timesteps], y[:min_timesteps], label=algo_legend[i], linewidth=2, color=color_list[i])
        else:
            plt.plot(x[:min_timesteps], y[:min_timesteps], linewidth=2, color=color_list[i])

        if(i%2 == 0): # reset y_min_max for each subplot
            plt.plot(xy_list[0][0][:min_timesteps], xy_list[0][1][:min_timesteps], linewidth=2,
                     color=color_list[0])
            plt.plot(xy_list[1][0][:min_timesteps], xy_list[1][1][:min_timesteps], linewidth=2, label=algo_legend[0], color = color_list[1])

            y_min_max = y_min_max_init
        if y_min_max_to_consider[i]:
            y_min_max = [min(y_min_max[0], min(y)), max(y_min_max[1], max(y))]
            plt.gca().set_ylim(y_min_max[0], y_min_max[1])
        else:
            if y_min_max != []:
                plt.gca().set_ylim(y_min_max)

        plt.ylabel("mean reward", fontsize=20)
        plt.xlabel("thousand steps", fontsize=20)
        plt.legend(loc='upper right', fontsize=20)
        plt.title(plot_title[i], fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    formatter = FuncFormatter(thousands)
    for axe in fig.axes:
        axe.xaxis.set_major_formatter(formatter)

    fig.set_figheight(15)
    fig.set_figwidth(20)
    fig.subplots_adjust(hspace=0.3)

plt.savefig('ablation.png')
plt.show()
