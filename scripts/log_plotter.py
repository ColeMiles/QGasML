#!/usr/bin/python3
import argparse
import os
import sys
import re

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def make_plot(logfiles, metric_keys, logfile_labels=None, title=None, smooth=1, running_max=False):
    """ Scans logfiles to generate plots from data contained
         logfiles:       List of file handles to pull data from
         metric_keys:    List of metric names to scan for in logfiles
         logfile_labels: List of labels to associate to each logfile
         title:          String title of the plot
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    if len(metric_keys) == 1:
        ax.set_ylabel(metric_keys[0])
    ax.set_title(title)

    if logfile_labels is None:
        logfile_labels = [os.path.split(f.name)[1].rstrip('.tx') for f in logfiles]

    log_metrics = [parse_logfile(f) for f in logfiles]
    if smooth != 1:
        smooth_filt = np.array([1.0 / smooth] * smooth)
        for metric_set in log_metrics:
            for metric_name in metric_set.keys():
                if metric_name != 'Epoch':
                    metric_set[metric_name] = signal.correlate(
                        metric_set[metric_name], smooth_filt, mode='valid'
                    )

    unique_labels = []
    for label in logfile_labels:
        if label not in unique_labels:
            unique_labels.append(label)

    # Assumes the number of epochs is the same in all log files with the same label!
    for label in unique_labels:
        for metric in metric_keys:
            metric_collection = []
            for metric_set, set_label in zip(log_metrics, logfile_labels):
                if set_label == label:
                    metric_collection.append(metric_set[metric])
                    epochs = metric_set['Epoch']
            # [N_LOGS, N_EPOCHS] array of data for this specific metric for this specific label
            metric_collection = np.array(metric_collection)
            # Might not be the true number of epochs due to smoothing
            n_epochs = metric_collection.shape[1]

            if running_max:
                # Do a 'cumulative maximum'
                #  x[i] = min_{j < i} x[j]
                for i in range(1, n_epochs):
                    metric_collection[:, i] = np.max(metric_collection[:, :i+1], axis=1)

            spread = ax.fill_between(
                epochs[:n_epochs], np.min(metric_collection, axis=0), np.max(metric_collection, axis=0),
                alpha=0.4
            )
            spread_color = spread.get_facecolor()[0].copy()
            # Turn alpha up to 1.0
            spread_color[3] = 1.0
            plot_label = "{}, {}".format(metric, label) if len(metric_keys) != 1 else label
            ax.plot(
                epochs[:n_epochs], np.median(metric_collection, axis=0), lw=3,
                color=spread_color, label=plot_label
            )

    fig.legend()
    plt.tight_layout()
    plt.show()


def parse_logfile(logfile):
    """ Given a logfile, returns a dictionary containing all of the metrics from the file
    """
    metrics = {k: [] for k in ['Epoch', 'Val Accuracy', 'Val Loss', 'Train Accuracy', 'Train Loss']}
    line = logfile.readline()
    while line != '':
        for metric_name in metrics.keys():
            match = re.search(metric_name + ': (\d+\.?\d*)$', line)
            if match is not None:
                metrics[metric_name].append(float(match.group(1)))
        line = logfile.readline()
    return metrics


def collect_logfiles(logprefixes, labels):
    """ Given a list of log filename prefixes, return a list of logfiles starting with
         that prefix, and a list of matching labels
    """
    logfiles = []
    loglabels = []
    for full_prefix, label in zip(logprefixes, labels):
        base_dir, prefix = os.path.split(full_prefix)
        all_files = os.listdir(base_dir)
        matching_files = [
            os.path.join(base_dir, filename) for filename in all_files if filename.startswith(prefix)
        ]
        logfiles.extend(matching_files)
        loglabels.extend(len(matching_files) * [label,])
    return logfiles, loglabels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Makes plots from log files')
    parser.add_argument('logprefixes', nargs='+', type=str,
                        help='Prefixes of log file names to generate plots from')
    parser.add_argument('--labels', nargs='+', type=str,
                        help='Labels to give each set of log file in plot')
    parser.add_argument('--title', type=str,
                        help='Title to give plot')
    parser.add_argument('-va', '--val-acc', action='store_true',
                        help='If set, includes val accuracy in the plot')
    parser.add_argument('-vl', '--val-loss', action='store_true',
                        help='If set, includes val loss in the plot')
    parser.add_argument('-ta', '--train-acc', action='store_true',
                        help='If set, includes train accuracy in the plot')
    parser.add_argument('-tl', '--train-loss', action='store_true',
                        help='If set, includes train loss in the plot')
    parser.add_argument('-s', '--smooth', type=int, default=1,
                        help='If set, smooths over a window of the given size')
    parser.add_argument('--max', action='store_true',
                        help='If set, plots represent the `best seen so far` at each epoch')
    args = parser.parse_args()

    params = {
        'font.family': 'CMU Sans Serif',
        'axes.titlesize': 20,
        'axes.labelsize': 36,
        'axes.linewidth': 2,
        #     'axes.labelpad': 40, # 3D only
        'axes.labelpad': 0,  # 2D
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'xtick.minor.width': 1,
        'xtick.minor.size': 5,
        'xtick.major.width': 2,
        'xtick.major.size': 10,
        'ytick.major.width': 2,
        'ytick.major.size': 10,
        'xtick.direction': 'in',
        'ytick.minor.width': 1,
        'ytick.minor.size': 5,
        'ytick.direction': 'in',
        'font.weight': 'normal',
        'axes.unicode_minus': False,
        'legend.fontsize': 36,
        'mathtext.fontset': 'cm'
    }

    plt.rcParams.update(params)

    if all([not args.val_acc, not args.val_loss, not args.train_acc, not args.train_loss]):
        print('No data set to be plotted! Exiting...')
        sys.exit(1)

    if args.labels is not None and len(args.labels) != len(args.logprefixes):
        print('Number of logfile labels not equal to the number of log prefixes!')
        sys.exit(1)

    metrics = []
    if args.val_acc:
        metrics.append('Val Accuracy')
    if args.val_loss:
        metrics.append('Val Loss')
    if args.train_acc:
        metrics.append('Train Accuracy')
    if args.train_loss:
        metrics.append('Train Loss')

    logfiles, file_labels = collect_logfiles(args.logprefixes, args.labels)
    logfiles = [open(file, 'r') for file in logfiles]

    # plt.style.use("seaborn-paper")
    make_plot(logfiles, metrics, file_labels, args.title, args.smooth, args.max)
