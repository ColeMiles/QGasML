#!/usr/bin/python3
import pickle
import argparse
import sys
import os

import matplotlib.pyplot as plt

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

from plot_util import plot_l1_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots L1 paths produced by lassopath.py')
    parser.add_argument('pathfile', type=str, help='Name of the .pkl file containing L1 path info')
    parser.add_argument('--correlator', action='store_true',
                        help='If set, performs absolute values and normalization on filters.')
    parser.add_argument('--order', type=int, default=4,
                        help='Order of CCNN model that produced the .pkl')
    parser.add_argument('--dropin', action='store_true',
                        help='If set, drop in before plot is shown')
    args = parser.parse_args()

    params = {
        'font.family': 'CMU Sans Serif',
        'axes.titlesize': 20,
        'axes.labelsize': 28,
        'axes.linewidth': 2,
        'axes.labelpad': 10,  # 2D
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
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

    if not args.correlator:
        args.order = 1

    pathdata = pickle.load(open(args.pathfile, 'rb'))
    if args.dropin:
        import IPython
        IPython.embed()

    # NOTE: Passing negative coeffs because sklearn's convention is backwards from what I want
    #  I want positive beta -> evidence FOR the phase (class 0)
    fig, axs = plot_l1_path(
        pathdata['cs'], -pathdata['coeffs'], pathdata['train_accs'],
        pathdata['val_accs'],
        correlator=args.correlator, order=args.order
    )
    plt.show()

