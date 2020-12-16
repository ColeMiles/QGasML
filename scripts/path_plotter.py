#!/usr/bin/python3
import pickle
import argparse
import sys
import os

import matplotlib.pyplot as plt

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

from plot_util import plot_l1_path, plotly_l1_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots L1 paths from reduced_lassosearch.py')
    parser.add_argument('pathfile', type=str, help='Name of the .pkl file containing L1 path info')
    parser.add_argument('--correlator', action='store_true', help='If set, plots for CorrelatorExtractor features')
    parser.add_argument('--order', type=int, default=4, help='Order of model that produced the .pkl')
    parser.add_argument('--plotly', action='store_true', help='If set, plots the plotly version with numbered lines')
    parser.add_argument('--dropin', action='store_true', help='If set, drop in before plot is shown')
    args = parser.parse_args()

    if not args.correlator:
        args.order = 1

    pathdata = pickle.load(open(args.pathfile, 'rb'))

    if not args.plotly:
        plt.style.use('seaborn-talk')

        fig, axs = plot_l1_path(
            pathdata['cs'], pathdata['coeffs'], pathdata['train_accs'],
            pathdata['val_accs'], pathdata['exp_fracs'],
            correlator=args.correlator, order=args.order
        )

        if args.dropin:
            import IPython
            IPython.embed()

        plt.show()
    else:
        plotly_l1_path(pathdata['cs'], pathdata['coeffs'])

