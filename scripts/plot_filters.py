#!/usr/bin/python
import argparse
import sys
import os

import numpy as np

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

from plot_util import plot_filters, plot_corr_filters

parser = argparse.ArgumentParser(description="Plots sets of filters in .npy files")
parser.add_argument('filter_file', type=str, help='A .npy file containing filters')
parser.add_argument('num_cols', type=int, nargs='?', default=3, help='Number of columns of filters.')
parser.add_argument('--hole-only', action='store_true')
parser.add_argument('--correlator', action='store_true')
parser.add_argument('--imshow', action='store_true')
parser.add_argument('--nonorm', action='store_true')
args = parser.parse_args()

filts = np.load(args.filter_file)

if args.correlator:
    plot_corr_filters(filts, num_cols=args.num_cols, imshow=args.imshow, norm=not args.nonorm)
else:
    plot_filters(filts, imshow=args.imshow, hole=args.hole_only, num_cols=args.num_cols,)
