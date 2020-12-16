#!/usr/bin/python
""" Plots a couple snapshots from two datasets
"""

import argparse
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_snaps(snaps1, snaps2, name1="A", name2="B", nsnaps=3):
    fig = plt.figure(constrained_layout=False)
    outergrid = fig.add_gridspec(1, 2, left=0.05, right=0.95, wspace=0.2)

    center_idx = nsnaps // 2

    # Plot snaps1
    innergrid1 = outergrid[0, 0].subgridspec(nsnaps, nsnaps, wspace=0.05, hspace=0.05)
    for i in range(nsnaps):
        for j in range(nsnaps):
            snap = snaps1[nsnaps * i + j]
            # Add a dummy green channel
            snap = np.stack((snap[0], np.zeros_like(snap[0]), snap[1]))
            snap = snap.transpose((1, 2, 0))
            ax = fig.add_subplot(innergrid1[i, j])
            ax.axis('off')
            ax.imshow(snap)

            if i == 0 and j == center_idx:
                ax.set_title(name1)

    # Plot snaps2
    innergrid2 = outergrid[0, 1].subgridspec(nsnaps, nsnaps, wspace=0.05, hspace=0.05)
    for i in range(nsnaps):
        for j in range(nsnaps):
            snap = snaps2[nsnaps * i + j]
            # Add a dummy green channel
            snap = np.stack((snap[0], np.zeros_like(snap[0]), snap[1]))
            # Transpose axes to what matplotlib expects
            snap = snap.transpose((1, 2, 0))
            ax = fig.add_subplot(innergrid2[i, j])
            ax.axis('off')
            ax.imshow(snap)

            if i == 0 and j == center_idx:
                ax.set_title(name2)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset1', type=str)
    parser.add_argument('dataset2', type=str)
    parser.add_argument('--nsnaps', type=int, default=3, help='sqrt of number of snaps to plot')
    args = parser.parse_args()

    dset1 = pickle.load(open(args.dataset1, 'rb'))
    dset2 = pickle.load(open(args.dataset2, 'rb'))

    dname1 = os.path.basename(os.path.dirname(args.dataset1))
    dname2 = os.path.basename(os.path.dirname(args.dataset2))

    snaps1 = dset1["snapshots"][:args.nsnaps**2]
    snaps2 = dset2["snapshots"][:args.nsnaps**2]

    plot_snaps(snaps1, snaps2, dname1, dname2, nsnaps=args.nsnaps)