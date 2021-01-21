#!/usr/bin/python

""" Plots Rydberg snapshots from a given point in parameter space.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

import numpy as np

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

import plot_util

# A lot easier to just hard code these
radiuses = np.array([1.01, 1.05, 1.13, 1.23, 1.30, 1.39, 1.46, 1.56, 1.65, 1.71, 1.81, 1.89, 1.97])


def plot_snaps(data_dir, target_radius, target_detune, postselect=False):
    detune_npz = np.load(os.path.join(data_dir, 'detunings.npz'))
    detunings = detune_npz['detunings']

    irad = np.argmin(np.abs(radiuses - target_radius))
    rad = radiuses[irad]
    idet = np.argmin(np.abs(detunings - target_detune))
    det = detunings[idet]
    print("(Rad, Det) = ({:.2f}, {:.2f})".format(rad, det))

    snap_npz_filename = "detuning_sweep_Rb_{:.2f}.npz".format(rad)
    snap_npz = np.load(os.path.join(data_dir, snap_npz_filename))
    all_snaps = snap_npz['rydberg_populations_per_param']
    # Transpose to [NDETUNE, NSNAPS, WIDTH, HEIGHT]
    all_snaps = np.transpose(all_snaps, (2, 3, 0, 1)).astype(np.float32)
    # Add a "channel" dimension to get [NDETUNE, NSNAPS, 1, WIDTH, HEIGHT]
    all_snaps = np.expand_dims(all_snaps, 2)

    mask_npz_filename = snap_npz_filename.replace('detuning_sweep', 'rearrangement_mask')
    mask_npz = np.load(os.path.join(data_dir, mask_npz_filename))
    all_masks = mask_npz['rearrangement_mask']

    snaps = all_snaps[idet]
    if postselect:
        snaps = snaps[all_masks[idet]]

    plt.ion()
    fig, axs = plt.subplots(2, 4)
    axs = axs.flatten()

    idx = 0
    while idx < len(snaps):
        for i in range(8):
            axs[i].clear()
            axs[i].axis('off')
            plot_util.plot_rydberg_snap(snaps[idx+i, 0], ax=axs[i])
        input("Next eight snapshots? [Enter]")
        idx += 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', type=str)
    parser.add_argument('radius', type=float)
    parser.add_argument('detuning', type=float)
    parser.add_argument('--postselect', action='store_true')
    args = parser.parse_args()

    plot_snaps(args.sweep_dir, args.radius, args.detuning, postselect=args.postselect)