#!/usr/bin/python
""" Creates mock snapshot datasets
"""

import argparse
import pickle
import sys

import numpy as np


def make_random_snapshots(size: int, num_snaps: int, split: float = 0.9):
    snaps = []
    num_sites = size * size
    for _ in range(num_snaps):
        rand_perm = np.random.permutation(num_sites)
        up_idxs = rand_perm[:num_sites//2]
        down_idxs = rand_perm[num_sites//2:]
        new_snap = np.zeros((2, num_sites))
        new_snap[0][up_idxs] = 1
        new_snap[1][down_idxs] = 1
        snaps.append(new_snap.reshape((2, size, size)))
    snaps = np.stack(snaps, axis=0)
    return snaps


def make_random_density_snapshots(size: int, num_snaps: int, split: float = 0.9):
    snaps = []
    for _ in range(num_snaps):
        density = np.random.random()
        new_snap = np.random.choice([0.0, 1.0], (size, size), p=[1 - density, density])
        snaps.append(new_snap)
    snaps = np.stack(snaps, axis=0)
    return snaps


def make_stripe_snapshots(size: int, num_snaps: int, split: float = 0.9, p_flip: float = 0.2):
    snaps = []
    for _ in range(num_snaps):
        # Randomly pick the parity of the stripes
        if np.random.randint(2) == 0:
            p = [p_flip, 1.0 - p_flip]
        else:
            p = [1.0 - p_flip, p_flip]

        new_snap = np.zeros((2, size, size))
        new_snap[0, :, ::2] = np.random.choice(
            [0, 1], size=new_snap[0, :, ::2].shape, p=p
        )
        # Fill in holes in other channel
        new_snap[1, :, ::2] = 1 - new_snap[0, :, ::2]
        new_snap[1, :, 1::2] = np.random.choice(
            [0, 1], size=new_snap[1, :, 1::2].shape, p=p
        )
        new_snap[0, :, 1::2] = 1 - new_snap[1, :, 1::2]
        snaps.append(new_snap)
    snaps = np.stack(snaps, axis=0)
    return snaps


def make_split(snaps, split: float = 0.9):
    num_snaps = len(snaps)
    split_idx = int(np.ceil(split * num_snaps))
    snap_dict = {
        "snapshots": snaps,
        "train_idxs": np.arange(split_idx),
        "val_idxs": np.arange(split_idx, num_snaps),
    }
    return snap_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('snaptype', type=str,
                        help='Type of snapshots to make. Options: ["random", "randomdensity", "stripe"])')
    parser.add_argument('ofilename', type=str, help='Name of output dataset file')
    parser.add_argument('-s', '--snapsize', type=int, default=16, help='Size of snapshots')
    parser.add_argument('-n', '--nsnaps', type=int, default=10000, help='Number of snapshots')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data put into train set')
    args = parser.parse_args()

    if args.snaptype == 'random':
        snaps = make_random_snapshots(args.snapsize, args.nsnaps)
    elif args.snaptype == 'randomdensity':
        snaps = make_random_density_snapshots(args.snapsize, args.nsnaps)
    elif args.snaptype == 'stripe':
        snaps = make_stripe_snapshots(args.snapsize, args.nsnaps)
    else:
        print('Invalid snap type provided. Valid options: ["random"]')
        sys.exit(1)

    if args.ofilename.endswith('pkl'):
        snap_dict = make_split(snaps, split=args.split)
        with open(args.ofilename, 'wb') as ofile:
            pickle.dump(snap_dict, ofile)
    elif args.ofilename.endswith('npz'):
        snaps = np.transpose(snaps, (1, 2, 0))
        np.savez(args.ofilename, rydberg_populations=snaps)
