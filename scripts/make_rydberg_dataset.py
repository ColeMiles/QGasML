#!/usr/bin/python
import argparse
import os

import numpy as np

""" Collects Rydberg snapshots from a set of detunings and radiuses, optionally perform
     some form of density normalization, and save out all snapshots as a .npz file.
"""

"""
For my note, all detunings:

array([-2.3255814 , -2.09302326, -1.86046512, -1.62790698, -1.39534884,
       -1.1627907 , -0.93023256, -0.69767442, -0.46511628, -0.23255814,
        0.        ,  0.23255814,  0.46511628,  0.69767442,  0.93023256,
        1.1627907 ,  1.39534884,  1.62790698,  1.86046512,  2.09302326,
        2.3255814 ,  2.55813953,  2.79069767,  3.02325581,  3.25581395,
        3.48837209,  3.72093023,  3.95348837,  4.18604651,  4.41860465,
        4.65116279])

And, all radiuses:

array([1.01, 1.05, 1.13, 1.23, 1.30, 1.39, 1.46, 1.56, 1.65, 1.71, 1.81, 1.89, 1.97])
"""

def make_dataset(sweep_dir, ofilename, dets, radiuses, density_norm=False, uniform_norm=False):
    actual_dets = np.load(os.path.join(sweep_dir, 'detunings.npz'))
    actual_dets = actual_dets['detunings']
    # Indexes mapping approximate detunings to the closest detunings in the dataset
    det_idxs = [
        np.argmin(np.abs(actual_dets - d)) for d in dets
    ]

    all_snaps = []
    all_rearranges = []

    for r in radiuses:
        npz_filename = os.path.join(sweep_dir, 'detuning_sweep_Rb_{}.npz'.format(r))
        rearr_filename = os.path.join(sweep_dir, 'rearrangement_mask_Rb_{}.npz'.format(r))

        snap_npz = np.load(npz_filename)
        rearr_npz = np.load(rearr_filename)

        for i in det_idxs:
            snaps = snap_npz['rydberg_populations_per_param'][:, :, i, :]
            snaps = snaps.astype(np.float64)
            if density_norm:
                snaps -= np.mean(snaps, axis=-1, keepdims=True)
            elif uniform_norm:
                snaps -= np.mean(snaps)
            all_snaps.append(snaps)
            all_rearranges.append(rearr_npz['rearrangement_mask'][i, :])

    all_snaps = np.concatenate(all_snaps, axis=-1)
    all_rearranges = np.concatenate(all_rearranges, axis=-1)

    print("\n{} snaps collected\n".format(all_snaps.shape[-1]))

    np.savez(ofilename, rydberg_populations=all_snaps)
    rearr_filename = ofilename.replace('rydberg', 'rearrangement_mask')
    np.savez(rearr_filename, rearragement_mask=all_rearranges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        descrption="Collects Rydberg snapshots from the directory containing the linear sweeps,"
                   " placing them into a .npy file, optionally with some density normalization"
                   " performed."
    )
    parser.add_argument('sweep_dir', type=str, help='Source of data')
    parser.add_argument('out_filename', type=str, help='Name of output filename')
    parser.add_argument('-d', '--detunings', type=float, nargs='+',
                        help='Detunings to collect data from')
    parser.add_argument('-r', '--radiuses', type=str, nargs='+',
                        help='Rydberg radii to collect data from')
    parser.add_argument('--density-norm', action='store_true', default=False,
                        help='Perform per-site density normalization on snapshots before saving.')
    parser.add_argument('--uniform-norm', action='store_true', default=False,
                        help='Perform a uniform density normalization on snapshots before saving.')
    args = parser.parse_args()

    make_dataset(args.sweep_dir, args.out_filename, args.detunings, args.radiuses,
                 density_norm=args.density_norm, uniform_norm=args.uniform_norm)
