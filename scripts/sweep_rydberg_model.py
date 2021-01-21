#!/usr/bin/python

""" Given a trained OVRModel for the Rydberg dataset, predict on each point in parameter
     space.
"""

import argparse
import os
import sys
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

import nn_models
import config_util
from .rydberg_unsup import make_mesh

# A lot easier to just hard code these
radiuses = [1.01, 1.05, 1.13, 1.23, 1.30, 1.39, 1.46, 1.56, 1.65, 1.71, 1.81, 1.89, 1.97]

def sweep_ovr(model, data_dir, postselect=False, savedir=None):
    num_classes = model.num_classes
    detune_npz = np.load(os.path.join(data_dir, 'detunings.npz'))
    detunings = detune_npz['detunings']
    order_params = np.empty((num_classes, len(radiuses), len(detunings)))
    cmaps = [plt.get_cmap(mapname) for mapname in
             ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']]

    with torch.no_grad():
        for i, radius in enumerate(radiuses):
            print("Rb =", radius)
            snap_npz_filename = "detuning_sweep_Rb_{:.2f}.npz".format(radius)
            snap_npz = np.load(os.path.join(data_dir, snap_npz_filename))
            all_snaps = snap_npz['rydberg_populations_per_param']

            # Transpose to [NDETUNE, NSNAPS, WIDTH, HEIGHT]
            all_snaps_re = np.transpose(all_snaps, (2, 3, 0, 1)).astype(np.float32)
            all_snaps = np.expand_dims(all_snaps_re, 2)

            mask_npz_filename = snap_npz_filename.replace('detuning_sweep',
                                                      'rearrangement_mask')
            mask_npz = np.load(os.path.join(data_dir, mask_npz_filename))
            all_masks = mask_npz['rearrangement_mask']

            for j, detuning in enumerate(detunings):
                snaps = all_snaps[j]

                if postselect:
                    snaps = snaps[all_masks[j]]

                # Normalize density of snapshots to be 0-mean per site
                snaps -= np.mean(snaps, axis=0, keepdims=True)

                snaps = torch.tensor(snaps)

                preds = model(snaps).detach().cpu().numpy()

                order_params[:, i, j] = np.mean(preds, axis=0)

    # Apply the logistic func
    order_params = 1.0 / (1.0 + np.exp(-order_params))

    # Plot that image!
    det_edges, r_edges = make_mesh(detunings, radiuses)

    for n in range(num_classes):
        fig, ax = plt.subplots(figsize=(6.4, 4.8 * 1.05))
        m = ax.pcolormesh(det_edges, r_edges, order_params[n], cmap=cmaps[n],
                          rasterized=True)

        ax.set_xlabel(r'$\Delta / \Omega$')
        ax.set_ylabel(r'$R_b / a$')
        ax.set_xticks([-2, 0, 2, 4])
        ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])

        ax.contour(detunings, radiuses, order_params[n], [0.75], colors=[cmaps[n](1.0)],
                   linewidths=4)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(m, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=18, direction='inout')
        cbar.ax.xaxis.set_ticks_position('top')
        plt.tight_layout()
        if savedir is not None:
            savepath = os.path.join(savedir, "map{}.svg".format(n))
            plt.savefig(savepath, dpi=200)
        else:
            plt.show()

    fig, ax = plt.subplots(figsize=(6.4, 4.8 * 1.0))
    ax.set_xlabel(r'$\Delta / \Omega$')
    ax.set_ylabel(r'$R_b / a$')
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])
    for n in range(num_classes):
        ax.contour(detunings, radiuses, order_params[n], [0.6, 0.65, 0.7, 0.75],
                   colors=[cmaps[n](0.2), cmaps[n](0.4), cmaps[n](0.6), cmaps[n](0.8)],
                   linewidths=3)
    plt.tight_layout()
    plt.show()

    return order_params


def sweep_explicit(data_dir, postselect=False):
    detune_npz = np.load(os.path.join(data_dir, 'detunings.npz'))
    detunings = detune_npz['detunings']

    corr_grid_check = np.zeros((len(radiuses), len(detunings)))
    corr_grid_stri = np.zeros((len(radiuses), len(detunings)))
    corr_grid_star = np.zeros((len(radiuses), len(detunings)))
    for i, radius in enumerate(radiuses):
        print("Rb = " + str(radius))
        snap_npz_filename = "detuning_sweep_Rb_{:.2f}.npz".format(radius)
        snap_npz = np.load(os.path.join(data_dir, snap_npz_filename))
        all_snaps = snap_npz['rydberg_populations_per_param']
        # Transpose to [NDETUNE, NSNAPS, WIDTH, HEIGHT]
        all_snaps = np.transpose(all_snaps, (2, 3, 0, 1)).astype(np.float32)

        mask_npz_filename = snap_npz_filename.replace('detuning_sweep',
                                                      'rearrangement_mask')
        mask_npz = np.load(os.path.join(data_dir, mask_npz_filename))
        all_masks = mask_npz['rearrangement_mask']

        for j, detuning in enumerate(detunings):
            snaps = all_snaps[j]

            if postselect:
                snaps = snaps[all_masks[j]]

            snaps = np.square(np.abs(np.fft.fft2(snaps, (16, 16))))
            corr_grid_stri[i, j] = np.mean(snaps[..., 8, 0] + snaps[..., 0, 8]) - np.mean(snaps[..., 8, 4] + snaps[..., 4, 8])
            corr_grid_check[i, j] = 2 * np.mean(snaps[..., 8, 8]) - np.mean(snaps[..., 8, 0] + snaps[..., 0, 8])
            corr_grid_star[i, j] = np.mean(snaps[..., 8, 4] + snaps[..., 4, 8]) - 2 * np.mean(snaps)

    det_edges, r_edges = make_mesh(detunings, radiuses)

    # Plot those images!
    fig, ax = plt.subplots(figsize=(6, 8))
    m = ax.pcolormesh(det_edges, r_edges, corr_grid_check, cmap='Reds')
    #     m.set_array(None)
    ax.set_xlabel(r'$\Delta / \Omega$')
    ax.set_ylabel(r'$R_b / a$')
    xticks = [np.round(detunings[x], 2) for x in range(0, 31, 5)]
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 8))
    m = ax.pcolormesh(det_edges, r_edges, corr_grid_stri, cmap='Blues')
    ax.set_xlabel(r'$\Delta / \Omega$')
    ax.set_ylabel(r'$R_b / a$')
    xticks = [np.round(detunings[x], 2) for x in range(0, 31, 5)]
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 8))
    m = ax.pcolormesh(det_edges, r_edges, corr_grid_star, cmap='Greens')
    ax.set_xlabel(r'$\Delta / \Omega$')
    ax.set_ylabel(r'$R_b / a$')
    xticks = [np.round(detunings[x], 2) for x in range(0, 31, 5)]
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])
    plt.show()

# Load linear_sweeps, predict, make map of classification across diagram
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Torch .pt model file to use in prediction sweeps.')
    parser.add_argument('config', type=str, help='Config the model was built/trained from.')
    parser.add_argument('data_dir', type=str, help='Base directory containing linear sweeps.')
    parser.add_argument('--explicit', action='store_true', help='Plot explicit Fourier order parameters.')
    parser.add_argument('--postselect', action='store_true',
                        help='If set, only uses snapshots with perfect rearrangement.')
    parser.add_argument('--dropin', action='store_true')
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--savedir', type=str, default=None,
                        help='Directory to save plots to.')
    parser.add_argument('--spat', action='store_true',
                        help='If set, loads model as CCNNSpatWgt.')
    args = parser.parse_args()
    config = config_util.parse_config(args.config)

    # Set plotting settings
    params = {
        'font.family': 'CMU Sans Serif',
        'axes.titlesize': 20,
        'axes.labelsize': 36,
        'axes.linewidth': 2,
        'axes.labelpad': 0,
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

    if args.explicit:
        sweep_explicit(args.data_dir, postselect=args.postselect, crop=args.crop,
                       border_crop=args.border_crop)
    else:
        config['Model']['saved_model'] = args.model
        
        if args.spat:
            submodels = [nn_models.CCNNSpatWgt(
                num_filts=config['Model Kwargs']['num_filts'],
                filter_size=config['Model Kwargs']['filter_size'],
                order=config['Model Kwargs']['order'],
                abs_coeff=config['Model Kwargs']['abs_coeff'],
                abs_filt=config['Model Kwargs']['abs_filt'])
            for _ in range(args.nclasses)]
        else:
            submodels = [nn_models.CCNN(
                num_filts=config['Model Kwargs']['num_filts'],
                filter_size=config['Model Kwargs']['filter_size'],
                order=config['Model Kwargs']['order'],
                abs_coeff=config['Model Kwargs']['abs_coeff'],
                abs_filt=config['Model Kwargs']['abs_filt'])
            for _ in range(args.nclasses)]

        model = nn_models.OVRModel(submodels)
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        model.in_channels = config['Model Kwargs']['in_channels']
        model = model.eval()

        sweep_ovr(model, args.data_dir, postselect=args.postselect, savedir=args.savedir)
