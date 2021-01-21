""" Various plotting functions to make diagrams from papers.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import torch


def plot_corr_filters(filters, num_cols=3, imshow=False, norm=True, title="", show=True):
    if type(filters) is torch.Tensor:
        filters = filters.detach().cpu().numpy()
    filters = np.abs(filters)

    num_filt = filters.shape[0]
    num_chans = filters.shape[1]

    num_rows = np.ceil(num_filt / num_cols).astype(np.int64)
    fig = plt.figure(figsize=(num_cols * 2, max(6, 3 * num_rows)))
    fig.suptitle(title)
    outergrid = fig.add_gridspec(num_rows, num_cols + 1)

    # Make color maps
    reds = plt.get_cmap('Reds')
    trunc_reds = colors.LinearSegmentedColormap.from_list(
        'TruncReds', reds(np.linspace(0.01, 1.0))
    )
    trunc_reds.set_under('white')
    blues = plt.get_cmap('Blues')
    trunc_blues = colors.LinearSegmentedColormap.from_list(
        'TruncBlues', blues(np.linspace(0.01, 1.0))
    )
    trunc_blues.set_under('white')
    greens = plt.get_cmap('Greens')
    trunc_greens = colors.LinearSegmentedColormap.from_list(
        'TruncGreens', greens(np.linspace(0.01, 1.0))
    )
    trunc_greens.set_under('white')

    hole_idx = 0 if num_chans == 1 else 2
    for i, filt in enumerate(filters):
        # Normalize the filt such that max is 1 -- be careful with filters that are all off!
        if norm:
            maxval = np.max(filt)
            # Hard zero out filter where all pixels < 1e-4 -- this filter was turned off
            if maxval < 1e-4:
                filt[:] = 0
            else:
                # Normalize filt so maxval = 1
                filt /= maxval
                # Hard zero out things < 1% of largest value
                filt[filt < 0.01] = 0
        row_idx = i % num_rows
        col_idx = i // num_rows

        innergrid = outergrid[row_idx, col_idx].subgridspec(num_chans, 1)

        if num_chans >= 2:
            ax1 = fig.add_subplot(innergrid[0, 0])
            ax1.axis('off')
            if imshow:
                im1 = ax1.imshow(filt[0], cmap=trunc_reds, vmin=0.01, vmax=1.0, origin='lower')
            else:
                im1 = ax1.pcolor(filt[0], cmap=trunc_reds, vmin=0.01, vmax=1.0, linewidth=1, edgecolors='k')
            ax1.set_aspect('equal')
            ax1.set_title(i)

            ax2 = fig.add_subplot(innergrid[1, 0])
            ax2.axis('off')
            if imshow:
                im2 = ax2.imshow(filt[1], cmap=trunc_blues, vmin=0.01, vmax=1.0, origin='lower')
            else:
                im2 = ax2.pcolor(filt[1], cmap=trunc_blues, vmin=0.01, vmax=1.0, linewidth=1, edgecolor='k')
            ax2.set_aspect('equal')

        if num_chans == 1:
            ax3 = fig.add_subplot(innergrid[hole_idx, 0])
            ax3.axis('off')
            if imshow:
                im3 = ax3.imshow(filt[hole_idx], cmap=trunc_reds, vmin=0.01, vmax=1.0, origin='lower')
            else:
                im3 = ax3.pcolor(filt[hole_idx], cmap=trunc_reds, vmin=0.01, vmax=1.0, linewidth=1,
                                 edgecolor='k')
            ax3.set_aspect('equal')
            ax3.set_title(i)
        if num_chans == 3:
            ax3 = fig.add_subplot(innergrid[hole_idx, 0])
            ax3.axis('off')
            if imshow:
                im3 = ax3.imshow(filt[hole_idx], cmap=trunc_greens, vmin=0.01, vmax=1.0, origin='lower')
            else:
                im3 = ax3.pcolor(filt[hole_idx], cmap=trunc_greens, vmin=0.01, vmax=1.0, linewidth=1, edgecolor='k')
            ax3.set_aspect('equal')

    caxgrid = outergrid[:, -1].subgridspec(1, num_chans)
    if num_chans >= 2:
        cax1 = fig.add_subplot(caxgrid[0, 0])
        # cax1.set_title('↑')
        cax1.set_title('Mag')
        cax1.axis('off')
        fig.colorbar(im1, cax=cax1, extend='min')
        cax2 = fig.add_subplot(caxgrid[0, 1])
        # cax2.set_title('↓')
        cax2.set_title('Hole')
        cax2.axis('off')
        cbar2 = fig.colorbar(im2, cax=cax2, extend='min')
        if num_chans == 2:
            cax2.axis('on')
            cbar2.set_ticks([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    if num_chans == 1:
        cax3 = fig.add_subplot(caxgrid[0, hole_idx])
        cax3.set_title('Weight')
        cbar3 = fig.colorbar(im3, cax=cax3, extend='min')
        cbar3.set_ticks([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

    if num_chans == 3:
        cax3 = fig.add_subplot(caxgrid[0, hole_idx])
        cax3.set_title('○')
        cbar3 = fig.colorbar(im3, cax=cax3, extend='min')
        cbar3.set_ticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

    if show:
        plt.show()
    else:
        return fig


def diverging_cmap(name1, name2):
    """ Stitches two matplotlib colormaps together to make a new one.
    """
    cmap1 = plt.cm.get_cmap(name1)
    cmap2 = plt.cm.get_cmap(name2)
    cmap1_arr = cmap1(np.arange(256))
    cmap2_arr = cmap2(np.arange(256))
    new_arr = np.concatenate((cmap2_arr[::-1], cmap1_arr))
    new_cmap = colors.ListedColormap(new_arr)
    return new_cmap


def plot_filters(filters, imshow=False, hole=False, num_cols=3):
    """ Plots all of the filters of a convolutional layer, along with their
          average importance values if provided. Supports both partial and full
          info filters. ([N, 1, W, H] or [N, C, W, H] shape tensors).
    """
    if type(filters) is torch.Tensor:
        filters = filters.detach().cpu().numpy()

    # Make some custom colormaps
    red_purple = diverging_cmap('Reds', 'Purples')
    blue_orange = diverging_cmap('Blues', 'Oranges')
    green_grey = diverging_cmap('Greens', 'Greys')

    is_fullinfo = filters.shape[1] == 2
    is_3chan = filters.shape[1] == 3

    num_filt = filters.shape[0]
    num_rows = np.ceil(num_filt / num_cols).astype(np.int64)
    fig = plt.figure(figsize=(num_cols * 2, max(6, 3 * num_rows)))
    colorbar_cols = 2 if is_fullinfo else 1
    outergrid = fig.add_gridspec(num_rows, num_cols + colorbar_cols)

    vmax = np.max(filters)
    vmin = np.min(filters)
    vmax = max(vmax, -vmin)
    vmin = min(-vmax, vmin)

    for i, filt in enumerate(filters):
        row_idx = i % num_rows
        col_idx = i // num_rows
        if is_fullinfo:
            innergrid = outergrid[row_idx, col_idx].subgridspec(2, 1)
            ax1 = fig.add_subplot(innergrid[0, 0])
            ax1.axis('off')
            im1 = ax1.imshow(filt[0], cmap=red_purple, vmin=vmin, vmax=vmax, origin='lower')
            ax1.set_title(i)
            ax2 = fig.add_subplot(innergrid[1, 0])
            ax2.axis('off')
            im2 = ax2.imshow(filt[1], cmap=green_grey, origin='lower')
        elif is_3chan:
            innergrid = outergrid[row_idx, col_idx].subgridspec(3, 1)
            ax1 = fig.add_subplot(innergrid[0, 0])
            ax1.axis('off')
            if imshow:
                im1 = ax1.imshow(filt[0], cmap=red_purple, vmin=vmin, vmax=vmax, origin='lower')
            else:
                im1 = ax1.pcolor(filt[0], cmap=red_purple, vmin=vmin, vmax=vmax, linewidth=1, edgecolors='k')
            ax1.set_aspect('equal')
            ax1.set_title(i)

            ax2 = fig.add_subplot(innergrid[1, 0])
            ax2.axis('off')
            if imshow:
                im2 = ax2.imshow(filt[1], cmap=blue_orange, vmin=vmin, vmax=vmax, origin='lower')
            else:
                im2 = ax2.pcolor(filt[1], cmap=blue_orange, vmin=vmin, vmax=vmax, linewidth=1, edgecolor='k')
            ax2.set_aspect('equal')

            ax3 = fig.add_subplot(innergrid[2, 0])
            ax3.axis('off')
            if imshow:
                im3 = ax3.imshow(filt[2], cmap=green_grey, vmin=vmin, vmax=vmax, origin='lower')
            else:
                im3 = ax3.pcolor(filt[2], cmap=green_grey, vmin=vmin, vmax=vmax, linewidth=1, edgecolor='k')
            ax3.set_aspect('equal')
        else:
            cmap = 'PRGn' if hole else 'bwr'
            ax1 = fig.add_subplot(outergrid[row_idx, col_idx])
            ax1.axis('off')
            im1 = ax1.imshow(filt[0], cmap=cmap)
            ax1.set_title(i)

    if is_fullinfo:
        cax1 = fig.add_subplot(outergrid[:, -1])
        fig.colorbar(im1, cax=cax1)
        cax1.set_title('Spin')
        cax2 = fig.add_subplot(outergrid[:, -2])
        fig.colorbar(im2, cax=cax2)
        cax2.set_title('Holes')
    elif is_3chan:
        caxgrid = outergrid[:, -1].subgridspec(1, 3)
        cax1 = fig.add_subplot(caxgrid[0, 0])
        cax1.set_title('↑')
        cax1.axis('off')
        fig.colorbar(im1, cax=cax1, extend='min')
        cax2 = fig.add_subplot(caxgrid[0, 1])
        cax2.set_title('↓')
        cax2.axis('off')
        fig.colorbar(im2, cax=cax2, extend='min')
        cax3 = fig.add_subplot(caxgrid[0, 2])
        cax3.set_title('Hole')
        cbar3 = fig.colorbar(im3, cax=cax3, extend='min')
    else:
        cax1 = fig.add_subplot(outergrid[:, -1])
        fig.colorbar(im1, cax=cax1)
        if hole:
            cax1.set_title('Hole Weight')
        else:
            cax1.set_title('Spin Weight')

    plt.show()
    return fig


def plot_l1_path(cs, coeffs, train_accs, val_accs, order=4, correlator=True):
    if len(coeffs.shape) == 2:
        coeffs = coeffs.reshape((1, order, coeffs.shape[-1] // order))
    if len(val_accs.shape) == 1:
        train_accs = np.expand_dims(train_accs, axis=0)
    if len(val_accs.shape) == 1:
        val_accs = np.expand_dims(val_accs, axis=0)

    num_classes, order, num_filts, num_cs = coeffs.shape

    plt.rcParams["font.family"] = "CMU Sans Serif"
    plt.rcParams["text.usetex"] = True

    fig, axes = plt.subplots(2, num_classes, figsize=(10, 10), sharex=True, squeeze=False)

    coeff_axes = axes[0, :]
    for ax in coeff_axes:
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylabel(r'$\beta$ Coeffs')
    acc_axes = axes[1, :]
    for n, ax in enumerate(acc_axes):
        ax.set_xlabel(r"$1/\lambda$")
        ax.set_xscale('log')
        ax.set_ylabel("Accuracy")
        ax.plot(cs, val_accs[n], lw=3.5, color='crimson', label="Val")
        ax.plot(cs, train_accs[n], lw=3.5, color='black', label="Train")
        ax.legend()
    fig.tight_layout()

    # Assume we'll never plot paths with more than five filters, and usually only up to fourth order.
    # First two filters can go up to sixth order
    colors = [
        ["xkcd:black", "xkcd:royal purple", "xkcd:violet", "xkcd:light purple", "xkcd:magenta", "xkcd:pink"],
        ["xkcd:burnt umber", "xkcd:dark orange", "xkcd:orange", "xkcd:pale orange", "xkcd:gold", "xkcd:yellow"],
        ["xkcd:dark red", "crimson", "xkcd:bright red", "xkcd:light red"],
        ["xkcd:dark green", "xkcd:deep green", "xkcd:emerald", "xkcd:seafoam green"],
        ["xkcd:dark blue", "xkcd:royal blue", "xkcd:azure", "xkcd:aqua blue"],
    ]

    for c in range(num_classes):
        coeff_ax = coeff_axes[c]
        cmin, cmax = np.min(coeffs[c]), np.max(coeffs[c])
        cmin = min(cmin, -cmax)
        cmax = max(cmax, -cmin)
        for n in range(order):
            for i in range(num_filts):
                if correlator:
                    try:
                        color = colors[i][n]
                    except IndexError:
                        color = None
                    if n == 1:
                        linestyle = '--'
                    else:
                        linestyle = '-'
                    coeff_ax.plot(cs, coeffs[c, n, i], color=color, lw=3.5, linestyle=linestyle)
                    coeff_ax.set_ylim(cmin, cmax)
                else:
                    coeff_ax.plot(cs, coeffs[c, n, i], lw=3.5)

    return fig, (coeff_axes, acc_axes)


def plot_rydberg_snap(snap, ax=None):
    """ Plots a single-channel Rydberg snapshot, optionally into the given `ax`.
    """
    from matplotlib.collections import CircleCollection

    W, H = snap.shape[-1], snap.shape[-2]

    plt.rc('axes', linewidth=6)

    radius = 200
    rads = radius * np.ones_like(snap.flatten())

    lattice = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).reshape(-1, 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    circs = CircleCollection(rads, offsets=lattice, cmap='Purples', transOffset=ax.transData,
                             edgecolors='black', linewidths=3,)
    circs.set_array(snap.flatten())
    ax.add_collection(circs)

    ax.autoscale_view()
    ax.set_aspect(1.0)
    ax.set_facecolor('xkcd:grey')
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False
    )

    return ax

