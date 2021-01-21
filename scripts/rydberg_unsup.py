#!/usr/bin/python
import argparse
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

# While not clean, it's a lot easier to just hard code these Rb at which we have data files
radiuses = [1.01, 1.05, 1.13, 1.23, 1.30, 1.39, 1.46, 1.56, 1.65, 1.71, 1.81, 1.89, 1.97]


def make_mesh(xs, ys):
    """ Given a set of (xs, ys) that are the center of rectangular pixels, returns
         the grids of cell boundaries defined to be halfway between the xs and ys
    """
    xs = np.array(xs)
    ys = np.array(ys)

    x_dists = xs[1:] - xs[:-1]
    y_dists = ys[1:] - ys[:-1]
    x_edges = np.concatenate((
        [xs[0] - x_dists[0] / 2],
        xs[:-1] + x_dists / 2,
        [xs[-1] + x_dists[-1] / 2],
    ))
    y_edges = np.concatenate((
        [ys[0] - y_dists[0] / 2],
        ys[:-1] + y_dists / 2,
        [ys[-1] + y_dists[-1] / 2],
    ))
    return np.meshgrid(x_edges, y_edges)


def phase_detect_fourier(data_dir, dropin=False, mask=False):
    """ Perform Fourier-based phase detection.
            1) Collect density-invariant Fourier intensities at each (Δ, Rb)
            2) Perform PCA to reduce these intensitites down to a few variables
                -> Already, some of these are easily interpretable and point out interesting phases
            3) Perform a Gaussian mixture model clustering to pull out rough guesses at phases
    """
    detune_npz = np.load(os.path.join(data_dir, 'detunings.npz'))
    detunings = detune_npz['detunings']   # Available Δ values

    det_edges, r_edges = make_mesh(detunings, radiuses)

    ## (1) Collect density-invariant Fourier intensities ##
    ## [For a different application, this feature-engineering stage will look different] ##
    input_size, fourier_size = 13, 16
    fourier_feats = np.empty((len(radiuses), len(detunings), fourier_size * fourier_size))
    fourier_errs = np.empty((len(radiuses), len(detunings), fourier_size * fourier_size))
    for i, radius in enumerate(radiuses):
        snap_npz_filename = "detuning_sweep_Rb_{:.2f}.npz".format(radius)
        snap_npz = np.load(os.path.join(data_dir, snap_npz_filename))
        all_snaps = snap_npz['rydberg_populations_per_param']

        # Transpose to [NDETUNE, NSNAPS, WIDTH, HEIGHT]
        all_snaps = np.transpose(all_snaps, (2, 3, 0, 1)).astype(np.float32)

        mask_npz_filename = snap_npz_filename.replace('detuning_sweep', 'rearrangement_mask')
        mask_npz = np.load(os.path.join(data_dir, mask_npz_filename))
        all_masks = mask_npz['rearrangement_mask']

        for j, detuning in enumerate(detunings):
            snaps = all_snaps[j]

            if mask:
                snaps = snaps[all_masks[j]]

            # Normalize density of snapshots to be 0-mean on average
            snaps -= np.mean(snaps)

            # FFT!
            fft_snaps = np.fft.fft2(snaps, s=(fourier_size, fourier_size))
            fft_snaps = np.square(np.abs(fft_snaps))

            # To finally make these density-invariant, subtract off the background.
            fft_snaps -= np.mean(fft_snaps)

            # Flatten out intensities into a long 256-dimensional vector
            fourier_feats[i, j] = np.mean(fft_snaps, axis=0).flatten()
            fourier_errs[i, j] = np.var(fft_snaps, axis=0).flatten() / np.sqrt(len(fft_snaps))


    ## (2) Perform and visualize PCA ##
    orig_shape = fourier_feats.shape
    # Flatten out (Δ, Rb) space into one dimension to pass into PCA
    fourier_feats = fourier_feats.reshape(-1, fourier_feats.shape[-1])
    pca_comps = 14
    pca = PCA(pca_comps)

    pca.fit(fourier_feats)
    # These are the actual principal component vectors
    # This is shape [NPCA, NFOURIER = 16 x 16]
    pca_vecs = pca.components_
    # These are the magnitude of the projection of the Fourier features onto each component
    # This is now shape [NRAD * NDET, NPCA]
    fourier_feats = pca.transform(fourier_feats)

    # The overall sign on a PCA component is arbitrary: the negative of PCA5 is a bit easier
    #  to visualize and explain
    pca_vecs[4, :] *= -1
    fourier_feats[:, 4] *= -1

    if dropin:
        import IPython
        IPython.embed()

    # Plot fall-off of explained variance
    plt.plot(np.arange(1, pca_comps+1), pca.explained_variance_ratio_, "o-")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance fraction")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


    ## (3) Perform a GMM clustering with 6 clusters ##
    # This is just a quick trial run for fast visualization -- the "real" clusters
    #   are obtained below by repeating this until no improvement has been seen
    #   for some number of trials
    gaus = GaussianMixture(n_components=6, n_init=100, random_state=1111)
    clusts = gaus.fit_predict(fourier_feats.reshape(-1, pca_comps))

    # Plot these clusters in (Δ, Rb) space
    clusts = clusts.reshape((*orig_shape[:2],))
    plt.pcolormesh(det_edges, r_edges, clusts, cmap='Set1')
    plt.xlabel(r'$\Delta / \Omega$')
    plt.ylabel(r'$R_b / a$')
    plt.xticks([-2, 0, 2, 4])
    plt.yticks([1, 1.2, 1.4, 1.6, 1.8, 1.9])
    plt.tight_layout()

    # Visualize the data/clusters in the space of [PCA1, PCA2, PCA5]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = fourier_feats[:, 0].reshape(orig_shape[:2])
    ys = fourier_feats[:, 1].reshape(orig_shape[:2])
    zs = fourier_feats[:, 4].reshape(orig_shape[:2])
    ax.plot_wireframe(xs, ys, zs, linewidths=1, colors='k')
    ax.scatter(fourier_feats[:, 0], fourier_feats[:, 1], fourier_feats[:, 4],
               c=clusts.flatten(), cmap='Set1')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC5")
    plt.show()

    # Unfold the (Δ, Rb) dimensions back out, to shape [NRAD, NDET, NPCA]
    fourier_feats = fourier_feats.reshape((*orig_shape[:2], pca_comps))
    
    # Some extra visualizations from the appendix:

    # Visualize the top 12 PCA components
    make_pca_plot(pca_vecs, fourier_feats, det_edges, r_edges, n_rows=2, fourier_size=fourier_size)
    # Perform a sweep over the number of PCA components kept in the clustering
    make_vary_pca_plots(fourier_feats, det_edges, r_edges, n_rows=4)
    # Perform a sweep over the number of clusters fit to the data
    make_clusterings_plots(fourier_feats, det_edges, r_edges, n_rows=2)


def make_pca_plot(pca_vecs, fourier_feats, det_edges, r_edges,
                  n_rows=1, n_cols=None, fourier_size=16):
    """ Make an expansive plot showing the top 12 PCA components, and the value of each
        of them everywhere in (Δ, Rb) space.
    """
    print("Preparing Top-12 PCA visualization plot...")
    num_pca_comps = pca_vecs.shape[0]
    if n_cols is None:
        n_cols = int(np.ceil(num_pca_comps / n_rows))

    fig, axs = plt.subplots(2*n_rows, n_cols, figsize=(7*n_cols, 11*n_rows),
                            gridspec_kw={"wspace": 0.1, "hspace": 0.4, "top": 0.97, "bottom": 0.05,
                                         "left": 0.04, "right": 0.97})

    # Custom color map going from black to yellow
    cdict = {'red': [[0.0, 0.0, 0.0],
                     [0.5, 1.0, 1.0],
                     [0.6, 245 / 255, 245 / 255],
                     [1.0, 132 / 255, 132 / 255]],
             'green': [[0.0, 0.0, 0.0],
                       [0.5, 1.0, 1.0],
                       [0.6, 218 / 255, 218 / 255],
                       [1.0, 130 / 255, 130 / 255]],
             'blue': [[0.0, 0.0, 0.0],
                      [0.5, 1.0, 1.0],
                      [0.6, 10 / 255, 10 / 255],
                      [1.0, 26 / 255, 26 / 255]]}
    newcmp = colors.LinearSegmentedColormap('ylblk', segmentdata=cdict, N=256)

    for i in range(num_pca_comps):
        row, col = divmod(i, n_cols)

        fourier_vec = pca_vecs[i]
        # Un-flatten back out into the [16 x 16] 2D fourier space
        fourier_vec = fourier_vec.reshape((fourier_size, fourier_size))
        # Add on the extra row/column to add 2π to k space
        fourier_vec = np.pad(fourier_vec, ((0, 1), (0, 1)), mode='wrap')

        # Add the top panel: The actual principal component in Fourier space
        im = axs[2*row, col].imshow(fourier_vec, norm=colors.CenteredNorm(),
                                    cmap=newcmp, origin='lower')
        axs[2*row, col].set_xlabel(r'$k_x$')
        axs[2*row, col].set_ylabel(r'$k_y$')
        axs[2*row, col].set_xticks(fourier_size * np.array([0, 1/4, 1/2, 3/4, 1]))
        axs[2*row, col].set_yticks(fourier_size * np.array([0, 1/4, 1/2, 3/4, 1]))
        axs[2*row, col].set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        axs[2*row, col].set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

        divider = make_axes_locatable(axs[2*row, col])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=18, direction='inout')
        cbar.ax.xaxis.set_ticks_position('top')
        axs[2*row, col].text(fourier_size / 16, (7 * fourier_size) / 8,
                             "PC{}".format(i+1), fontweight='bold', fontsize=30)

        # Add the bottom panel: The projection onto this component across (Δ, Rb) space
        m = axs[2*row+1, col].pcolormesh(det_edges, r_edges, fourier_feats[..., i], cmap='inferno')
        axs[2*row+1, col].set_xlabel(r'$\Delta / \Omega$')
        axs[2*row+1, col].set_xticks([-2, 0, 2, 4])
        axs[2*row+1, col].set_yticks([1, 1.2, 1.4, 1.6, 1.8, 2])
        if col == 0:
            axs[2*row, col].set_ylabel(r"$k_y$")
            axs[2 * row + 1, col].set_ylabel(r'$R_b / a$')
        if col != 0:
            axs[2*row+1, col].set_yticklabels(["", "", "", "", "", ""])
        axs[2*row+1, col].tick_params(axis='both', direction='inout')

        divider = make_axes_locatable(axs[2*row+1, col])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = plt.colorbar(m, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=18, direction='inout')
        cbar.ax.xaxis.set_ticks_position('top')

    plt.show()


def make_vary_pca_plots(fourier_feats, det_edges, r_edges,
                        min_pca=3, n_rows=3, random_state=1111,
                        max_trials=2000, converge_trials=500):
    """ Re-perform the clustering many times, each time keeping a different number of PCA
         components. For each number, repeat the clustering until convergence (a large
         number of sequential clusterings don't improve over the previous best result).
    """
    print("Preparing Varying-Num-PCA plot...")
    num_pca_comps = fourier_feats.shape[-1]
    n_cols = int(np.ceil((num_pca_comps-min_pca) / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6.4*n_cols, 4.8*n_rows),
                            gridspec_kw={"wspace": 0.1, "hspace": 0.1, "top": 0.97, "bottom": 0.1,
                                         "left": 0.04, "right": 0.97})
    orig_shape = fourier_feats.shape
    fourier_feats = fourier_feats.reshape(-1, num_pca_comps)
    for npca in range(min_pca, num_pca_comps+1):
        print("\tNPCA =", npca)
        row, col = divmod(npca - min_pca, n_cols)
        bestscore, bestgaus = -float('inf'), None
        num_not_improved = 0
        for trial in range(max_trials):
            gaus = GaussianMixture(
                n_components=6, n_init=1, random_state=random_state+trial,
                ## Interestingly, activating this improves final log-likelihood,
                ## but makes visually worse clusters. Overfitting to noise?
                # init_params='random'
            )
            gaus.fit(fourier_feats[..., :npca])
            score = gaus.score(fourier_feats[..., :npca])
            if score > bestscore:
                bestgaus = gaus
                bestscore = score
                num_not_improved = 0
            else:
                num_not_improved += 1
                if num_not_improved >= converge_trials:
                    print("\t\tConverged after {} trials".format(trial+1))
                    break
        print("\t\tBest Score:", bestscore)
        dumppath = os.path.join(REPODIR, 'scripts', 'clusterings', 'NPCA{}.pkl'.format(npca))
        pickle.dump(bestgaus, open(dumppath, 'wb'))

        clusts = bestgaus.predict(fourier_feats[..., :npca])
        clusts = clusts.reshape((*orig_shape[:2],))
        axs[row, col].pcolormesh(det_edges, r_edges, clusts, cmap='Set1')
        axs[row, col].set_xticks([-2, 0, 2, 4])
        axs[row, col].set_yticks([1, 1.2, 1.4, 1.6, 1.8, 1.9])
        if row == n_rows - 1:
            axs[row, col].set_xlabel(r'$\Delta / \Omega$')
        else:
            axs[row, col].set_xticklabels(["", "", "", ""])
        if col == 0:
            axs[row, col].set_ylabel(r'$R_b / a$')
        else:
            axs[row, col].set_yticklabels(["", "", "", "", "", ""])
        axs[row, col].text(-1.9, 1.9, str(npca), fontweight='bold', fontsize=40)

    plt.show()


def make_clusterings_plots(fourier_feats, det_edges, r_edges,
                           n_clusts=9, n_rows=2, random_state=1111,
                           max_trials=2000, converge_trials=500):
    """ Re-perform the clustering many times, each time fitting a different number of
         Gaussians in the mixture. For each number, repeat the clustering until convergence
         (a large number of sequential clusterings don't improve over the previous best result).
    """
    print("Preparing Varying-Num-Clusters plot...")
    n_cols = int(np.ceil((n_clusts-1) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6.4*n_cols, 4.8*n_rows),
                            gridspec_kw={"wspace": 0.1, "hspace": 0.1, "top": 0.97, "bottom": 0.1,
                                         "left": 0.04, "right": 0.97})
    orig_shape = fourier_feats.shape
    pca_comps = fourier_feats.shape[-1]
    fourier_feats = fourier_feats.reshape(-1, pca_comps)
    scores = []
    bcrits = []
    for nc in range(2, n_clusts+1):
        print("\tNum Clusters =", nc)
        row, col = divmod(nc-2, n_cols)
        bestscore, bestgaus = -float('inf'), None
        num_trials, num_not_improved = 0, 0
        for trial in range(max_trials):
            gaus = GaussianMixture(n_components=nc, n_init=1, random_state=random_state+trial)
            score = gaus.score(fourier_feats)
            if score > bestscore:
                bestgaus = gaus
                bestscore = score
                num_not_improved = 0
            else:
                num_not_improved += 1
                if num_not_improved >= converge_trials:
                    print("\tConverged after {} trials".format(trial+1))
                    break
            num_trials += 1

        scores.append(bestgaus.score(fourier_feats))
        bcrits.append(bestgaus.bic(fourier_feats))

        clusts = bestgaus.predict(fourier_feats)
        clusts = clusts.reshape((*orig_shape[:2],))
        axs[row, col].pcolormesh(det_edges, r_edges, clusts, cmap='Set1')
        axs[row, col].set_xticks([-2, 0, 2, 4])
        axs[row, col].set_yticks([1, 1.2, 1.4, 1.6, 1.8, 1.9])
        if row == n_rows - 1:
            axs[row, col].set_xlabel(r'$\Delta / \Omega$')
        else:
            axs[row, col].set_xticklabels(["", "", "", ""])
        if col == 0:
            axs[row, col].set_ylabel(r'$R_b / a$')
        else:
            axs[row, col].set_yticklabels(["", "", "", "", "", ""])
        axs[row, col].text(-1.9, 1.9, str(nc), fontweight='bold', fontsize=40)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(range(2, n_clusts+1), scores, "o-", lw=3)
    ax2.set_xlabel("Num Clusters")
    ax2.set_ylabel("Log Likelihood")
    ax2.set_tight_layout()

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.set_plot(range(2, n_clusts+1), bcrits, "o-", lw=3)
    ax3.set_xlabel("Num Clusters")
    ax3.set_ylabel("BIC")
    ax3.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sweeps', type=str, default=None,
                        help='Directory containing linear sweep data.')
    parser.add_argument('-d', '--dropin', action='store_true',
                        help='If set, drop into an IPython instance after features have been'
                             ' collected, but no plots have been made. [Requires IPython].')
    parser.add_argument('-m', '--mask', action='store_true',
                        help='If set, only measure statistics from snapshots with no '
                             'rearrangement errors. [Paper does not use this option].')
    args = parser.parse_args()

    params = {
        'font.family': 'CMU Sans Serif',
        'axes.titlesize': 30,
        'axes.labelsize': 28,
        'axes.linewidth': 2,
        'axes.labelpad': 10, # 2D
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

    phase_detect_fourier(args.sweeps, dropin=args.dropin, mask=args.mask)
