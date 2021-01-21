#!/usr/bin/python
import argparse
import os
import sys
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(SCRIPTPATH))
sys.path.append(REPODIR)

import nn_models
import config_util
import data_util


def load_data(datadir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Analogous to data_util.load_rydberg_data, but also outputs the postselection maps.
        Returns:
            (train_snaps, train_postselect, val_snaps, val_postselect)
    """
    # Find the datafiles
    datafiles = [
        os.path.join(datadir, f) for f in os.listdir(datadir)
        if f.count('rydberg') > 0
    ]

    train_snaps = []
    train_mask = []
    val_snaps = []
    val_mask = []
    for npz_file in datafiles:
        loaded_npz = np.load(npz_file)
        snaps = loaded_npz['rydberg_populations']
        snaps = np.transpose(snaps, (2, 1, 0)).astype(np.float32)

        # Normalize density of snapshots to be 1 per site on both channels
        snaps /= np.mean(snaps)

        mask_npz = npz_file.replace('rydberg', 'rearrangement_mask')
        loaded_mask_npz = np.load(mask_npz)
        mask = loaded_mask_npz['rearrangement_mask']

        # Train/val split is first 90% / last 10% of the data
        split_idx = int(0.9 * len(snaps))
        train_snaps.append(snaps[:split_idx])
        train_mask.append(mask[:split_idx])
        val_snaps.append(snaps[split_idx:])
        val_mask.append(mask[split_idx:])

    train_snaps = np.concatenate(train_snaps, axis=0)
    train_mask = np.concatenate(train_mask, axis=0)
    val_snaps = np.concatenate(val_snaps, axis=0)
    val_mask = np.concatenate(val_mask, axis=0)
    train_snaps = torch.tensor(train_snaps)
    train_mask = torch.tensor(train_mask)
    val_snaps = torch.tensor(val_snaps)
    val_mask = torch.tensor(val_mask)
    return train_snaps, train_mask, val_snaps, val_mask


def preprocess(model, train_snaps):
    """Pass all the snaps through the model so that BatchNorm is set correctly."""
    model.train()
    model.bn.reset_running_stats()
    batch_size = 256
    num_batches = int(np.ceil(len(train_snaps) / 256))
    for _ in range(20):
        for i in range(num_batches):
            batch = train_snaps[batch_size * i:batch_size * (i + 1)].to(device='cuda')
            preds = model(batch)

    return model


def val(model, snaps, labels):
    model.eval()

    acc_total = 0
    batch_size = 256
    num_batches = int(np.ceil(len(snaps) / 256))
    for i in range(num_batches):
        batch = snaps[batch_size * i:batch_size * (i + 1)].to(device='cuda')
        batch_labels = labels[batch_size * i:batch_size * (i + 1)]
        preds = model(batch)
        pred_classes = preds.argmax(dim=1).cpu()
        acc_total += torch.count_nonzero(torch.eq(pred_classes, batch_labels)).item()
    return acc_total / len(snaps)


def incorrect_slideshow(incorrect_snaps, incorrect_labels, incorrect_scores, incorrect_mask,
                        classes=['Checker', 'Star', 'Striated']):
    """ Show 4 snapshots we got wrong at a time. """
    plt.ion()
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    incorrect_snaps = incorrect_snaps.cpu().numpy()
    incorrect_scores = incorrect_scores.cpu().numpy()

    for i in range(len(incorrect_snaps) // 4):
        show_snaps = incorrect_snaps[4*i:4*(i+1)]
        show_labels = incorrect_labels[4*i:4*(i+1)]
        show_scores = incorrect_scores[4*i:4*(i+1)]
        show_mask = incorrect_mask[4*i:4*(i+1)]

        show_scores = np.round(show_scores, 2)

        for j in range(4):
            axs[j].imshow(show_snaps[j, 0])
            axs[j].axis('off')
            title = "Class: {}, Pred: ({:.2f}, {:.2f}, {:.2f})".format(
                classes[show_labels[j]], *show_scores[j], show_mask[j]
            )
            axs[j].set_title(title, fontsize=20)

        input("Next?")


def error_analyze(model, snaps, labels, mask):
    model.eval()

    # What fraction of all snapshots have rearrangement errors?
    rearrange_err = torch.count_nonzero(torch.logical_not(mask)) / len(mask)

    # Of the snapshots the model is incorrect on, what fraction have rearrangement error?
    batch_size = 256
    num_batches = int(np.ceil(len(snaps) / 256))
    incorrect_snaps = []
    incorrect_labels = []  # Label OF the correct one, not the incorrect label
    incorrect_scores = []
    incorrect_mask = []

    with torch.no_grad():
        for i in range(num_batches):
            batch = snaps[batch_size*i:batch_size*(i+1)].to(device='cuda')
            batch_mask = mask[batch_size*i:batch_size*(i+1)]
            batch_labels = labels[batch_size*i:batch_size*(i+1)]
            preds = model(batch)
            scores = torch.nn.functional.softmax(preds, dim=-1)
            pred_classes = scores.argmax(dim=1).cpu()

            incorrect = torch.not_equal(pred_classes, batch_labels)
            incorrect_snaps.append(batch[incorrect])
            incorrect_labels.append(batch_labels[incorrect])
            incorrect_scores.append(scores[incorrect])
            incorrect_mask.append(batch_mask[incorrect])

    incorrect_snaps = torch.cat(incorrect_snaps)
    incorrect_labels = torch.cat(incorrect_labels)
    incorrect_scores = torch.cat(incorrect_scores)
    incorrect_mask = torch.cat(incorrect_mask)
    incorrect_rearrange_err = torch.count_nonzero(torch.logical_not(incorrect_mask)) / len(incorrect_mask)
    print("Orig Rearrange Err:", rearrange_err, ", Inc Rearrange Err:", incorrect_rearrange_err)

    incorrect_slideshow(incorrect_snaps, incorrect_labels, incorrect_scores, incorrect_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to a saved .pt torch model.')
    parser.add_argument('config', type=str, help='Path to config of training run.')
    args = parser.parse_args()
    config = config_util.parse_config(args.config)

    config['Model']['saved_model'] = args.model
    model = nn_models.from_config(config).to(device='cuda')

    train_snaps = []
    train_masks = []
    val_snaps = []
    val_masks = []
    base_data_dir = config['Data Files']['base_dir']
    for dset in config['Data Files']['datasets']:
        data_dir = os.path.join(base_data_dir, "Dataset{}".format(dset))
        ts, tm, vs, vm = load_data(data_dir)
        train_snaps.append(ts)
        train_masks.append(tm)
        val_snaps.append(vs)
        val_masks.append(vm)

    train_labels = torch.cat(tuple(
        torch.full((len(snaps),), label, dtype=torch.int64)
        for label, snaps in enumerate(train_snaps)
    ), dim=0)
    train_snaps = torch.cat(train_snaps)
    train_snaps = train_snaps.unsqueeze(1)
    train_masks = torch.cat(train_masks)
    val_labels = torch.cat(tuple(
        torch.full((len(snaps),), label, dtype=torch.int64)
        for label, snaps in enumerate(val_snaps)
    ), dim=0)
    val_snaps = torch.cat(val_snaps)
    val_snaps = val_snaps.unsqueeze(1)
    val_masks = torch.cat(val_masks)

    init_acc = val(model, val_snaps, val_labels)
    print("Model Val Accuracy: ", init_acc)

    error_analyze(model, val_snaps, val_labels, val_masks)
