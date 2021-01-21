import os
import sys
import itertools

import numpy as np
import torch
import pytest

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

import data_util


@pytest.fixture
def fake_snapshots():
    """ Return a batch of 24 random 'snapshots' """
    return torch.randint(2, (24, 2, 10, 10)).float()


def test_center_crop(fake_snapshots):
    cropped = data_util._center_crop(fake_snapshots, 6)
    assert cropped.shape[-1] == cropped.shape[-2] == 6
    assert torch.all(torch.eq(cropped, fake_snapshots[..., 2:-2, 2:-2]))


def test_circle_crop(fake_snapshots):
    cropped = data_util._circle_crop(fake_snapshots.numpy())

    assert np.all(cropped[..., 0, 0:3] == 0)
    assert np.all(cropped[..., 0, -3:] == 0)
    assert np.all(cropped[..., -1, 0:3] == 0)
    assert np.all(cropped[..., -1, -3:] == 0)
    assert np.all(cropped[..., 0:3, 0] == 0)
    assert np.all(cropped[..., -3:, 0] == 0)
    assert np.all(cropped[..., 0:3, -1] == 0)
    assert np.all(cropped[..., -3:, -1] == 0)


def test_balanced_sampler():
    num_class_zero = 10
    num_class_one = 97
    num_max_class = max(num_class_zero, num_class_one)

    labels = torch.tensor([0] * num_class_zero + [1] * num_class_one)
    sampler = data_util.BalancedSampler(labels)
    samples = np.array(list(iter(sampler)))

    assert len(sampler) == len(samples)

    # Check that the number of class zero indices is num_max_class
    class_zero_samples = samples < 10
    assert np.count_nonzero(class_zero_samples) == num_max_class

    # Check that the number of each index oversampled is equal up to +- 1
    # That is, that the snapshots are oversampled equally
    index_counts = np.bincount(samples[class_zero_samples])
    zero_count = index_counts[0]
    for count in index_counts:
        assert abs(zero_count - count) <= 1
