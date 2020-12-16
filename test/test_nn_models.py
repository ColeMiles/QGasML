import os
import sys
import itertools

import numpy as np
import torch
import pytest

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

import nn_models


@pytest.fixture
def fake_snapshots():
    """ Return a batch of 5 random 'snapshots' """
    return torch.randint(2, (5, 2, 10, 10)).float()


def test_nonlinear_conv(fake_snapshots):
    """ Test that CorrelatorExtractor computes the equations I claim it does.
    """
    # To reduce floating-point error issues, use doubles
    snapshots = fake_snapshots.double()
    corr_ext = nn_models.NonlinearConvolution(in_channels=2, num_filts=1).double()
    corr_ext.eval()

    filt = corr_ext.conv_filt.data.abs()

    # Test a bunch of random cases
    # Manual computation is super slow, so 5 will have to be sufficient for "a bunch"
    for snap in snapshots:
        snap = snap.reshape((1, 2, 10, 10))
        corr_output = corr_ext(snap)

        # Brute force compute the exact equation
        true_output = torch.zeros_like(corr_output)
        # Padding is necessary for the equation to exactly hold; the model does this internally
        padded_snap = torch.tensor(np.pad(snap, [(0,), (0,), (2,), (2,)], 'constant'))
        for order in range(1, 5):
            for i, j in itertools.product(range(true_output.shape[-2]), range(true_output.shape[-1])):
                conv_window = padded_snap[0, :, i:i+3, j:j+3]
                mult_filt = (conv_window * filt).flatten()
                true_output[0, order-1, i, j] = unique_functions[order-1](mult_filt)

        assert torch.allclose(corr_output, true_output)


def test_sym_ops(fake_snapshots):
    """ Test various things about SymFold / SymPool
    """
    symfold = nn_models.SymFold()
    sympool = nn_models.SymPool()

    folded = symfold(fake_snapshots)
    # Check shape is correct
    assert folded.shape[0] == 8 * fake_snapshots.shape[0]
    assert folded.shape[1:] == fake_snapshots.shape[1:]

    pooled = sympool(folded)
    # Check shape is correct
    assert pooled.shape == fake_snapshots.shape

    # Check that the inverse symmetry ops were done correctly
    # If so, the final "average" will give the original snapshot
    assert torch.all(torch.eq(fake_snapshots, pooled))


def _unique_order1(x):
    return x.sum()


def _unique_order2(x):
    sum = 0.0
    for i in range(len(x) - 1):
        for j in range(i+1, len(x)):
            sum += x[i] * x[j]
    return sum


def _unique_order3(x):
    sum = 0.0
    for i in range(len(x) - 2):
        for j in range(i+1, len(x)-1):
            for k in range(j+1, len(x)):
                sum += x[i] * x[j] * x[k]
    return sum


def _unique_order4(x):
    sum = 0.0
    for i in range(len(x) - 3):
        for j in range(i+1, len(x) - 2):
            for k in range(j+1, len(x) - 1):
                for l in range(k+1, len(x)):
                    sum += x[i] * x[j] * x[k] * x[l]
    return sum


unique_functions = [_unique_order1, _unique_order2, _unique_order3, _unique_order4]
