""" Defines the neural network models which can be used.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_util import _vflip, _hflip, _rot


class ConvNetBN(nn.Module):
    """ A standard few-layer convolutional network. """
    def __init__(self, input_size=10, input_height=None, in_channels=1, num_filts=24, num_classes=2):
        super().__init__()
        self.input_width = input_size
        self.input_height = input_size if input_height is None else input_height
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filts)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filts, 2 * num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * num_filts)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(2 * num_filts, 2 * num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * num_filts)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(2 * num_filts, 4 * num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(4 * num_filts)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * num_filts * (self.input_width // 4) * (self.input_height // 4), 150)
        self.fcbn = nn.BatchNorm1d(150, affine=False)
        self.fcrelu = nn.ReLU()
        self.fc2 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fcrelu(self.fcbn(self.fc1(x)))
        x = self.fc2(x)
        return x

## Various reduction functions as modules

class Sum2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sum((-2, -1))


class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean((-2, -1))

## Modules used for performing symmetrization

class SymFold(nn.Module):
    """ Given an array of shape [B, ..., W, H], produces an array of shape [B, 8, ..., W, H]
          of all symmetry transformations of the input, then folds the symmetry-equivalent versions
          into the batch dimension to make an array of shape [B * 8, ..., W, H]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        sym_x = torch.empty(batch_size, 8, *x.shape[1:]).to(x)
        sym_x[:, 0] = x
        sym_x[:, 1] = _hflip(x)
        sym_x[:, 2] = _vflip(x)
        sym_x[:, 3] = _rot(x, 1)
        sym_x[:, 4] = _rot(x, 2)
        sym_x[:, 5] = _rot(x, 3)
        sym_x[:, 6] = _rot(_hflip(x), 1)
        sym_x[:, 7] = _rot(_vflip(x), 1)
        x = sym_x.reshape(batch_size * 8, *x.shape[1:])
        return x


class SymFoldNoRot(nn.Module):
    """ Given an array of shape [B, ..., W, H], produces an array of shape [B, 4, ..., W, H]
          of all symmetry transformations of the input except rotations, then folds the
          symmetry-equivalent versions into the batch dimension to make an array of
          shape [B * 4, ..., W, H]
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        sym_x = torch.empty(batch_size, 4, *x.shape[1:]).to(x)
        sym_x[:, 0] = x
        sym_x[:, 1] = _hflip(x)
        sym_x[:, 2] = _vflip(x)
        sym_x[:, 3] = _hflip(_vflip(x))
        x = sym_x.reshape(batch_size * 4, *x.shape[1:])
        return x


class SymPool(nn.Module):
    """ Performs inverse symmetry operations to bring feature maps back in line, then
        averages activations pixelwise. Must operate on a feature map that has been through
        SymFold.
    """
    def __init__(self, reduction='mean'):
        if reduction == 'mean':
            self.reduction = lambda x: torch.sum(x, dim=1)
        elif reduction == 'max':
            self.reduction = lambda x: torch.max(x, dim=1)[0]
        else:
            raise ValueError("Unknown reduction method passed to SymPool")
        super().__init__()

    def forward(self, x):
        inv_x = torch.empty(x.shape[0] // 8, 8, *x.shape[1:], device=x.device)
        x = x.reshape(x.shape[0] // 8, 8, *x.shape[1:])

        inv_x[:, 0] = x[:, 0]
        inv_x[:, 1] = _hflip(x[:, 1])
        inv_x[:, 2] = _vflip(x[:, 2])
        inv_x[:, 3] = _rot(x[:, 3], -1)
        inv_x[:, 4] = _rot(x[:, 4], -2)
        inv_x[:, 5] = _rot(x[:, 5], -3)
        inv_x[:, 6] = _hflip(_rot(x[:, 6], -1))
        inv_x[:, 7] = _vflip(_rot(x[:, 7], -1))
        return self.reduction(inv_x)


class SymPoolNoOp(nn.Module):
    """ Like SymPool, but doesn't perform the inverse operations. For use when you want
        to pool after fully-connected layers, in which case all spatial information is lost.
    """
    def __init__(self, reduction='mean'):
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(1)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(1)[0]
        super().__init__()

    def forward(self, x):
        return self.reduction(x.reshape(x.shape[0] // 8, 8, *x.shape[1:]))


class SymPoolNoRotNoOp(nn.Module):
    """ Like SymPool, but doesn't perform the inverse operations. For use when you want
        to pool after fully-connected layers, in which case all spatial information is lost.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0] // 4, 4, *x.shape[1:]).mean(1)


class NonlinearConvolution(nn.Module):
    """ Module defining our non-linear operations, recursively constructed from a convolutional map.
        I.e. this module performs the convolution and ''cascading'' to higher orders.
    """
    def __init__(self, in_channels=1, num_filts=8, filter_size=3, order=4, bias=False, abs=True):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.pad = 2 * (self.filter_size // 2)
        self.abs = abs

        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        weight_sample_k = np.sqrt(1.0 / (9 * in_channels))
        init_filt = (
            2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size)
        )
        self.conv_filt = torch.nn.Parameter(init_filt)

        if bias:
            init_bias = 2 * weight_sample_k * torch.rand(order * num_filts, 1, 1) - weight_sample_k
            self.bias = torch.nn.Parameter(init_bias)
        else:
            self.bias = torch.nn.Parameter(torch.zeros(order * num_filts, 1, 1))
            self.bias.requires_grad = False

    def forward(self, x):
        if self.abs:
            abs_filt = self.conv_filt.abs()
        else:
            abs_filt = self.conv_filt

        # Convolutions of powers of the input with powers of the absolute valued filter
        # These are necessary in general, but can be avoided if we know the input is limited
        #   to the values {0, 1}. For safety and for demonstration, the general formula used here.
        conv_powers = [F.conv2d(x ** k, abs_filt ** k, stride=1, padding=self.pad)
                       for k in range(1, self.order + 1)]

        # C^(0) is one everywhere
        corrs = [torch.ones_like(conv_powers[0])]

        # Compute each order of C^(n) recursively
        for n in range(1, self.order + 1):
            Fn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                Fn += (-1) ** k * conv_powers[k] * corrs[n - k - 1]
            Fn /= n
            corrs.append(Fn)

        # Stack correlator maps along the channel dimension, dropping the zeroth order
        return torch.cat(corrs[1:], dim=-3) + self.bias

    def set_order(self, new_order):
        new_bias = torch.zeros(new_order * self.num_filts, 1, 1)
        new_bias[:self.order * self.num_filts] = self.bias.data[:new_order * self.num_filts]
        self.bias = torch.nn.Parameter(new_bias, requires_grad=self.bias.requires_grad)
        self.order = new_order


class CCNN(nn.Module):
    """ Our interpretable architecture which extracts sums of multi-site correlators
        from a convolution operation.

        Parameters
        ----------
        in_channels : int = 1
            The size of the channel dimension (1) of the input
        num_filts : int = 5
            The number of filters to learn
        filter_size : int = 5
            The spatial size of the filters
        order : int = 2
            The order to truncate the cascaded correlators at
        num_classes : int = 2
            The number of logits the model regresses
        zerofinal : bool = False
            Fixes the weights connected to the final logit to zero. When used with binary
             classification (num_classes = 2), this makes the first logit the standard
             binary-classification logistic logit.
        input_size : int = 13
            The size of the spatial dimensions (2,3) of the input
        abs_filt : bool = True
            If set, enforces the filter weights to be positive
        abs_coeff : bool = True
            If set, enforces the final layer weights to be positive
        cut_first : bool = False
            If set, removes connections from first-order correlators to the output
    """
    def __init__(self, in_channels=1, num_filts=3, filter_size=5, order=2, num_classes=2,
                 zerofinal=False, input_size=13, abs_filt=True, abs_coeff=True, cut_first=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.num_classes = num_classes
        self.zerofinal = zerofinal
        self.input_size = input_size
        self.abs_filt = abs_filt
        self.abs_coeff = abs_coeff
        self.cut_first = cut_first

        self.reduction = Sum2d()
        self.symfold = SymFold()
        self.sympool = SymPoolNoOp()
        self.correlator = NonlinearConvolution(in_channels, num_filts, filter_size, order,
                                               bias=False, abs=abs_filt)
        self.bn = nn.BatchNorm1d(num_filts * order, affine=False, eps=1e-6)
        if cut_first:
            self.linear = nn.Linear(num_filts * (order - 1), num_classes, bias=True)
        else:
            self.linear = nn.Linear(num_filts * order, num_classes, bias=True)

        self.register_buffer('linear_mask', torch.ones_like(self.linear.weight.data))
        if self.zerofinal:
            self.linear.weight.data[-1, :] = 0.0
            if abs_coeff:
                self.linear.weight.data = self.linear.weight.data.abs()
            self.linear_mask[-1, :] = 0.0

    def forward(self, x):
        if self.abs_coeff:
            self.linear.weight.data.abs_()
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.reduction(x)
        x = self.sympool(x)
        x = self.bn(x)

        # If requested, remove 1st order feats
        if self.cut_first:
            x = x[:, self.num_filts:]

        # Mask coupling to output -- all ones if self.zerofinal = False
        self.linear.weight.data.mul_(self.linear_mask)
        pred = self.linear(x)
        return pred

    def chop_pred(self, x):
        """ Outputs model latent variables pre-final Linear layer
        """
        if self.abs_coeff:
            self.linear.weight.data.abs_()
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.reduction(x)
        x = self.sympool(x)
        x = self.bn(x)
        return x


def _symmetrize(x):
    """ Symmetrizes x with respect to D4
    """
    symx = x.clone()
    symx += _hflip(x)
    symx += _vflip(x)
    symx += _rot(x, 1)
    symx += _rot(x, 2)
    symx += _rot(x, 3)
    symx += _rot(_hflip(x), 1)
    symx += _rot(_vflip(x), 1)
    return symx / 8


class SymmetricWeighting(nn.Module):
    """ Performs a spatial weighting where weights on different sites
         are tied together if symmetry equivalent.
    """
    def __init__(self, input_size=13):
        super().__init__()
        self.input_size = input_size
        init_weight = torch.ones(
            (int(input_size), int(input_size))
        )
        self.spatial_weights = nn.Parameter(
            init_weight, requires_grad=True
        )

    def forward(self, x):
        sym_wgts = _symmetrize(self.spatial_weights)
        return sym_wgts * x


class CCNNSpatWgt(nn.Module):
    """ Our interpretable architecture which extracts sums of multi-site correlators
        from a convolution operation. This version learns a spatially-weighted
        sum over the correlator maps in the reduction to the scalar correlator features.

        Parameters
        ----------
        in_channels : int = 1
            The size of the channel dimension (1) of the input
        num_filts : int = 5
            The number of filters to learn
        filter_size : int = 5
            The spatial size of the filters
        order : int = 2
            The order to truncate the cascaded correlators at
        num_classes : int = 2
            The number of logits the model regresses
        zerofinal : bool = False
            Fixes the weights connected to the final logit to zero. When used with binary
             classification (num_classes = 2), this makes the first logit the standard
             binary-classification logistic logit.
        input_size : int = 13
            The size of the spatial dimensions (2,3) of the input
        abs_filt : bool = True
            If set, enforces the filter weights to be positive
        abs_coeff : bool = True
            If set, enforces the final layer weights to be positive
        cut_first : bool = False
            If set, removes connections from first-order correlators to the output
    """
    def __init__(self, in_channels=1, num_filts=3, filter_size=5, order=2, num_classes=2,
                 zerofinal=True, input_size=13, abs_filt=True, abs_coeff=True, cut_first=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.num_classes = num_classes
        self.zerofinal = zerofinal
        self.input_size = input_size
        self.abs_filt = abs_filt
        self.abs_coeff = abs_coeff
        self.cut_first = cut_first

        self.reduction = Sum2d()
        self.symfold = SymFold()
        self.sympool = SymPoolNoOp(reduction='mean')

        self.correlator = NonlinearConvolution(in_channels, num_filts, filter_size, order,
                                               bias=False, abs=abs_filt)
        spat_size = input_size + 4 * (filter_size // 2) - filter_size + 1
        self.spatial_weighting = SymmetricWeighting(spat_size)

        self.bn = nn.BatchNorm1d(num_filts * order, affine=False, eps=1e-6)
        if cut_first:
            self.linear = nn.Linear(num_filts * (order - 1), num_classes, bias=True)
        else:
            self.linear = nn.Linear(num_filts * order, num_classes, bias=True)
        self.register_buffer('linear_mask', torch.ones_like(self.linear.weight.data))
        if self.zerofinal:
            self.linear.weight.data[-1, :] = 0.0
            if abs_coeff:
                self.linear.weight.data = self.linear.weight.data.abs()
            self.linear_mask[-1, :] = 0.0

    def forward(self, x):
        if self.abs_coeff:
            self.linear.weight.data.abs_()
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.spatial_weighting(x)
        x = self.reduction(x)
        x = self.sympool(x)
        x = self.bn(x)

        # TEMPORARY: Remove 1st-order feats
        if self.cut_first:
            x = x[:, self.num_filts:]

        # Mask coupling to output -- all ones if self.zerofinal = False
        self.linear.weight.data.mul_(self.linear_mask)
        pred = self.linear(x)
        return pred

    def chop_pred(self, x):
        """ Outputs model latent variables pre-final Linear layer
        """
        if self.abs_coeff:
            self.linear.weight.data.abs_()

        x = self.symfold(x)
        x = self.correlator(x)
        x = self.spatial_weighting(x)
        x = self.reduction(x)
        x = self.sympool(x)
        x = self.bn(x)
        return x


class OVRModel(nn.Module):
    """ Constructs a classifier from a bundle of `One-vs-Rest` models"""
    def __init__(self, models, **kwargs):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.num_classes = len(models)
        # Sort of a hack: Allow arbitrary kwargs to be associated to this model for other scripts
        self.__dict__.update(kwargs)

    def forward(self, x):
        """ Predict on x using each of the models, returns the logits for class 0 from each model
        """
        pred_logits = torch.empty(x.shape[0], self.num_classes)
        for i, model in enumerate(self.models):
            pred_logits[:, i] = model(x)[:, 0]
        return pred_logits


def from_config(config: Dict) -> torch.nn.Module:
    """ Constructs a model from a configuration file.
    """
    model_type = config['Model']['model']
    if config['Model']['fixed_filts'] is not None:
        filters = np.load(config['Model']['fixed_filts'])
        if len(filters.shape) < 4:
            filters = np.expand_dims(filters, 1)
        config['Model Kwargs']['fixed_filts'] = filters
    model = models[model_type](**config['Model Kwargs'])

    # If provided, load previously saved model parameters.
    if config['Model']['saved_model'] is not None:
        state_dict = torch.load(config['Model']['saved_model'], map_location='cpu')
        model.load_state_dict(state_dict)

    return model


models = {
    'ConvNetBN': ConvNetBN,
    'CCNN': CCNN,
    'CCNNSpatWgt': CCNNSpatWgt,
}
