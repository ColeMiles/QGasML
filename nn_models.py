""" Defines the neural network models which can be used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_util import _vflip, _hflip, _rot, _flip_spins


class ConvNetBN(nn.Module):
    def __init__(self, input_size=10, in_channels=1, num_filts=24):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size*input_size*24, 150)
        self.bn = nn.BatchNorm1d(150, affine=False)
        self.dropout = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(150, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SymConvNetBN(nn.Module):
    def __init__(self, input_size=10, in_channels=1, num_filts=24):
        super().__init__()
        self.symfold = SymFold()
        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.sympool = SymPool()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size*input_size*num_filts, 150)
        self.relu2 = nn.ReLU()
        self.bn = nn.BatchNorm1d(150, affine=False)
        self.fc2 = nn.Linear(150, 2)

    def forward(self, x):
        x = self.symfold(x)
        x = self.relu1(self.conv1(x))
        x = self.sympool(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x


class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean((-2, -1))


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
    def __init__(self):
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
        return inv_x.mean(1)


class SymPoolNoOp(nn.Module):
    """ Like SymPool, but doesn't perform the inverse operations. For use when you want
        to pool after fully-connected layers, in which case all spatial information is lost.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0] // 8, 8, *x.shape[1:]).mean(1)


class SymPoolNoRotNoOp(nn.Module):
    """ Like SymPool, but doesn't perform the inverse operations. For use when you want
        to pool after fully-connected layers, in which case all spatial information is lost.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0] // 4, 4, *x.shape[1:]).mean(1)


class FixedSymConvLinear(nn.Module):
    def __init__(self, filters, input_size=10, in_channels=1):
        super().__init__()

        num_filts = filters.shape[0]
        if type(filters) is not torch.Tensor:
            filters = torch.tensor(filters).to(torch.float32)

        state_dict = {'weight': filters}

        self.reduction = Mean2d()

        self.symfold = SymFold()
        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.load_state_dict(state_dict, strict=False)
        self.conv1.weight.requires_grad = False
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_filts, affine=False)
        self.sympool = SymPoolNoOp()
        self.linear = nn.Linear(num_filts, 1, bias=True)

    def forward(self, x):
        x = self.symfold(x)
        x = self.relu(self.conv1(x))
        x = self.sympool(x)
        x = self.reduction(x)
        x = self.bn(x)
        x = self.linear(x)

        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
        return x


class NonlinearConvolution(nn.Module):
    """ Module defining our non-linear operations, recursively constructed from a convolutional map.
    """
    def __init__(self, in_channels=1, num_filts=8, filter_size=3, order=4, bias=False):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order

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
        abs_filt = self.conv_filt.abs()

        # Convolutions of powers of the input with powers of the absolute valued filter
        # These are necessary in general, but can be avoided if we know the input is limited
        #   to the values {0, 1}. For safety and for demonstration, the general formula used here.
        conv_powers = [F.conv2d(x ** k, abs_filt ** k, stride=1, padding=2) for k in range(1, self.order + 1)]

        # F_0 is one everywhere
        corrs = [torch.ones_like(conv_powers[0])]

        # Compute each order of F_n recursively
        for n in range(1, self.order + 1):
            Fn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                Fn += (-1) ** k * conv_powers[k] * corrs[n - k - 1]
            Fn /= n
            corrs.append(Fn)

        # Stack correlator maps along the channel dimension, dropping the zeroth order
        return torch.cat(corrs[1:], dim=-3) + self.bias


class CorrelatorExtractor(nn.Module):
    """ Architecture which extracts sums of multi-site correlators from a convolution operation.
        Main architecture examined in our work!
    """
    def __init__(self, input_size=10, in_channels=1, num_filts=8,
                 filter_size=3, order=4, absbeta=True):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order

        self.reduction = Mean2d()
        self.symfold = SymFold()
        self.sympool = SymPoolNoOp()

        self.correlator = NonlinearConvolution(in_channels, num_filts, filter_size, order, bias=False)
        self.bn = nn.BatchNorm1d(num_filts * order, affine=False, eps=1e-7)
        self.linear = nn.Linear(num_filts * order, 1, bias=True)
        self.absbeta = absbeta
        if self.absbeta:
            self.linear.weight.data.abs_()

    def forward(self, x):
        if self.absbeta:
            self.linear.weight.data.abs_()
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.sympool(x)
        x = x.sum(dim=(-1, -2))
        x = self.bn(x)
        pred = self.linear(x)
        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        pred = torch.cat((pred, torch.zeros_like(pred)), dim=-1)
        return pred


class CorrelatorExtractorLessSym(nn.Module):
    """ Architecture which extracts sums of multi-site correlators from a convolution operation.
        Main architecture examined in our work! Lower symmetry, no rotations.
    """
    def __init__(self, input_size=10, in_channels=1, num_filts=8,
                 filter_size=3, order=4, absbeta=False):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order

        self.reduction = Mean2d()
        self.symfold = SymFoldNoRot()
        self.sympool = SymPoolNoRotNoOp()

        self.correlator = NonlinearConvolution(in_channels, num_filts, filter_size, order, bias=False)
        self.bn = nn.BatchNorm1d(num_filts * order, affine=False, eps=1e-7)
        self.linear = nn.Linear(num_filts * order, 1, bias=True)
        self.absbeta = absbeta
        if self.absbeta:
            self.linear.weight.data.abs_()

    def forward(self, x):
        if self.absbeta:
            self.linear.weight.data.abs_()
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.sympool(x)
        x = x.sum(dim=(-1, -2))
        x = self.bn(x)
        pred = self.linear(x)
        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        pred = torch.cat((pred, torch.zeros_like(pred)), dim=-1)
        return pred


class ProxySymConvLinear(nn.Module):
    """ Like FixedSymConvLinear, but starting with random (trainable) filters. Purpose is for comparison of
         from-scratch learning versus pruning a complicated model
    """
    def __init__(self, num_filts=8, input_size=10, in_channels=1):
        super().__init__()
        self.reduction = Mean2d()
        self.symfold = SymFold()
        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.sympool = SymPoolNoOp()
        self.bn1 = nn.BatchNorm1d(num_filts, affine=False)
        self.linear = nn.Linear(num_filts, 1)

    def forward(self, x):
        x = self.symfold(x)
        x = self.relu(self.conv1(x))
        x = self.sympool(x)
        x = self.reduction(x)
        x = self.bn1(x)
        x = self.linear(x)

        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
        return x


class ProxySymConvTanh(nn.Module):
    """ Like ProxySymConvLinear, but with tanh as an analytic nonlinearity as opposed to ReLU
    """
    def __init__(self, num_filts=8, input_size=10, in_channels=1):
        super().__init__()
        self.reduction = Mean2d()
        self.symfold = SymFold()
        self.conv1 = nn.Conv2d(in_channels, num_filts, kernel_size=3, stride=1, padding=1, bias=True)
        self.tanh = nn.Tanh()
        self.sympool = SymPoolNoOp()
        self.bn1 = nn.BatchNorm1d(num_filts, affine=False)
        self.linear = nn.Linear(num_filts, 1)

    def forward(self, x):
        x = self.symfold(x)
        x = self.tanh(self.conv1(x))
        x = self.sympool(x)
        x = self.reduction(x)
        x = self.bn1(x)
        x = self.linear(x)

        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
        return x


class ProxySymConvLinearAbs(nn.Module):
    """ Like FixedSymConvLinear, but starting with random (trainable) filters.
         Purpose is for comparison of from-scratch learning versus pruning a complicated model.
         This version uses abs(conv_filt).
    """
    def __init__(self, num_filts=8, input_size=10, in_channels=1):
        super().__init__()
        self.reduction = Mean2d()
        self.symfold = SymFold()

        weight_sample_k = np.sqrt(1.0 / (9 * in_channels))
        init_filt = (
                2 * weight_sample_k * torch.rand(num_filts, in_channels, 3, 3) - weight_sample_k
        ).abs()
        self.conv_filt = torch.nn.Parameter(init_filt)

        init_bias = 2 * weight_sample_k * torch.rand(num_filts, 1, 1) - weight_sample_k
        self.bias = torch.nn.Parameter(init_bias)

        self.relu = nn.ReLU()
        self.sympool = SymPoolNoOp()
        self.bn1 = nn.BatchNorm1d(num_filts, affine=False)
        self.linear = nn.Linear(num_filts, 1)

    def forward(self, x):
        x = self.symfold(x)
        x = F.conv2d(x, self.conv_filt.abs(), stride=1, padding=2) + self.bias
        x = self.relu(x)
        x = self.sympool(x)
        x = self.reduction(x)
        x = self.bn1(x)
        x = self.linear(x)

        # Treat resulting number as logistic regression on class -- logit for other class should be 0
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
        return x


models = {
    'ConvNetBN': ConvNetBN,
    'SymConvNetBN': SymConvNetBN,
    'FixedConvLinear': FixedSymConvLinear,
    'FixedSymConvLinear': FixedSymConvLinear,
    'ProxySymConvLinear': ProxySymConvLinear,
    'ProxySymConvTanh': ProxySymConvTanh,
    'ProxySymConvLinearAbs': ProxySymConvLinearAbs,
    'CorrelatorExtractor': CorrelatorExtractor,
    'CorrelatorExtractorLessSym': CorrelatorExtractorLessSym,
}
