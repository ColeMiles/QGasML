import argparse
import os
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import nn_models
from plot_util import plot_l1_path
from train import create_datasets

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(SCRIPTPATH)


class ChoppedFixedSymConv(nn.Module):
    """ Like nn_models.FixedSymConvLinear, but just returns the summed activations
         instead of performing the logistic regression inside the model.
        Also, BatchNorm layer is moved to end.
    """
    def __init__(self, filters):
        super().__init__()

        self.n_filts = filters.shape[0]
        self.in_channels = filters.shape[1]
        if type(filters) is not torch.Tensor:
            filters = torch.tensor(filters).to(torch.float32)

        state_dict = {'weight': filters}
        self.symfold = nn_models.SymFold()
        self.conv1 = nn.Conv2d(self.in_channels, self.n_filts, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.load_state_dict(state_dict, strict=False)
        self.conv1.weight.requires_grad = False
        self.relu = nn.ReLU()
        self.sympool = nn_models.SymPoolNoOp()
        self.reduction = nn_models.Mean2d()
        self.bn = nn.BatchNorm1d(self.n_filts, affine=False, momentum=None)

    def forward(self, x):
        x = self.symfold(x)
        x = self.relu(self.conv1(x))
        x = self.sympool(x)
        x = self.reduction(x)
        x = self.bn(x)
        return x


class ChoppedCorrelatorExtractor(nn.Module):
    """ Like nn_models.CorrelatorExtractor, but holds the filters fixed and just returns the
         averaged correlations instead of performing the logistic regression inside the model.
    """
    def __init__(self, filters, input_size=10, order=4, abs=True):
        super().__init__()
        self.input_size = input_size
        self.num_filts = filters.shape[0]
        self.in_channels = filters.shape[1]
        self.filter_size = filters.shape[-1]
        self.n_sym = 8
        self.order = order

        self.reduction = nn_models.Mean2d()
        self.symfold = nn_models.SymFold()
        self.sympool = nn_models.SymPoolNoOp()

        if abs:
            self.correlator = nn_models.NonlinearConvolution(
                self.in_channels, self.num_filts, self.filter_size, order
            )
        else:
            self.correlator = nn_models.NonlinearConvolutionNoAbs(
                self.in_channels, self.num_filts, self.filter_size, order
            )

        self.correlator.conv_filt.data = filters
        self.correlator.conv_filt.requires_grad = False

    def forward(self, x):
        x = self.symfold(x)
        x = self.correlator(x)
        x = self.sympool(x)
        x = x.sum(dim=(-1, -2))
        return x


def preprocess_data(model, train_loader, val_loader, pred_loader, device='cuda', order_rescaling=None):
    """ Does a series of preprocessing steps:
         1) Passes all train data through the model to obtain the correlator features
         2) Normalizes all features to zero mean, unit variance as BatchNorm would ideally
         3) Returns tensors with all features concatenated together
    """
    if order_rescaling is None:
        order_rescaling = np.array([1.] * model.order)

    # Run the data through a few times to ensure batchnorm statistics are correct
    model.eval()
    predict = pred_loader is not None

    # To avoid computing the same convolutions over and over again, just compute all
    #  of them and make a new dataset from just these.
    train_predictors, train_labels = [], []
    val_predictors, val_labels = [], []
    pred_predictors = [] if predict else None
    for snapshot, label in train_loader:
        snapshot = snapshot.to(device=device)
        train_predictors.append(model(snapshot[:, 0]))
        train_labels.append(label)
    for snapshot, label in val_loader:
        snapshot = snapshot.to(device=device)
        val_predictors.append(model(snapshot[:, 0]))
        val_labels.append(label)

    train_predictors = torch.cat(train_predictors, dim=0).detach().cpu().numpy().astype(np.float64)
    train_labels = torch.cat(train_labels, dim=0).detach().cpu().numpy()
    val_predictors = torch.cat(val_predictors, dim=0).detach().cpu().numpy().astype(np.float64)
    val_labels = torch.cat(val_labels, dim=0).detach().cpu().numpy()

    # Normalize all predictors
    train_means = train_predictors.mean(axis=0)
    train_std = train_predictors.std(axis=0)
    val_means = val_predictors.mean(axis=0)
    val_std = val_predictors.std(axis=0)
    train_predictors = (train_predictors - train_means) / (train_std + 1e-7)
    val_predictors = (val_predictors - val_means) / (val_std + 1e-7)
    train_predictors = train_predictors.astype(np.float32)
    val_predictors = val_predictors.astype(np.float32)

    # We can effectively get a different loss coefficient on each order of feature
    #  by instead rescaling each order by a different coefficient
    order_rescaling = np.repeat(order_rescaling, model.num_filts)
    train_predictors /= order_rescaling
    val_predictors /= order_rescaling

    if predict:
        for snapshot, _ in pred_loader:
            snapshot = snapshot.to(device=device)
            pred_predictors.append(model(snapshot[:, 0]))
        pred_predictors = torch.cat(pred_predictors, dim=0).cpu().numpy().astype(np.float64)
        pred_means = pred_predictors.mean(axis=0)
        pred_std = pred_predictors.std(axis=0)
        pred_predictors = (pred_predictors - pred_means) / (pred_std + 1e-7)
        pred_predictors = pred_predictors.astype(np.float32)
        pred_predictors /= order_rescaling

    return train_predictors, train_labels, val_predictors, val_labels, pred_predictors


def save_path(filename, cs, coeffs, biases, train_accs, val_accs, exp_fracs):
    save_dict = {
        'cs': cs, 'coeffs': coeffs, 'biases': biases,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'exp_fracs': exp_fracs
    }
    with open(filename, 'wb') as f:
        pickle.dump(save_dict, f)


def run_lassopath(train_predictors, train_labels, val_predictors, val_labels, pred_predictors,
                  c_start, c_end, steps=20,
                  save=None, plot=False, elastic=None, val_precision=False):

    if elastic is not None:
        penalty = 'elasticnet'
        l1_ratio = elastic
    else:
        penalty = 'l1'
        l1_ratio = None

    logistic_clf = linear_model.LogisticRegression(
        penalty=penalty, l1_ratio=l1_ratio, solver='saga', warm_start=True, tol=1e-4,
        max_iter=int(1e8), fit_intercept=True,
    )

    cs = np.logspace(c_start, c_end, num=steps)
    coeffs = []
    biases = []
    val_accuracies = []
    train_accuracies = []
    predict = pred_predictors is not None
    pred_fractions = []
    for num, c in tqdm(enumerate(cs), mininterval=10, total=steps):
        logistic_clf.set_params(C=c)
        logistic_clf.fit(train_predictors, train_labels)
        coeffs.append(logistic_clf.coef_.ravel().copy())
        biases.append(logistic_clf.intercept_.copy())

        train_acc = val(logistic_clf, train_predictors, train_labels)
        train_accuracies.append(train_acc)
        val_acc = val(logistic_clf, val_predictors, val_labels, precision=val_precision)
        val_accuracies.append(val_acc)

        if predict:
            num_each_class = predict(logistic_clf, pred_predictors)
            pred_fractions.append(num_each_class[0] / np.sum(num_each_class))

    coeffs = np.array(coeffs)
    val_accuracies = np.array(val_accuracies)
    pred_fractions = np.array(pred_fractions)

    if save is not None:
        save_path(
            save + '.pkl', cs, coeffs, biases, train_accuracies, val_accuracies,
            pred_fractions
        )
        plt.savefig(save + '.png')

    # Plot path
    if plot:
        fig, (coeff_ax, acc_ax, pred_ax) = plot_l1_path(
            cs, coeffs, val_accuracies, pred_fractions
        )
        plt.show()


def val(classifier, inpts, labels, precision=False):
    preds = classifier.predict(inpts)
    conf_mat = confusion_matrix(labels, preds)
    if precision:
        # Avg Precision
        return (conf_mat[0, 0] / conf_mat[0, :].sum() + conf_mat[1, 1] / conf_mat[1, :].sum()) / 2
    else:
        # Standard accuracy
        return np.diag(conf_mat).sum() / conf_mat.sum()


def predict(logistic_clf, predictors):
    """ Using the given convs and logistic classifier, returns the number
         of inpts in pred_loader which are classified as [classA, classB]
    """
    num_each_class = np.zeros(2)
    preds = logistic_clf.predict(predictors)
    num_B = np.sum(preds)
    num_A = len(preds) - num_B
    num_each_class[0] = num_A
    num_each_class[1] = num_B
    return num_each_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Runs LARS to iteratively enable filters')
    parser.add_argument('model_file', type=str, help='Torch .pt file containing trained model')
    parser.add_argument('c_start', type=float, help='Minimum log(C) searched')
    parser.add_argument('c_end', type=float, help='Maximum log(C) searched')
    parser.add_argument('--symconv', action='store_true',
                        help='If set, uses SymConvLinear for paths rather than Correlator')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(REPODIR, '..', 'QGasData'),
                        help='Directory where data is located (Default: REPODIR/../QGasData)')
    parser.add_argument('--doping-level', type=str, default='9.0',
                        help='Doping level dataset to use')
    parser.add_argument('-a', '--dataset-a', type=str, default='AS',
                        help='Dataset A to train on')
    parser.add_argument('-b', '--dataset-b', type=str, default='Pi',
                        help='Dataset B to train on')
    parser.add_argument('-p', '--dataset-p', type=str, default=None,
                        help='Dataset to predict on')
    parser.add_argument('--group', type=int, default=1,
                        help='If set, collects snapshots into groups of the given size which are classified together.')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps in between c_start and c_end')
    parser.add_argument('--save', type=str, default=None,
                        help='If set, serializes regularization path and plots to files'
                             ' with the given prefix name')
    parser.add_argument('--elastic', type=float, default=None,
                        help='If set, does elastic net with the given l1_ratio.'
                             ' (0 corresponds to purely L2, 1 corresponds to purely L1)')
    parser.add_argument('--crop', type=int, default=None,
                        help='If set, does not crop snapshots when loaded to a square of the given size.')
    parser.add_argument('--circle-crop', action='store_true',
                        help='If set, crops snapshots to the circular area of experiment')
    parser.add_argument('--plot', action='store_true',
                        help='If set, plots the paths at the end')
    parser.add_argument('--order', type=int, default=4,
                        help='Sets the order to which models are constructed')
    parser.add_argument('--oversample', action='store_true',
                        help='Rough oversampling to attempt to even class distribution')
    parser.add_argument('--val-precision', action='store_true',
                        help='Use average precision as the "val accuracy" metric')
    parser.add_argument('--order-rescale', nargs='+', type=float, default=None)

    args = parser.parse_args()

    train_loader, val_loader, pred_loader = create_datasets(
        args.data_dir, args.dataset_a, args.dataset_b, args.doping_level,
        group=1, batch_size=256, dataset_pred=args.dataset_p, oversample=args.oversample,
        crop=args.crop, circle_crop=args.circle_crop,
    )

    if torch.cuda.is_available():
        print('Loaded CUDA successfully!')
        device = 'cuda'
    else:
        print('Could not load CUDA!')
        device = 'cpu'

    model_state_dict = torch.load(args.model_file, map_location=device)

    if args.symconv:
        convmodel = ChoppedFixedSymConv(model_state_dict['correlator.conv_filt']).to(device)
    else:
        convmodel = ChoppedCorrelatorExtractor(
            model_state_dict['correlator.conv_filt'],
            abs=True, order=args.order
        ).to(device)

    # Copy in the rest of the trained model parameters
    convmodel.correlator.bias.data = model_state_dict['correlator.bias']

    # Produce the output correlational predictors from the model
    order_rescaling = None if args.order_rescale is None else np.array(args.order_rescale)
    train_predictors, train_labels, val_predictors, val_labels, pred_predictors = preprocess_data(
        convmodel, train_loader, val_loader, pred_loader, device=device,
        order_rescaling=order_rescaling
    )

    # De-allocate loaders we don't need
    del train_loader
    del pred_loader

    run_lassopath(train_predictors, train_labels, val_predictors, val_labels, pred_predictors,
                  args.c_start, args.c_end,
                  steps=args.steps, save=args.save, plot=args.plot,
                  elastic=args.elastic, val_precision=args.val_precision)
