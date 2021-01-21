import argparse
import os
import matplotlib.pyplot as plt
import pickle

import torch
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import nn_models
import data_util
from plot_util import plot_l1_path
from config_util import parse_config

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(SCRIPTPATH)


def preprocess_data(model, train_loader, val_loader, device='cuda', order_rescaling=None):
    """ Does a series of preprocessing steps:
         1) Passes all train data through the model to obtain the correlator features
         2) Normalizes all features to zero mean, unit variance as BatchNorm would ideally
         3) Returns tensors with all features concatenated together
    """
    if order_rescaling is None:
        order_rescaling = np.array([1.] * model.order)

    # Run the data through a few times to ensure batchnorm statistics are correct
    model.eval()

    # To avoid computing the same convolutions over and over again, just compute all
    #  of them and make a new dataset from just these.
    train_predictors, train_labels = [], []
    val_predictors, val_labels = [], []
    for snapshot, label in train_loader:
        snapshot = snapshot.to(device=device)
        train_predictors.append(
            model.chop_pred(snapshot).detach().cpu().numpy().astype(np.float64)
        )
        train_labels.append(label)
    for snapshot, label in val_loader:
        snapshot = snapshot.to(device=device)
        val_predictors.append(
            model.chop_pred(snapshot).detach().cpu().numpy().astype(np.float64)
        )
        val_labels.append(label)

    train_predictors = np.concatenate(train_predictors, axis=0)
    train_labels = torch.cat(train_labels, dim=0).detach().cpu().numpy()
    val_predictors = np.concatenate(val_predictors, axis=0)
    val_labels = torch.cat(val_labels, dim=0).detach().cpu().numpy()

    # Normalize all predictors
    train_means = train_predictors.mean(axis=0)
    train_std = train_predictors.std(axis=0)
    train_predictors = (train_predictors - train_means) / (train_std + 1e-7)
    val_predictors = (val_predictors - train_means) / (train_std + 1e-7)
    train_predictors = train_predictors.astype(np.float32)
    val_predictors = val_predictors.astype(np.float32)

    # We can effectively get a different loss coefficient on each order of feature
    #  by instead rescaling each order by a different coefficient
    order_rescaling = np.repeat(order_rescaling, model.num_filts)
    train_predictors /= order_rescaling
    val_predictors /= order_rescaling

    return train_predictors, train_labels, val_predictors, val_labels


def save_path(filename, cs, coeffs, biases, train_accs, val_accs):
    save_dict = {
        'cs': cs, 'coeffs': coeffs, 'biases': biases,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }
    with open(filename, 'wb') as f:
        pickle.dump(save_dict, f)


def run_lassopath(train_predictors, train_labels, val_predictors, val_labels,
                  c_start, c_end, steps=20,
                  save=None, plot=False, elastic=None, val_precision=False):

    # Save those shapes off, then reshape
    feat_shape = train_predictors.shape[1:]
    num_classes = np.max(train_labels) + 1
    train_predictors = train_predictors.reshape((train_predictors.shape[0], -1))
    val_predictors = val_predictors.reshape((val_predictors.shape[0], -1))

    if elastic is not None:
        penalty = 'elasticnet'
        l1_ratio = elastic
    else:
        penalty = 'l1'
        l1_ratio = None

    logistic_clf = linear_model.LogisticRegression(
        penalty=penalty, l1_ratio=l1_ratio, solver='saga', warm_start=True, tol=1e-4,
        max_iter=int(1e5), fit_intercept=True, multi_class='ovr',
        class_weight='balanced'
    )

    cs = np.logspace(c_start, c_end, num=steps)
    coeffs = []
    biases = []
    val_accuracies = []
    train_accuracies = []
    for num, c in tqdm(enumerate(cs), mininterval=10, total=steps):
        logistic_clf.set_params(C=c)
        logistic_clf.fit(train_predictors, train_labels)
        coeffs.append(logistic_clf.coef_.copy())
        biases.append(logistic_clf.intercept_.copy())

        train_acc = val(logistic_clf, train_predictors, train_labels, precision=val_precision)
        train_accuracies.append(train_acc)
        val_acc = val(logistic_clf, val_predictors, val_labels, precision=val_precision)
        val_accuracies.append(val_acc)

    coeffs = np.array(coeffs)
    # Reshape coeffs to re-instate info about (NCS, NCLASS, NORDER, NFILTS)
    # NCLASS = 2 is a special case, there's just one set of coefficients
    if num_classes == 2:
        coeffs = coeffs.reshape((-1, 1, *feat_shape))
    else:
        coeffs = coeffs.reshape((-1, num_classes, *feat_shape))
    # Transpose to (NCLASS, NORDER, NFILTS, NCS)
    coeffs = np.transpose(coeffs, (1, 2, 3, 0))
    biases = np.array(biases)
    val_accuracies = np.array(val_accuracies)



    if save is not None:
        save_path(
            save + '.pkl', cs, coeffs, biases, train_accuracies, val_accuracies,
        )
        plt.savefig(save + '.png')

    # Plot path
    if plot:
        fig, (coeff_axes, acc_ax) = plot_l1_path(
            cs, coeffs, val_accuracies
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
    parser = argparse.ArgumentParser('Creates regularization paths from trained models')
    parser.add_argument('model_file', type=str, help='Torch .pt file containing trained model')
    parser.add_argument('config', type=str, help='Path to a .ini config file used to train model')
    parser.add_argument('c_start', type=float, help='Minimum log(C) searched')
    parser.add_argument('c_end', type=float, help='Maximum log(C) searched')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps in between c_start and c_end')
    parser.add_argument('--save', type=str, default=None,
                        help='If set, serializes regularization path and plots to files'
                             ' with the given prefix name')
    parser.add_argument('--elastic', type=float, default=None,
                        help='If set, does elastic net with the given l1_ratio.'
                             ' (0 corresponds to purely L2, 1 corresponds to purely L1)')
    parser.add_argument('--plot', action='store_true',
                        help='If set, plots the paths at the end')
    parser.add_argument('--val-precision', action='store_true',
                        help='Use average precision as the "val accuracy" metric')
    parser.add_argument('--order-rescale', nargs='+', type=float, default=None)
    parser.add_argument('--doping-level', type=float, default=None)

    args = parser.parse_args()
    config = parse_config(args.config)
    config['Model']['saved_model'] = args.model_file
    config['Preprocessing']['oversample'] = False

    if args.doping_level is not None:
        config['Loader Kwargs']['doping_level'] = args.doping_level

    if torch.cuda.is_available():
        print('Loaded CUDA successfully!')
        device = 'cuda'
    else:
        print('Could not load CUDA!')
        device = 'cpu'

    model = nn_models.from_config(config).to(device=device)

    train_loader, val_loader = data_util.from_config(config)

    # Produce the output correlational predictors from the model
    order_rescaling = None if args.order_rescale is None else np.array(args.order_rescale)
    train_predictors, train_labels, val_predictors, val_labels = preprocess_data(
        model, train_loader, val_loader, device=device,
        order_rescaling=order_rescaling
    )

    # Reshape predictors so that the shape is [NSAMP, NORDER, NFILTS]
    train_predictors = train_predictors.reshape((-1, model.order, model.num_filts))
    val_predictors = val_predictors.reshape((-1, model.order, model.num_filts))

    # TEMP: Manually zero out order-1 features
    train_predictors[:, 0] = 0.0
    val_predictors[:, 0] = 0.0

    # De-allocate loaders we don't need
    del train_loader
    del val_loader

    run_lassopath(train_predictors, train_labels, val_predictors, val_labels,
                  args.c_start, args.c_end,
                  steps=args.steps, save=args.save, plot=args.plot,
                  elastic=args.elastic, val_precision=args.val_precision)
