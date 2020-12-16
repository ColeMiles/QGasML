import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import data_util
import nn_models

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(SCRIPTPATH)
REPOPARDIR = os.path.dirname(REPODIR)


def print_and_log(message):
    print(message)
    logging.info(message)


def pretty_print_args(args):
    print_and_log("Running QGasML training with the following parameters:")
    for arg in vars(args):
        print_and_log("| {} {}".format(arg, getattr(args, arg) or 'False'))
    print_and_log("----------------------------------")


def create_datasets(data_dir, dataset_a, dataset_b, doping_level,
                    dataset_pred=None, group=1, batch_size=256, augment=False,
                    oversample=True, crop=None, circle_crop=False,):
    # Create paths to data files
    datadir_a = os.path.join(data_dir, 'Dataset{}'.format(dataset_a))
    datadir_b = os.path.join(data_dir, 'Dataset{}'.format(dataset_b))
    train_datafiles_a = [
        os.path.join(datadir_a, f) for f in os.listdir(datadir_a)
        if f.endswith('d{}.pkl'.format(doping_level))
    ]
    train_datafiles_b = [
        os.path.join(datadir_b, f) for f in os.listdir(datadir_b)
        if f.endswith('d{}.pkl'.format(doping_level))
    ]

    # Build datasets / loaders
    transform = data_util.RandomAugmentation() if augment else None
    train_loader, val_loader = data_util.build_datasets(
        train_datafiles_a + train_datafiles_b,
        [0] * len(train_datafiles_a) + [1] * len(train_datafiles_b),
        group_size=group, batch_size=batch_size, transform=transform,
        oversample=oversample, crop=crop, circle_crop=circle_crop,
    )

    # Repeat above for prediction dataset if asked for
    if dataset_pred is not None:
        datadir_pred = os.path.join(data_dir, 'Dataset{}'.format(dataset_pred))
        pred_datafiles = [
            os.path.join(datadir_pred, f) for f in os.listdir(datadir_pred)
            if f.endswith('d{}.pkl'.format(doping_level))
        ]
        _, pred_loader = data_util.build_datasets(
            pred_datafiles, [0] * len(pred_datafiles), group_size=1,
            batch_size=batch_size, circle_crop=circle_crop,
        )
    else:
        pred_loader = None

    return train_loader, val_loader, pred_loader


def main(args):
    if torch.cuda.is_available():
        print_and_log('Successfully loaded CUDA!')
        device = 'cuda:' + str(args.gpu)
    else:
        print('Could not load CUDA!')
        logging.warning('Could not load CUDA!')
        device = 'cpu'

    # Build datasets
    train_loader, val_loader, pred_loader = create_datasets(
        args.data_dir, args.dataset_a, args.dataset_b, args.doping_level,
        dataset_pred=args.dataset_pred, group=args.group, batch_size=args.batch_size,
        augment=args.augment, crop=args.crop, circle_crop=args.circle_crop,
    )

    # Set the seeds to control the batching/initialization
    if args.seed is not None:
        torch.random.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Peep at a snapshot to see what size it is
    input_shape = next(iter(train_loader))[0].shape
    in_channels = input_shape[-3]
    input_size = input_shape[-1]
    kwargs = {
        'in_channels': in_channels,
        'input_size': input_size,
    }
    # Temporary: Set model parameters which may not exist for all available models
    if args.model.find('Correlator') != -1:
        kwargs['order'] = args.order
        kwargs['absbeta'] = args.absbeta
    if args.filter_size != 3:
        kwargs['filter_size'] = args.filter_size

    # Build model
    if args.model.startswith('Fixed'):
        filters = np.load(args.fixed_filts)
        if len(filters.shape) < 4:
            filters = np.expand_dims(filters, 1)
        model = nn_models.models[args.model](filters, **kwargs).to(device)
    elif args.model in nn_models.models.keys():
        kwargs['num_filts'] = args.num_filts
        model = nn_models.models[args.model](**kwargs).to(device)
    else:
        raise ValueError('Specified model not in list of options:', nn_models.models.keys())

    if args.saved_model is not None:
        state_dict = torch.load(args.saved_model)
        model.load_state_dict(state_dict)

    # Main train/val loop
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.000005)
    criterion = nn.NLLLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.epochs):
        print_and_log('Epoch: {}'.format(epoch))

        # Train
        train_loss, train_conf_matrix = train(
            model, optimizer, criterion, train_loader,
            lambd1=args.l1_loss, lambd2=args.l2_loss,
            device=device
        )
        train_acc = train_conf_matrix.diag().sum() / train_conf_matrix.sum()
        print_and_log('\tTrain Loss: {}'.format(train_loss))
        print_and_log('\tTrain Accuracy: {}'.format(train_acc.item()))
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Val
        val_loss, val_conf_matrix = val(
            model, criterion, val_loader,
            lambd1=args.l1_loss, lambd2=args.l2_loss,
            device=device
        )
        val_acc = val_conf_matrix.diag().sum() / val_conf_matrix.sum()
        print_and_log('\tVal Loss: {}'.format(val_loss))
        print_and_log('\tVal Accuracy: {}'.format(val_acc.item()))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_scheduler.step()

    print('Final confusion matrix (val set):')
    print(val_conf_matrix)

    # Predict on experimental data
    if args.dataset_pred is not None:
        predict(model, pred_loader, device=device)

    # Save final model
    try:
        os.mkdir(args.save_root)
    except FileExistsError:
        pass
    model_path = os.path.join(
        args.save_root, 
        '{}_{}{}_d{}_{}seed{}.pt'.format(
            args.model,
            args.dataset_a,
            args.dataset_b,
            args.doping_level,
            args.tag,
            args.seed,
        )
    )
    torch.save(model.state_dict(), model_path)
    print_and_log('Saved model to ' + model_path)

    # TODO: Move details into plot_util
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        epochs = list(range(args.epochs))
        ax.plot(epochs, train_losses, color='xkcd:vermillion', label='Train Loss')
        ax.plot(epochs, val_losses, color='xkcd:golden yellow', label='Val Loss')
        ax.plot(epochs, train_accs, color='xkcd:ultramarine', label='Train Accuracy')
        ax.plot(epochs, val_accs, color='xkcd:apple green', label='Val Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0.0, 1.0)
        fig.legend()
        plt.show()

        plt.imshow(val_conf_matrix)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.xticks([0.5, 1.5], [args.dataset_a, args.dataset_b])
        plt.yticks([0.5, 1.5], [args.dataset_a, args.dataset_b])
        plt.text()


def train(model, optimizer, criterion, loader, lambd1=0.0, lambd2=0.0, device='cuda'):
    model.train()
    loss_total = torch.tensor(0.0, device=device)
    conf_matrix = torch.zeros(2, 2)
    for inpts, labels in loader:
        optimizer.zero_grad()
        inpts = inpts.to(device)
        labels = labels.to(device)

        # Fold the group dimension into the batch dimension
        batch_size = inpts.shape[0]
        group_size = inpts.shape[1]
        inpts = inpts.view(batch_size * group_size, *inpts.shape[2:])

        pred = model(inpts)

        # Unfold the group dimension
        pred = pred.view(batch_size, group_size, -1)
        # nn.NLLLoss expects log class scores
        pred = nn.functional.softmax(pred, dim=-1).mean(1).log()

        loss = criterion(pred, labels)
        l1_loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        parameters = model.named_parameters()
        parameters = filter(lambda namedparam: namedparam[1].requires_grad, parameters)
        for name, parameter in parameters:
            if name.startswith('conv') or name == 'correlator.conv_filt':
                l1_loss += parameter.norm(1)
            elif lambd2 != 0.0:
                l2_loss += parameter.norm(2)

        loss += lambd1 * l1_loss + lambd2 * l2_loss

        loss.backward()
        optimizer.step()

        loss_total += loss

        # Could be made (probably) faster with torch.scatter_add_
        pred_classes = pred.argmax(dim=1).cpu()
        labels = labels.cpu()
        for pred, label in zip(pred_classes, labels):
            conf_matrix[pred.item(), label.item()] += 1
    avg_loss = loss_total / len(loader)
    return avg_loss.item(), conf_matrix


def val(model, criterion, loader, lambd1=0.0, lambd2=0.0, device='cuda'):
    model.eval()
    loss_total = 0.0
    conf_matrix = torch.zeros(2, 2)

    with torch.no_grad():
        for inpts, labels in loader:
            inpts = inpts.to(device)
            labels = labels.to(device)

            # Fold the group dimension into the batch dimension
            batch_size = inpts.shape[0]
            group_size = inpts.shape[1]
            inpts = inpts.view(batch_size * group_size, *inpts.shape[2:])

            pred = model(inpts)

            # Unfold the group dimension, and average class scores
            pred = pred.view(batch_size, group_size, -1)
            # nn.CrossEntropyLoss expects class logits, not scores, so need to log after averaging
            pred = nn.functional.softmax(pred, dim=-1).mean(1).log()

            loss = criterion(pred, labels)

            l1_loss = torch.tensor(0.0, device=device)
            l2_loss = torch.tensor(0.0, device=device)
            parameters = model.named_parameters()
            parameters = filter(lambda namedparam: namedparam[1].requires_grad, parameters)

            for name, parameter in parameters:
                if name.startswith('conv') or name == 'correlator.conv_filt':
                    l1_loss += parameter.norm(1)
                elif lambd2 != 0.0:
                    l2_loss += parameter.norm(2)

            loss += lambd1 * l1_loss + lambd2 * l2_loss

            loss_total += loss.item()

            pred_classes = pred.argmax(dim=1).cpu()
            labels = labels.cpu()
            for pred, label in zip(pred_classes, labels):
                conf_matrix[pred.item(), label.item()] += 1
    avg_loss = loss_total / len(loader)
    return avg_loss, conf_matrix


def predict(model, loader, device='cuda'):
    model.eval()
    num_each_class = torch.zeros(2)
    with torch.no_grad():
        for inpts, _ in loader:
            inpts = inpts.to(device)

            # Fold the group dimension into the batch dimension
            batch_size = inpts.shape[0]
            group_size = inpts.shape[1]
            inpts = inpts.view(batch_size * group_size, *inpts.shape[2:])

            pred = model(inpts)

            # Unfold the group dimension, and average class scores
            pred = pred.view(batch_size, group_size, -1)
            pred = nn.functional.softmax(pred, dim=-1).mean(1)

            pred_classes = pred.argmax(dim=1)
            num_each_class[0] += pred_classes.eq(0).sum().item()
            num_each_class[1] += pred_classes.eq(1).sum().item()
        total = num_each_class.sum()
    print_and_log('Prediction set classification results:')
    print_and_log('A: {0} / {2}\tB: {1} / {2}'.format(num_each_class[0], num_each_class[1], total))
    return num_each_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (Default: 256)')
    parser.add_argument('-e', '--epochs', type=int, default=400,
                        help='Number of epochs to train (Default: 400)')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(REPOPARDIR, 'QGasData'),
                        help='Directory where data is located (Default: <REPODIR>/../QGasData)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Starting learning rate (Default: 0.005)')
    parser.add_argument('--doping-level', type=str, default='9.0',
                        help='Doping level dataset to use. (Default: 9.0)')
    parser.add_argument('-a', '--dataset-a', type=str, default='FullInfoAS',
                        help='Dataset A to train on. (Default: FullInfoAS)')
    parser.add_argument('-b', '--dataset-b', type=str, default='FullInfoPi',
                        help='Dataset B to train on. (Default: FullInfoPi)')
    parser.add_argument('-p', '--dataset-pred', type=str, default=None,
                        help='Dataset to be predict on')
    parser.add_argument('--save-root', type=str,
                        default=os.path.join(REPODIR, 'model'),
                        help='Directory to save model in')
    parser.add_argument('--log-root', type=str,
                        default=os.path.join(REPODIR, 'log'),
                        help='Directory to save training logs to')
    parser.add_argument('--model', type=str, default='CorrelatorExtractor',
                        help='Sets the model used. See bottom of nn_models.py for options.'
                             ' (Default: CorrelatorExtractor).')
    parser.add_argument('--num-filts', type=int, default=2,
                        help='Number of filters to use in the trained model.'
                             ' (Default: 2')
    parser.add_argument('--filter-size', type=int, default=3,
                        help='The spatial size of the learned filters.'
                             ' (Default: 3)')
    parser.add_argument('--order', type=int, default=4,
                        help='Sets the order of the correlators used in the model. '
                             'Only has an effect with Correlator architectures.'
                             ' (Default: 4)')
    parser.add_argument('--crop', type=int, default=None,
                        help='If set, crops snapshots to a square of the given size.')
    parser.add_argument('--circle-crop', action='store_true',
                        help='If set, crops snapshots to the circular area of experiment')
    parser.add_argument('--fixed-filts', type=str, default=None,
                        help='Only use along with fixed filter models. '
                             'Name of numpy file containing fixed filters to use.')
    parser.add_argument('--saved-model', type=str, default=None,
                        help='Load from saved model and continue training')
    parser.add_argument('--group', type=int, default=1,
                        help='If set, collects snapshots into groups of the given size which'
                             ' are classified together.')
    parser.add_argument('--seed', type=int, default=4444,
                        help='Sets the random seed controlling parameter initialization/batching.'
                             '(Default: 4444)')
    parser.add_argument('--augment', action='store_true',
                        help='If set, performs data augmentation on the training data')
    parser.add_argument('--plot', action='store_true',
                        help='If set, plots losses and accuracies at the end of training')
    parser.add_argument('--l1-loss', type=float, default=0.005,
                        help='Coefficient on L1 norm regularization loss for convolutional filters'
                             ' (Default: 0.005)')
    parser.add_argument('--l2-loss', type=float, default=0.0,
                        help='Coefficient on L2 norm regularization loss for fully-connected layers'
                             ' (Default: 0.0)')
    parser.add_argument('--absbeta', action='store_true', default=False,
                        help="If set, forces logistic beta coefficients to be positive")
    parser.add_argument('--tag', type=str, default='',
                        help='A tag to append to the log/model filenames')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use')
    args = parser.parse_args()

    try:
        os.mkdir(args.log_root)
    except FileExistsError:
        pass

    if args.fixed_filts is not None and not args.model.startswith('Fixed'):
        print_and_log(
            '--fixed-filts given but --model is not a type which uses fixed filters. Ignoring.'
        )
        args.fixed_filts = None
    elif args.fixed_filts is None and args.model in ['FixedSymConvLinear', 'FixedConvLinear']:
        print('Must supply --fixed-filts if using FixedConvLinear or FixedSymConvLinear!')
        sys.exit(1)

    if args.model not in nn_models.models.keys():
        print('--model {} not in list of available models'.format(args.model))
        sys.exit(1)

    logfile_name = '{}_{}{}_d{}_{}seed{}.txt'.format(
        args.model,
        args.dataset_a,
        args.dataset_b,
        args.doping_level,
        args.tag,
        args.seed
    )
    logging.basicConfig(
        filename=os.path.join(args.log_root, logfile_name),
        format='%(levelname)s | %(asctime)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level='INFO'
    )

    pretty_print_args(args)

    main(args)
