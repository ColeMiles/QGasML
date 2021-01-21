import os
import sys
import logging
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import data_util
import nn_models
import config_util

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(SCRIPTPATH)
REPOPARDIR = os.path.dirname(REPODIR)


def get_git_hash():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPODIR
    ).strip()


def print_and_log(message):
    print(message)
    logging.info(message)


def pretty_print_config(config):
    print_and_log("Running QGasML training with the following parameters:")
    for _, sect_dict in config.items():
        for k, v in sect_dict.items():
            print_and_log("| {} {}".format(k, v if v is not None else 'False'))
    print_and_log("----------------------------------")


def main(config):
    if torch.cuda.is_available():
        print_and_log('Successfully loaded CUDA!')
        device = 'cuda:' + str(config['Training']['gpu'])
    else:
        print('Could not load CUDA!')
        logging.warning('Could not load CUDA!')
        device = 'cpu'

    # Build datasets
    train_loader, val_loader = data_util.from_config(config)

    # Set the seeds to control the batching/initialization
    seed = config['Training']['seed']
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    # Build model
    model = nn_models.from_config(config).to(device)

    # If provided, load previously saved model parameters.
    if config['Model']['saved_model'] is not None:
        state_dict = torch.load(config['Model']['saved_model'])
        model.load_state_dict(state_dict)

    start_lr = config['Training']['lr']
    epochs = config['Training']['epochs']
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.000005)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lambd1 = config['Training']['l1_loss']

    # Main train/val loop
    for epoch in range(epochs):
        print_and_log('Epoch: {}'.format(epoch))

        # Train
        train_loss, train_acc, train_ce = train(
            model, optimizer, criterion, train_loader,
            lambd1=lambd1, device=device
        )
        print_and_log('\tTrain Loss: {}'.format(train_ce))
        print_and_log('\tTrain Accuracy: {}'.format(train_acc))
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Val
        val_loss, val_acc, val_ce = val(
            model, criterion, val_loader,
            lambd1=lambd1, device=device
        )
        print_and_log('\tVal Loss: {}'.format(val_ce))
        print_and_log('\tVal Accuracy: {}'.format(val_acc))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        lr_scheduler.step()

    num_classes = model.num_classes
    conf_mat = confusion_matrix(model, val_loader, num_classes, device=device)
    print_and_log('Final eval confusion matrix:')
    print_and_log(conf_mat)

    # Save final model
    save_dir = config['Logging']['save_root']
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    model_type = config['Model']['model']
    model_path = os.path.join(
        save_dir,
        '{}_{}_{}seed{}.pt'.format(
            model_type,
            "".join(config['Data Files']['datasets']),
            config['Logging']['tag'],
            seed
        )
    )
    torch.save(model.state_dict(), model_path)
    print_and_log('Saved model to ' + model_path)

def train(model, optimizer, criterion, loader, lambd1=0.0, device='cuda'):
    model.train()
    ce_total = torch.tensor(0.0, device=device)
    loss_total = torch.tensor(0.0, device=device)
    acc_total = torch.tensor(0.0, device=device)
    num_snaps = 0
    for inpts, labels in loader:
        optimizer.zero_grad()
        inpts = inpts.to(device)
        labels = labels.to(device)

        pred = model(inpts)
        loss = criterion(pred, labels)

        ce_total += loss

        l1_loss = torch.tensor(0.0, device=device)
        parameters = model.named_parameters()
        parameters = filter(lambda namedparam: namedparam[1].requires_grad, parameters)
        for name, parameter in parameters:
            if name.startswith('conv') or name == 'correlator.conv_filt' or name == 'correlator.filters':
                l1_loss += parameter.norm(1)

        loss += lambd1 * l1_loss

        loss.backward()
        optimizer.step()

        loss_total += loss

        pred_classes = pred.argmax(dim=1).cpu()
        labels = labels.cpu()
        acc_total += torch.count_nonzero(torch.eq(pred_classes, labels))
        num_snaps += len(labels)
    avg_loss = loss_total / num_snaps
    avg_acc = acc_total / num_snaps
    avg_ce = ce_total / num_snaps
    return avg_loss.item(), avg_acc.item(), avg_ce.item()


def val(model, criterion, loader, lambd1=0.0, device='cuda'):
    model.eval()
    ce_total = 0.0
    loss_total = 0.0
    acc_total = 0.0
    num_snaps = 0

    with torch.no_grad():
        for inpts, labels in loader:
            inpts = inpts.to(device)
            labels = labels.to(device)

            pred = model(inpts)
            loss = criterion(pred, labels)
            ce_total += loss

            l1_loss = torch.tensor(0.0, device=device)
            parameters = model.named_parameters()
            parameters = filter(lambda namedparam: namedparam[1].requires_grad, parameters)

            for name, parameter in parameters:
                if name.startswith('conv') or name == 'correlator.conv_filt' or name == 'linear_fourier.weight':
                    l1_loss += parameter.norm(1)\

            loss += lambd1 * l1_loss

            loss_total += loss.item()

            pred_classes = pred.argmax(dim=1).cpu()
            labels = labels.cpu()
            acc_total += torch.count_nonzero(torch.eq(pred_classes, labels)).item()
            num_snaps += len(labels)
    avg_loss = loss_total / num_snaps
    avg_acc = acc_total / num_snaps
    avg_ce = ce_total / num_snaps
    return avg_loss, avg_acc, avg_ce


def confusion_matrix(model, loader, num_classes, device='cuda'):
    model.eval()

    conf_mat = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for inpts, labels in loader:
            inpts = inpts.to(device)
            labels = labels.to(device)

            pred = model(inpts)
            scores = nn.functional.softmax(pred, dim=-1)

            pred_classes = scores.argmax(dim=1).cpu()
            labels = labels.cpu()
            for pred, label in zip(pred_classes, labels):
                conf_mat[pred, label] += 1

    return conf_mat


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
    parser = config_util.make_parser()
    args = parser.parse_args()
    config = config_util.parse_config(args.config)
    config_util.update_config(config, args)

    # Insert the git hash into the config so that it gets logged
    config['Logging']['githash'] = get_git_hash()

    try:
        os.mkdir(config['Logging']['log_root'])
    except FileExistsError:
        pass

    model_config = config['Model']
    if model_config['model'] not in nn_models.models.keys():
        print('--model {} not in list of available models'.format(args.model))
        sys.exit(1)

    logfile_name = '{}_{}_{}seed{}.txt'.format(
        config['Model']['model'],
        "".join(config['Data Files']['datasets']),
        args.tag,
        args.seed
    )
    logging.basicConfig(
        filename=os.path.join(config['Logging']['log_root'], logfile_name),
        format='%(levelname)s | %(asctime)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level='INFO'
    )

    pretty_print_config(config)

    main(config)
