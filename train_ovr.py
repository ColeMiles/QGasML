import os
import sys
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import data_util
import nn_models
import config_util
import train
from train import print_and_log

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(SCRIPTPATH)
REPOPARDIR = os.path.dirname(REPODIR)


def main(config):
    if torch.cuda.is_available():
        print_and_log('Successfully loaded CUDA!')
        device = 'cuda:' + str(config['Training']['gpu'])
    else:
        print('Could not load CUDA!')
        logging.warning('Could not load CUDA!')
        device = 'cpu'

    seed = config['Training']['seed']
    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    num_classes = config['Model Kwargs']['num_classes']
    config['Model Kwargs']['num_classes'] = 2

    ovr_models = []

    loader_gen = data_util.from_config_ovr(config)
    for nclass in range(num_classes):
        print_and_log('Training {} vs. Rest'.format(config['Data Files']['datasets'][nclass]))

        train_loader, val_loader = next(loader_gen)

        model = nn_models.from_config(config).to(device)

        start_lr = config['Training']['lr']
        epochs = config['Training']['epochs']
        optimizer = optim.Adam(model.parameters(), lr=start_lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.000005)
        criterion = nn.CrossEntropyLoss()
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        lambd1 = config['Training']['l1_loss']

        best_val_loss, best_model = float('inf'), model

        for epoch in range(epochs):
            print_and_log('Epoch: {}'.format(epoch))

            # Train
            train_loss, train_acc, train_ce = train.train(
                model, optimizer, criterion, train_loader,
                lambd1=lambd1, device=device
            )
            print_and_log('\tTrain Loss: {}'.format(train_ce))
            print_and_log('\tTrain Accuracy: {}'.format(train_acc))
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Val
            val_loss, val_acc, val_ce = train.val(
                model, criterion, val_loader,
                lambd1=lambd1, device=device
            )
            print_and_log('\tVal Loss: {}'.format(val_ce))
            print_and_log('\tVal Accuracy: {}'.format(val_acc))
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            lr_scheduler.step()

            if val_loss < best_val_loss:
                best_model = copy.deepcopy(model)

        num_classes = model.num_classes
        conf_mat = train.confusion_matrix(model, val_loader, num_classes, device=device)
        print_and_log('Final eval confusion matrix:')
        print_and_log(conf_mat)

        ovr_models.append(best_model)

    final_model = nn_models.OVRModel(ovr_models)

    # Save model
    save_dir = config['Logging']['save_root']
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    model_path = os.path.join(
        save_dir,
        '{}_{}_{}seed{}.pt'.format(
            'OVRModel',
            "".join(config['Data Files']['datasets']),
            config['Logging']['tag'],
            seed
        )
    )
    torch.save(final_model.state_dict(), model_path)
    print_and_log('Saved model to ' + model_path)


if __name__ == "__main__":
    parser = config_util.make_parser()
    args = parser.parse_args()
    config = config_util.parse_config(args.config)
    config_util.update_config(config, args)

    config['Logging']['githash'] = train.get_git_hash()

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
        config['Training']['seed']
    )
    logging.basicConfig(
        filename=os.path.join(config['Logging']['log_root'], logfile_name),
        format='%(levelname)s | %(asctime)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w',
        level='INFO'
    )

    train.pretty_print_config(config)

    main(config)
