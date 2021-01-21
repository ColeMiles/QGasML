import argparse
import configparser
import os
from typing import List, Dict, Any

FILEPATH = os.path.abspath(__file__)
REPODIR = os.path.dirname(os.path.dirname(FILEPATH))


def parse_str_list(lines) -> List[str]:
    return list(line for line in lines.splitlines() if len(line) > 0)


# Tempate that the config should match, providing types for each option
TEMPLATE = {
    'Training': {
        'batch_size': int,
        'epochs': int,
        'lr': float,
        'seed': int,
        'l1_loss': float,
        'gpu': int,
    },
    'Model': {
        'model': str,
        'fixed_filts': str,
        'saved_model': str,
    },
    'Model Kwargs': dict,
    'Data Files': {
        'base_dir': str,
        'datasets': parse_str_list,
        'loader_func': str,
    },
    'Loader Kwargs': dict,
    'Preprocessing': {
        'augment': bool,
        'oversample': bool,
    },
    'Logging': {
        'save_root': str,
        'log_root': str,
        'tag': str,
    }
}

DEFAULTS = {
    'Training': {
        'batch_size': 256,
        'epochs': 400,
        'lr': 0.005,
        'seed': 1111,
        'l1_loss': 0.005,
        'gpu': 0,
    },
    'Model': {
        'model': 'CCNN',
        'fixed_filts': None,
        'saved_model': None,
    },
    'Model Kwargs': {
        'in_channels': 1,
        'num_filts': 2,
        'filter_size': 3,
        'order': 4,
        'num_classes': 2,
        'absbeta': False,
    },
    'Data Files': {
        'base_dir': os.path.join(os.path.dirname(REPODIR), 'QGasData'),
        'datasets': ['FullInfoAS', 'FullInfoPi'],
        'loader_func': 'load_qgm_data',
    },
    'Loader Kwargs': {
        'doping_level': 9.0,
        'crop': None,
        'circle_crop': False,
    },
    'Preprocessing': {
        'augment': False,
        'oversample': False,
    },
    'Logging': {
        'save_root': os.path.join(REPODIR, 'model'),
        'log_root': os.path.join(REPODIR, 'log'),
        'tag': '',
    }
}


def eval_parse_dict(items) -> Dict:
    parsed_dict = dict()
    for key, val in items:
        parsed_dict[key] = eval(val)
    return parsed_dict


# noinspection PyTypeChecker
def parse_config(filename: str) -> Dict:
    config = configparser.ConfigParser()
    with open(filename, 'r') as f:
        config.read_file(f)

    parsed_dict = {}
    for section, options_dict in TEMPLATE.items():
        config_sect = config[section]

        # Special case, parse dict using Python evaluator
        # Note: `options_dict is dict` is checking if `options_dict` is literally the
        #   *function* dict, not checking if it is a dictionary.
        if options_dict is dict:
            parsed_dict[section] = eval_parse_dict(config.items(section))
            continue

        parsed_dict[section] = dict()
        parsed_section = parsed_dict[section]
        default_section = DEFAULTS[section]
        for option, converter in options_dict.items():
            if converter is int:
                parsed_section[option] = config_sect.getint(
                    option, fallback=default_section[option]
                )
            elif converter is float:
                parsed_section[option] = config_sect.getfloat(
                    option, fallback=default_section[option]
                )
            elif converter is bool:
                parsed_section[option] = config_sect.getboolean(
                    option, fallback=default_section[option]
                )
            elif converter is str:
                parsed_section[option] = config_sect.get(
                    option, fallback=default_section[option]
                )
            else:
                option_str = config_sect.get(option)
                if option_str is None:
                    parsed_section[option] = default_section[option]
                else:
                    parsed_section[option] = converter(option_str)

    return parsed_dict


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Config file for training.'
                             ' Settings of the config are overridden by command line arguments.')
    parser.add_argument('--batch-size', type=int,
                        help='Training batch size (Default: 256)')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Number of epochs to train (Default: 400)')
    parser.add_argument('--data-dir', type=str,
                        help='Directory where data is located (Default: <REPODIR>/../QGasData)')
    parser.add_argument('--lr', type=float,
                        help='Starting learning rate (Default: 0.005)')
    parser.add_argument('--save-root', type=str,
                        help='Directory to save model in')
    parser.add_argument('--log-root', type=str,
                        help='Directory to save training logs to')
    parser.add_argument('--model', type=str,
                        help='Sets the model used. See bottom of nn_models.py for options.'
                             ' (Default: CCNN).')
    parser.add_argument('--num-filts', type=int,
                        help='Number of filters to use in the trained model.'
                             ' (Default: 2')
    parser.add_argument('--filter-size', type=int,
                        help='The spatial size of the learned filters.'
                             ' (Default: 3)')
    parser.add_argument('--order', type=int,
                        help='Sets the order of the correlators used in the model. '
                             'Only has an effect with Correlator architectures.'
                             ' (Default: 4)')
    parser.add_argument('--crop', type=int,
                        help='If set, crops snapshots to a square of the given size.')
    parser.add_argument('--circle-crop', action='store_true', default=None,
                        help='If set, crops snapshots to the circular area of the Fermi-Hubbard'
                             ' experiment')
    parser.add_argument('--fixed-filts', type=str,
                        help='Only use along with fixed filter models. '
                             'Name of numpy file containing fixed filters to use.')
    parser.add_argument('--saved-model', type=str,
                        help='Load from saved model and continue training')
    parser.add_argument('--group', type=int,
                        help='If set, collects snapshots into groups of the given size which'
                             ' are classified together.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Sets the random seed controlling parameter initialization/batching.')
    parser.add_argument('--augment', action='store_true',
                        help='If set, performs data augmentation on the training data')
    parser.add_argument('--l1-loss', type=float,
                        help='Coefficient on L1 norm regularization loss for convolutional filters'
                             ' (Default: 0.005)')
    parser.add_argument('--absbeta', action='store_true',
                        help="If set, forces logistic beta coefficients to be positive")
    parser.add_argument('--reach', type=int,
                        help="Keyword argument for kagome models, defining filter sizes.")
    parser.add_argument('--tag', type=str,
                        help='A tag to append to the log/model filenames')
    parser.add_argument('--gpu', type=int, help='Index of GPU to use')
    parser.add_argument('--doping-level', type=float, default=None,
                        help='For Fermi-Hubbard data, set the doping level to load.')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold for 10-fold cross validation (Rydberg data only).')
    return parser


# Updates a config dict with options provided through the command line
def update_config(config: Dict[str, Dict], args: argparse.Namespace):
    for _, sect_dict in config.items():
        for key in sect_dict.keys():
            if key in args:
                arg_val = getattr(args, key)
                if arg_val is not None:
                    sect_dict[key] = arg_val
