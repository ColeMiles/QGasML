#!/usr/bin/python3
import argparse

import torch
import numpy as np


def extract_filts(model_files, out_filename):
    filts = []
    for filename in model_files:
        state_dict = torch.load(filename, map_location='cpu')
        # Hacky way to detect CorrelatorExtractor log files
        if "CorrelatorExtractor" in filename:
            filts.append(state_dict['correlator.conv_filt'].cpu().numpy())
        else:
            filts.append(state_dict['conv1.weight'].cpu().numpy())
    filts = np.concatenate(filts, axis=0)
    np.save(out_filename, filts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract filters from .pt model files')
    parser.add_argument('model_files', type=str, nargs='+', help='List of model .pt files to extract from')
    parser.add_argument('out_filename', type=str, help='Name of output (.npy) file')
    args = parser.parse_args()

    extract_filts(args.model_files, args.out_filename)
