#!/usr/bin/python3
import argparse
import sys

import torch
import numpy as np


def extract_filts(model_files, out_filename):
    filts = []
    for filename in model_files:
        state_dict = torch.load(filename, map_location='cpu')
        # Hacky way to find the conv_filt
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('conv_filt') >= 0 or key.find('conv_stencil') >= 0:
                filts.append(state_dict[key].cpu().numpy())
                break
        else:
            print("Couldn't find a conv_filt")
            sys.exit(1)

    filts = np.concatenate(filts, axis=0)
    np.save(out_filename, filts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract filters from .pt model files')
    parser.add_argument('model_files', type=str, nargs='+', help='List of model .pt files to extract from')
    parser.add_argument('out_filename', type=str, help='Name of output (.npy) file')
    args = parser.parse_args()

    extract_filts(args.model_files, args.out_filename)
