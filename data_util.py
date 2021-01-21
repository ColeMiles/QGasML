""" Utility functions and classes for loading / processing / augmenting data
"""
import pickle
import itertools
from typing import List, Tuple, Dict, Generator
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def _center_crop(arr, size):
    """ Like torchvision.transforms.CenterCrop, but on numpy arrays
    """
    w, h = arr.shape[-2:]
    if w < size or h < size:
        raise ValueError("Array to crop is smaller than crop region.")

    margin_w, margin_h = w - size, h - size
    left, right = margin_w // 2, int(np.ceil(margin_w / 2))
    bot, top = margin_h // 2, int(np.ceil(margin_h / 2))
    return arr[..., left:w-right, bot:h-top]


def _border_crop(arr, size):
    """ Zeros out all pixels in the image, except for a boundary layer
         of the given size.
    """
    w, h = arr.shape[-2:]
    if w < 2 * size or h < 2 * size:
        raise ValueError("Array to boundary crop is smaller than the asked size.")

    arr[..., size:-size, size:-size] = 0
    return arr


def _circle_crop(arr):
    """ Crops out the square containing the circle of the microscope,
          zeroing out the area surrounding the circle.
    """
    cropped_arr = _center_crop(arr, 10)
    # Pixels at the masked locations should be zero'd out
    zero_mask = np.array([
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    ]).astype(np.bool)
    cropped_arr[:, :, zero_mask] = 0
    return cropped_arr


def _vflip(arr):
    return torch.flip(arr, [-2])


def _hflip(arr):
    return torch.flip(arr, [-1])


def _rot(arr, k):
    return torch.rot90(arr, k, [-2, -1])

_sym_ops = [
    lambda x: x,
    lambda x: np.flip(x, axis=-1),
    lambda x: np.flip(x, axis=-2),
    lambda x: np.rot90(x, axes=(-2, -1)),
    lambda x: np.rot90(x, 2, axes=(-2, -1)),
    lambda x: np.rot90(x, 3, axes=(-2, -1)),
    lambda x: np.rot90(np.flip(x, axis=-1), axes=(-2, -1)),
    lambda x: np.rot90(np.flip(x, axis=-2), axes=(-2, -1))
]

_inv_ops = [
    lambda x: x,
    lambda x: np.flip(x, axis=-1),
    lambda x: np.flip(x, axis=-2),
    lambda x: np.rot90(x, -1, axes=(-2, -1)),
    lambda x: np.rot90(x, -2, axes=(-2, -1)),
    lambda x: np.rot90(x, -3, axes=(-2, -1)),
    lambda x: np.flip(np.rot90(x, -1, axes=(-2, -1)), axis=-1),
    lambda x: np.flip(np.rot90(x, -1, axes=(-2, -1)), axis=-2)
]

def _staggered_mag(snapshot, parity=0):
    """ Calculates the staggered magnetization of a snapshot, defined as
        the sum
            Σ_{ij} (-1)^{i + j} s_{ij}
    """
    width, height = snapshot.shape[-2:]
    idxs = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    checkerboard = (-1) ** (np.stack(idxs, axis=0).sum(axis=0) + parity)
    return np.sum(snapshot * checkerboard, axis=(-1, -2))


def _checkerboard(shape, parity=0):
    """ Returns a Neel state of the desired shape, with the given parity
    """
    checker = np.empty(shape)
    for idx in itertools.product(*[np.arange(size) for size in shape]):
        checker[idx] = (-1) ** (np.sum(idx) + parity)
    return checker


def _repeat_iter(iterable):
    """ Makes an infinite iterator that repeatedly chains together iterators
         created from the given iterable.
    """
    while True:
        for item in iter(iterable):
            yield item


class RandomFlip:
    """ Randomly either horizontally or vertically flips each snapshot
    """
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def __call__(self, arr):
        flip = self.rng.randint(0, 4)
        if flip == 1:
            return _vflip(arr)
        elif flip == 2:
            return _hflip(arr)
        elif flip == 3:
            return _hflip(_vflip(arr))
        else:
            return arr


class RandomRotation:
    """ Randomly rotates the snapshot
    """
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def __call__(self, arr):
        k = self.rng.randint(0, 4)
        return _rot(arr, k)


class RandomAugmentation:
    def __init__(self, seed=None):
        self.randflip = RandomFlip(seed)
        self.randrot = RandomRotation(seed)

    def __call__(self, arr):
        return self.randflip(self.randrot(arr))


class BalancedSampler(torch.utils.data.Sampler):
    """ A sampler which oversamples all classes to provide an even class distribution
    """
    def __init__(self, labels):
        self.num_classes = torch.max(labels).item() + 1
        self.labels = labels
        self.class_indices = [
            np.where(labels == k)[0] for k in range(self.num_classes)
        ]
        self.class_counts = torch.tensor([len(indices) for indices in self.class_indices])
        self.num_sample_per_class = self.class_counts.max().item()

    def __iter__(self):
        epoch_sample_indices = []

        # At base, we include all of the samples of each class
        # Then, randomly sample extra indices from each class to make up to the target number
        for i in range(self.num_classes):
            indices = self.class_indices[i]
            num_oversample = self.num_sample_per_class - len(indices)
            epoch_sample_indices.append(indices)

            if num_oversample > 0:
                num_copies_needed = int(np.floor(num_oversample / len(indices)))
                for _ in range(num_copies_needed):
                    epoch_sample_indices.append(indices)
                num_extra_needed = num_oversample - num_copies_needed * len(indices)

                # You get a scalar if you index with a length-1 tensor, so need separate
                #  logic for this case
                if num_extra_needed > 1:
                    oversample_indices = torch.randperm(len(indices))[:num_extra_needed]
                    epoch_sample_indices.append(indices[oversample_indices])
                elif num_extra_needed == 1:
                    oversample_index = torch.randint(len(indices), (1,))
                    epoch_sample_indices.append(torch.tensor([indices[oversample_index]]))

        epoch_sample_indices = np.concatenate(epoch_sample_indices)

        # Now, reshuffle this list again
        epoch_sample_indices = epoch_sample_indices[torch.randperm(len(epoch_sample_indices))]

        # Now, just iterate through for the epoch
        for ind in epoch_sample_indices:
            yield ind

    def __len__(self):
        return self.num_sample_per_class * self.num_classes


class TransformedDataset(torch.utils.data.Dataset):
    """ Acts like TensorDataset, but with a transform on the first tensor.
    """
    def __init__(self, *tensors, transform=None):
        super().__init__()
        if not all(t.shape[0] == tensors[0].shape[0] for t in tensors):
            raise ValueError('Not all tensors have the same number length in axis 0')
        self.tensors = tensors
        self.transform = transform

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform is not None:
            transformed_tensor = self.transform(self.tensors[0][idx])
            return (transformed_tensor,) + tuple(tensor[idx] for tensor in self.tensors[1:])
        return tuple(tensor[idx] for tensor in self.tensors)


def build_data_loader(tensors: List[torch.Tensor], batch_size=1, transform=None,
                      oversample=True, shuffle=False, chop_even=False,
                      **kwargs) -> torch.utils.data.DataLoader:
    """ Builds train/eval sets from pickled data files.
          tensors:      List of snapshot tensors of shape [NSNAPS, NCHAN, WIDTH, HEIGHT]. Each
                            tensor is assumed to be a different class.
          batch_size:   The batch size of the loaders
          transform:    An optional transformation to apply to training snapshots as they're sampled
          oversample:   If True, will oversample classes in the training set to provide an even distribution.
                         If False, will just completely drop snapshots to make the distribution even.
          crop:         If not None, crops to a square of the given size
          circle_crop:  If True, crops snapshots to the circular region obtained from experiment
          shuffle:      If True, batches of snapshots are shuffled each epoch
        The returned loaders will produce tensors of shape
            [batch_size, group_size, num_channels, width, height]
         along with label vectors of length [batch_size]
    """
    num_classes = len(tensors)

    # If we're not oversampling, drop snapshots to make the train distribution even
    if not oversample and chop_even:
        min_size = min(map(len, tensors))
        for i in range(num_classes):
            tensors[i] = tensors[i][:min_size]

    # Produce a tensor of labels matching the lengths of the lists of tensors
    labels = torch.cat(tuple(
        torch.full((len(snaps),), label, dtype=torch.int64)
        for label, snaps in enumerate(tensors)
    ), dim=0)

    tensors = torch.cat(tensors)
    dataset = TransformedDataset(tensors, labels, transform=transform)

    if oversample:
        sampler = BalancedSampler(labels) if oversample else None
        shuffle = False
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, **kwargs)
    return loader


## Functions below here are specialized data loaders for different datasets ##


def load_qgm_data(datadir: str, doping_level: float, crop=None,
                  circle_crop=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Specialized data-loader function for the data format used in the
         original CCNN paper. 
    """
    datafiles = [
        os.path.join(datadir, f) for f in os.listdir(datadir)
        if f.endswith('d{}.pkl'.format(doping_level))
    ]
    train_snaps = []
    val_snaps = []

    for pkl_file in datafiles:
        with open(pkl_file, 'rb') as f:
            snap_dict = pickle.load(f)
            snaps = np.stack(snap_dict['snapshots'], axis=0).astype(np.float64)
            if crop is not None:
                snaps = _center_crop(snaps, crop)
            elif circle_crop:
                snaps = _circle_crop(snaps)

            train_idxs = snap_dict['train_idxs']
            val_idxs = snap_dict['val_idxs']
            train_snaps.append(snaps[train_idxs])
            val_snaps.append(snaps[val_idxs])

    train_snaps = np.concatenate(train_snaps, axis=0)
    val_snaps = np.concatenate(val_snaps, axis=0)
    train_snaps = torch.tensor(train_snaps, dtype=torch.float32)
    val_snaps = torch.tensor(val_snaps, dtype=torch.float32)
    return train_snaps, val_snaps


def load_rydberg_data(datadir: str, postselect: bool,
                      crop=None, border_crop=None, fold=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Specialized data-loader function for the data format used in the
         Rydberg phase-discovery paper. This function supports the selection
         of one of ten folds using the `fold` argument.
    """
    # Find the datafiles
    datafiles = [
        os.path.join(datadir, f) for f in os.listdir(datadir)
        if f.count('rydberg') > 0
    ]

    if fold < 1 or fold > 10:
        raise ValueError("Valid values for `fold` are [1, 10]!")

    train_snaps = []
    val_snaps = []
    for npz_file in datafiles:
        loaded_npz = np.load(npz_file)
        snaps = loaded_npz['rydberg_populations']
        snaps = np.transpose(snaps, (2, 1, 0)).astype(np.float32)

        if crop is not None:
            snaps = _center_crop(snaps, crop)
        if border_crop is not None:
            snaps = _border_crop(snaps, crop)

        if postselect:
            mask_npz = npz_file.replace('rydberg', 'rearrangement_mask')
            loaded_mask_npz = np.load(mask_npz)
            mask = loaded_mask_npz['rearrangement_mask']
            snaps = snaps[mask]

        # Train/val split set by `fold`.
        val_start = int((fold - 1) * 0.1 * len(snaps))
        val_end = int(fold * 0.1 * len(snaps))
        val_snaps.append(snaps[val_start:val_end])

        if val_start > 0:
            train_snaps.append(snaps[:val_start])
        if val_end < len(snaps):
            train_snaps.append(snaps[val_end:])


    train_snaps = np.concatenate(train_snaps, axis=0)
    val_snaps = np.concatenate(val_snaps, axis=0)
    # Add a channel dimension to the snapshot tensors
    train_snaps = torch.tensor(train_snaps).unsqueeze(1)
    val_snaps = torch.tensor(val_snaps).unsqueeze(1)
    return train_snaps, val_snaps


def from_config(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """ From a config file, produce the train and validation data loaders.
    """
    base_data_dir = config['Data Files']['base_dir']
    datasets = config['Data Files']['datasets']
    batch_size = config['Training']['batch_size']
    augment = config['Preprocessing']['augment']
    oversample = config['Preprocessing']['oversample']
    # Evaluate the string to the proper function in this file
    load_func = eval(config['Data Files']['loader_func'])
    load_kwargs = config['Loader Kwargs']

    # Create paths to data files
    train_val_splits = map(
        lambda dset: load_func(
            os.path.join(base_data_dir, 'Dataset{}'.format(dset)), **load_kwargs
        ), datasets
    )
    train_tensors = []
    val_tensors = []
    for train_tens, val_tens in train_val_splits:
        train_tensors.append(train_tens)
        val_tensors.append(val_tens)

    # Build datasets / loaders
    transform = RandomAugmentation() if augment else None
    train_loader = build_data_loader(train_tensors, batch_size=batch_size,
                                     transform=transform, oversample=oversample,
                                     shuffle=True)
    val_loader = build_data_loader(val_tensors, batch_size=batch_size, transform=None,
                                   oversample=False, shuffle=False)

    return train_loader, val_loader


def from_config_ovr(config: Dict) -> Generator[Tuple[DataLoader, DataLoader], None, None]:
    """ From a config file, yields train and valdation loaders for all combinations
         where class 0 is one of the original classes, and class 1 is the rest of the
         classes together.
    """
    base_data_dir = config['Data Files']['base_dir']
    datasets = config['Data Files']['datasets']
    batch_size = config['Training']['batch_size']
    augment = config['Preprocessing']['augment']
    oversample = config['Preprocessing']['oversample']
    # Evaluate the string to the proper function in this file
    load_func = eval(config['Data Files']['loader_func'])
    load_kwargs = config['Loader Kwargs']

    # Create paths to data files
    train_val_splits = map(
        lambda dset: load_func(
            os.path.join(base_data_dir, 'Dataset{}'.format(dset)), **load_kwargs
        ), datasets
    )
    train_tensors = []
    val_tensors = []
    for train_tens, val_tens in train_val_splits:
        train_tensors.append(train_tens)
        val_tensors.append(val_tens)

    # For OVR, need to also handle oversampling here so that the "rest" class equally represents
    #  all of the other classes
    if oversample:
        max_len_t = max(map(len, train_tensors))
        max_len_v = max(map(len, val_tensors))
        for i, t in enumerate(train_tensors):
            num_copies, extra = divmod(max_len_t, len(t))
            # Add additional copies of snapshots up to match max_len
            train_tensors[i] = torch.cat(num_copies * [t] + [t[:extra]], dim=0)
        for i, t in enumerate(val_tensors):
            num_copies, extra = divmod(max_len_v, len(t))
            # Add additional copies of snapshots up to match max_len
            val_tensors[i] = torch.cat(num_copies * [t] + [t[:extra]], dim=0)

    # Build datasets / loaders
    transform = RandomAugmentation() if augment else None

    num_classes = len(train_tensors)
    for n in range(num_classes):
        out_class_train_tensors = [train_tensors[m] for m in range(num_classes) if m != n]
        ovr_train_tensors = [train_tensors[n], torch.cat(out_class_train_tensors, dim=0)]
        train_loader = build_data_loader(ovr_train_tensors, batch_size=batch_size,
                                         transform=transform, oversample=True,
                                         shuffle=True)
        out_class_val_tensors = [val_tensors[m] for m in range(num_classes) if m != n]
        ovr_val_tensors = [val_tensors[n], torch.cat(out_class_val_tensors, dim=0)]
        val_loader = build_data_loader(ovr_val_tensors, batch_size=batch_size, transform=None,
                                       oversample=True, shuffle=False)

        yield train_loader, val_loader
