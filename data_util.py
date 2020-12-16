""" Utility functions and classes for loading / processing / augmenting data
"""
import pickle
import itertools
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _center_crop(arr, size):
    """ Like torchvision.transforms.CenterCrop, but on numpy arrays
    """
    w, h = arr.shape[-2], arr.shape[-1]
    if w < size or h < size:
        raise ValueError("Array to crop is smaller than crop region.")

    margin_w, margin_h = w - size, h - size
    left, right = margin_w // 2, int(np.ceil(margin_w / 2))
    bot, top = margin_h // 2, int(np.ceil(margin_h / 2))
    return arr[..., left:w-right, bot:h-top]


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
    cropped_arr[..., zero_mask] = 0
    return cropped_arr


def _vflip(arr):
    return torch.flip(arr, [-2])


def _hflip(arr):
    return torch.flip(arr, [-1])


def _rot(arr, k):
    return torch.rot90(arr, k, [-2, -1])


def _flip_spins(snap):
    """ Flips all spins in the spin channel of an input snapshots
    """
    flipped = torch.empty_like(snap)
    flipped[..., 0, :, :] = -snap[..., 0, :, :]
    flipped[..., 1:, :, :] = snap[..., 1:, :, :]
    return flipped


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
                    oversample_index = torch.randint(len(indices))
                    epoch_sample_indices.append(indices[oversample_index])

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


class RandomGroupedLoader:
    """ Acts like a DataLoader, but each element in the batch returned is a group of snapshots
         of the same class. Also oversamples examples to make an even class distribution.
    """
    def __init__(self, snapshot_dict, group_size, transform=None, batch_size=1,
                 shuffle=False, balance=False):
        self.loaders = {
            label: DataLoader(
                TransformedDataset(snapshots, transform=transform),
                batch_size=group_size,
                drop_last=True,
                shuffle=shuffle,
                num_workers=4
            )
            for label, snapshots in snapshot_dict.items()
        }
        self.load_iters = {
            label: _repeat_iter(loader) for label, loader in self.loaders.items()
        }
        self.num_classes = len(snapshot_dict)
        self.largest_class_size = max(len(v) for v in self.loaders.values()) if self.num_classes != 0 else 0
        self.group_size = group_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance = balance
        if self.num_classes != 0:
            # This is really dumb, change this...
            avail_label = next(iter(snapshot_dict.keys()))
            self.snapshot_size = snapshot_dict[avail_label][0].shape[-1]
        else:
            self.snapshot_size = 0

    def __iter__(self):
        # For this iteration, decide in what order we'll sample from each class
        if self.balance:
            # Oversample every class so that we sample the same number of examples from each class
            class_sampler_order = np.concatenate(
                [np.full(self.largest_class_size, k) for k in self.loaders.keys()]
            )
        else:
            class_sampler_order = np.concatenate(
                [np.full(len(v), k) for k, v in self.loaders.items()]
            )

        if self.shuffle:
            class_sampler_order = np.random.permutation(class_sampler_order)
        for i in range(self.__len__()):
            start_idx = i * self.batch_size
            batch_snapshots = []
            batch_labels = class_sampler_order[start_idx:start_idx+self.batch_size]
            for label in batch_labels:
                batch_snapshots.append(next(self.load_iters[label])[0])
            yield torch.stack(batch_snapshots, dim=0), torch.tensor(batch_labels)

    def __len__(self):
        if self.balance:
            return int(np.ceil(self.largest_class_size * self.num_classes / self.batch_size))
        else:
            return int(np.ceil(sum(len(loader) for loader in self.loaders.values()) / self.batch_size))

    def extend(self, other):
        """ Extends this loader with the data contained in another RandomGroupedLoader
        """
        self.loaders = {
            label: DataLoader(
                torch.utils.data.ConcatDataset([self_loader.dataset, other.loaders[label].dataset]),
                batch_size=self.group_size,
                drop_last=True,
                shuffle=self.shuffle,
            )
            for label, self_loader in self.loaders.items()
        }
        self.load_iters = {
            label: _repeat_iter(loader) for label, loader in self.loaders.items()
        }
        self.largest_class_size = max(len(v) for v in self.loaders.values())


def single_grouped_loader(snapshot_dict, transform=None, shuffle=False, **kwargs):
    """ For a group size of 1, it is preferable to just add an extra dimension to the snapshots
         rather than using the (considerably slower) RandomGroupedLoader. This function makes a DataLoader
         which does that.
        Warning: This does not implement oversampling! So, ensure your class distributions are at least
         relatively close on your dataset if using this.
    """
    if len(snapshot_dict) == 0:
        return None

    snapshots = torch.cat(tuple(snapshot_dict.values()))
    # Add a fake "group" dimension of size 1
    snapshots = snapshots.unsqueeze(1)
    labels = torch.cat(tuple(
        torch.full((len(snapshot_dict[label]),), label, dtype=torch.int64)
        for label in snapshot_dict.keys()
    ), dim=0)
    dataset = TransformedDataset(snapshots, labels, transform=transform)

    sampler = BalancedSampler(labels) if shuffle else None
    loader = DataLoader(dataset, sampler=sampler, **kwargs)
    return loader


def build_datasets(pkl_files: List[str], labels: List[int], group_size=1, batch_size=1,
                   transform=None, oversample=True, crop=None, circle_crop=False,):
    """ Builds train/eval sets from pickled data files.
          pkl_files:    List of pickle files of snapshots
          labels:       List of class labels for each pickle file.
          group_size:   Collates snapshots of the same class into groups of this size along an extra dim
          batch_size:   The batch size of the loaders
          transform:    An optional transformation to apply to training snapshots as they're sampled
          oversample:   If True, will oversample classes in the training set to provide an even distribution.
                         If False, will just completely drop snapshots to make the distribution even.
          crop:         If not None, crops to a square of the given size
          circle_crop:  If True, crops snapshots to the circular region obtained from experiment
        The returned loaders will produce tensors of shape
            [batch_size, group_size, num_channels, 10, 10]
         along with label vectors of length [batch_size]
    """
    if len(pkl_files) != len(labels):
        raise ValueError('pkl_files and labels not the same length.')

    all_labels = torch.unique(torch.tensor(labels)).tolist()

    # These dicts will map class labels to tensors of snapshots
    train_snapshot_dict = {l: [] for l in all_labels}
    val_snapshot_dict = {l: [] for l in all_labels}
    for pkl_file, label in zip(pkl_files, labels):
        with open(pkl_file, "rb") as ifile:
            # List of lists [singles, spinups, spindowns]
            snapshot_dict = pickle.load(ifile)
            # Include both the spin up and the spin down snapshots -- the
            #   statistics should be identical for both of them
            raw_snapshots = np.stack(snapshot_dict["snapshots"], axis=0)

            if crop is not None:
                # Crop out the central square part of the snapshots, masked to shape of experiment
                cropped_snapshots = _center_crop(raw_snapshots, crop)
            elif circle_crop:
                cropped_snapshots = _circle_crop(raw_snapshots)
            else:
                cropped_snapshots = raw_snapshots

            train_idxs = snapshot_dict["train_idxs"]
            val_idxs = snapshot_dict["val_idxs"]
            train_snapshot_dict[label].append(cropped_snapshots[train_idxs])
            val_snapshot_dict[label].append(cropped_snapshots[val_idxs])

    # Collect snapshots from all of the files into single tensors
    for snapshot_dict in [train_snapshot_dict, val_snapshot_dict]:
        for label, snapshots in snapshot_dict.items():
            # Stack all of the snapshots into an array of shape [NSNAPSHOTS, 10, 10]
            snapshot_dict[label] = torch.tensor(np.concatenate(snapshots, axis=0), dtype=torch.float32)
            # If missing a channel dimension, add a singleton dimension
            if len(snapshot_dict[label].shape) < 4:
                snapshot_dict[label] = snapshot_dict[label].unsqueeze(1)
        # Remove empty tensors from dict
        empty_labels = [label for label, tensor in snapshot_dict.items() if len(tensor) == 0]
        for label in empty_labels:
            del snapshot_dict[label]

    # If we're not oversampling, drop snapshots to make the train distribution even
    if not oversample:
        min_size = min(len(snapshots) for snapshots in train_snapshot_dict.values())
        for label in train_snapshot_dict.keys():
            train_snapshot_dict[label] = train_snapshot_dict[label][:min_size]

    # This is *much* faster
    if group_size == 1:
        train_loader = single_grouped_loader(train_snapshot_dict, transform=transform,
                                             batch_size=batch_size, shuffle=True,)
        val_loader = single_grouped_loader(val_snapshot_dict, transform=transform,
                                           batch_size=batch_size, shuffle=False,)
    else:
        train_loader = RandomGroupedLoader(
            train_snapshot_dict, group_size,
            transform=transform, batch_size=batch_size, shuffle=True, balance=oversample,
        )
        val_loader = RandomGroupedLoader(
            val_snapshot_dict, group_size,
            transform=transform, batch_size=batch_size, shuffle=False, balance=False,
        )

    return train_loader, val_loader
