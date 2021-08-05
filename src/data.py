import random
import os

import torch
import numpy as np
from skimage import io

from constants import IMAGE_FOLDER_PATH, \
    LABELS_FOLDER_PATH, \
    CACHE, \
    BATCH_SIZE, \
    THRESHOLD_IMAGE_LABELS, \
    NAME_FORMAT

from utils import convert_from_color, \
    extract_random_patch,\
    calculate_image_labels, \
    split_into_tiles


class ISPRSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_indexes,
                 tile_size,
                 augmentation,
                 image_path=IMAGE_FOLDER_PATH,
                 label_path=LABELS_FOLDER_PATH,
                 name_format=NAME_FORMAT,
                 are_image_labels=False,
                 cache=False,
                 ):

        super(ISPRSDataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.are_image_labels = are_image_labels
        self.tile_size = tile_size

        self.data_files = [os.path.join(image_path, name_format.format(index))
                           for index in image_indexes]
        self.label_files = [os.path.join(label_path, name_format.format(index))
                            for index in image_indexes]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class ISPRSTrainDataset(ISPRSDataset):
    def __init__(self,
                 image_indexes,
                 tile_size,
                 augmentation,
                 image_path=IMAGE_FOLDER_PATH,
                 label_path=LABELS_FOLDER_PATH,
                 name_format=NAME_FORMAT,
                 are_image_labels=False,
                 cache=False,
                 ):

        super(ISPRSTrainDataset, self).__init__(image_indexes,
                                                tile_size,
                                                augmentation,
                                                image_path,
                                                label_path,
                                                name_format,
                                                are_image_labels,
                                                cache)

    def __len__(self):
        # Default epoch size is 10,000 samples
        return 10000

    @classmethod  # flipping and mirroring images for data augmentation
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1 / 255 * np.asarray(
                io.imread(self.data_files[random_idx]).transpose((2, 0, 1)),
                dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(
                convert_from_color(io.imread(self.label_files[random_idx])),
                dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = extract_random_patch(img=data,
                                              window_shape=self.tile_size)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        if self.augmentation:
            data_p, label_p = self.data_augmentation(data_p, label_p)

        if self.are_image_labels:
            label_p = \
                calculate_image_labels(label_p,
                                       threshold_image_labels=THRESHOLD_IMAGE_LABELS)
            data_p = resnet_normalization(data_p)
        data_pollo = data_p
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


class ISPRSTestDataset(ISPRSDataset):
    def __init__(self,
                 image_indexes,
                 tile_size,
                 augmentation=False,
                 image_path=IMAGE_FOLDER_PATH,
                 label_path=LABELS_FOLDER_PATH,
                 name_format=NAME_FORMAT,
                 are_image_labels=False,
                 cache=False,
                 ):

        super(ISPRSTestDataset, self).__init__(image_indexes,
                                               tile_size,
                                               augmentation,
                                               image_path,
                                               label_path,
                                               name_format,
                                               are_image_labels,
                                               cache)
        self.tiles = []
        self.tiles_labels = []
        for image_path in self.data_files:
            self.tiles.extend(split_into_tiles(image_path=image_path,
                                               tile_size=self.tile_size))
        for label_path in self.label_files:
            self.tiles_labels.extend(split_into_tiles(image_path=label_path,
                                                      tile_size=self.tile_size))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = 1 / 255 * np.asarray(self.tiles[idx].transpose((2, 0, 1)),
                                    dtype='float32')
        label = np.asarray(
            convert_from_color(self.tiles_labels[idx]),
            dtype='int64')
        if self.are_image_labels:
            label = \
                calculate_image_labels(label,
                                       threshold_image_labels=THRESHOLD_IMAGE_LABELS)

        # Return the torch.Tensor values
        return (torch.from_numpy(data),
                torch.from_numpy(label))


def split_data(labels_folder_path=LABELS_FOLDER_PATH,
               train_pixel_samples=3,
               train_image_samples=23,
               seed=341):
    random.seed(seed)
    label_image_names = os.listdir(labels_folder_path)
    all_ids = [f.split('area')[-1].split('.')[0] for f in label_image_names]

    # Random tile numbers for train/test split
    train_pixel_ids = random.sample(all_ids, train_pixel_samples)
    remaining_ids = [idx for idx in all_ids if idx not in train_pixel_ids]
    train_image_ids = random.sample(remaining_ids,
                                    train_image_samples)
    test_ids = [idx for idx in all_ids if idx not in train_pixel_ids and
                idx not in train_image_ids]

    print("Images indexes for training with pixel labels : ", train_pixel_ids)
    print("Images indexes for training with image labels: ", train_image_ids)
    print("Images indexes for testing : ", test_ids)
    return train_pixel_ids, train_image_ids, test_ids


def load_train_pixel_ids(train_pixel_ids,
                         tile_size,
                         batch_size,
                         cache=CACHE,
                         augmentation=True):
    train_pixel_set = ISPRSTrainDataset(train_pixel_ids,
                                        tile_size=tile_size,
                                        cache=cache,
                                        augmentation=augmentation)
    train_pixel_loader = torch.utils.data.DataLoader(train_pixel_set,
                                                     batch_size=batch_size)
    return train_pixel_loader


def load_train_image_ids(train_image_ids,
                         tile_size,
                         batch_size,
                         cache=CACHE,
                         augmentation=True):
    train_image_set = ISPRSTrainDataset(train_image_ids,
                                        tile_size=tile_size,
                                        are_image_labels=True,
                                        cache=cache,
                                        augmentation=augmentation)
    train_image_loader = torch.utils.data.DataLoader(train_image_set,
                                                     batch_size=batch_size)
    return train_image_loader


def load_test_ids(test_ids,
                  tile_size,
                  batch_size=BATCH_SIZE):
    test_set = ISPRSTestDataset(test_ids,
                                tile_size=tile_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size)
    return test_loader


def resnet_normalization(data):
    # Data is normalized with
    # mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    data_p = np.empty_like(data)

    data_p[0, ...] = (data[0, ...] - mean[0]) / \
                   std[0]
    data_p[1, ...] = (data[1, ...] - mean[1]) / \
                   std[1]
    data_p[2, ...] = (data[2, ...] - mean[2]) / \
                   std[2]
    return data_p
