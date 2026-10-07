"""
Data loading for the prepared Vaihingen layout (see scripts/prepare_vaihingen.py):

    <data_root>/images/area{id}.tif         IRRG uint8 (H, W, 3)
    <data_root>/labels/area{id}.png         class indices uint8 (H, W)
    <data_root>/labels_eroded/area{id}.png  class indices, boundaries = 255
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from skimage import io

from wsss.augment import random_geometric
from wsss.constants import CAR, IGNORE_INDEX, MEAN, STD
from wsss.labels import calculate_image_labels


def image_path(data_root, area_id):
    return os.path.join(data_root, "images", "area{}.tif".format(area_id))


def label_path(data_root, area_id, eroded=False):
    folder = "labels_eroded" if eroded else "labels"
    return os.path.join(data_root, folder, "area{}.png".format(area_id))


def list_area_ids(data_root):
    """Area ids sorted numerically, so splits do not depend on os.listdir order."""
    names = os.listdir(os.path.join(data_root, "labels"))
    return sorted((f[len("area"):-len(".png")] for f in names
                   if f.startswith("area") and f.endswith(".png")), key=int)


def split_data(all_ids, n_pixel=3, n_image=23, seed=0):
    """
    Deterministic split into N1 (pixel labels), N2 (crop tags) and test ids.
    Args:
        all_ids (list): area ids
        n_pixel (int): number of images with pixel-level labels (N1)
        n_image (int): number of images with crop-level tags (N2)
        seed (int): split seed

    Returns:
        tuple(list, list, list): N1 ids, N2 ids, test ids
    """
    ids = sorted(all_ids, key=int)
    shuffled = random.Random(seed).sample(ids, len(ids))
    pixel_ids = sorted(shuffled[:n_pixel], key=int)
    image_ids = sorted(shuffled[n_pixel:n_pixel + n_image], key=int)
    test_ids = sorted(shuffled[n_pixel + n_image:], key=int)
    return pixel_ids, image_ids, test_ids


def load_image(data_root, area_id):
    return np.ascontiguousarray(io.imread(image_path(data_root, area_id))[..., :3])


def load_label(data_root, area_id, eroded=False):
    return np.asarray(Image.open(label_path(data_root, area_id, eroded)))


def to_tensor(image):
    """(H, W, 3) uint8 -> normalized (3, H, W) float tensor."""
    image = image.astype(np.float32) / 255.
    image = (image - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class PixelCropDataset(torch.utils.data.Dataset):
    """
    Random crops from pixel-labelled images. With probability `car_probability`
    the crop is centered on a random car pixel, to counter the rarity of cars.
    """

    def __init__(self, images, labels, crop_size=256, length=10000,
                 augmentation=True, car_probability=0.0):
        self.images = images
        self.labels = labels
        self.crop_size = crop_size
        self.length = length
        self.augmentation = augmentation
        self.car_probability = car_probability
        self.car_pixels = [np.argwhere(label == CAR) for label in labels]

    def __len__(self):
        return self.length

    def _crop_origin(self, index):
        height, width = self.labels[index].shape
        size = self.crop_size
        if random.random() < self.car_probability and len(self.car_pixels[index]):
            y, x = self.car_pixels[index][random.randrange(len(self.car_pixels[index]))]
            y0 = int(np.clip(y - size // 2, 0, height - size))
            x0 = int(np.clip(x - size // 2, 0, width - size))
            return y0, x0
        return random.randint(0, height - size), random.randint(0, width - size)

    def __getitem__(self, item):
        index = random.randrange(len(self.images))
        y0, x0 = self._crop_origin(index)
        size = self.crop_size
        image = self.images[index][y0:y0 + size, x0:x0 + size]
        label = self.labels[index][y0:y0 + size, x0:x0 + size]
        if self.augmentation:
            image, label = random_geometric(image, label)
        return to_tensor(image), torch.from_numpy(label.astype(np.int64))


def grid_cells(height, width, cell_size):
    """
    Non-overlapping cells of a fixed grid covering the whole image. The cells
    of the last row/column are smaller when the image size is not a multiple
    of `cell_size`; they are not shifted inwards, since overlapping tagged
    cells would locate classes more finely than the grid.
    Returns:
        list: (y, x, height, width) of each cell
    """
    return [(y, x, min(cell_size, height - y), min(cell_size, width - x))
            for y in range(0, height, cell_size)
            for x in range(0, width, cell_size)]


def compute_grid_tags(labels, cell_size=200, min_fraction=0.0):
    """
    Crop-level tags on a fixed grid, computed once from the hidden ground truth.
    This simulates an annotator tagging each 200x200 crop; tags must not be
    recomputed on random crops, which would leak pixel-level information.
    Returns:
        list: one entry per cell, (image index, y, x, height, width, tags)
    """
    cells = []
    for index, label in enumerate(labels):
        for y, x, h, w in grid_cells(*label.shape, cell_size):
            tags = calculate_image_labels(label[y:y + h, x:x + w],
                                          min_fraction=min_fraction)
            cells.append((index, y, x, h, w, tags))
    return cells


class TagCellDataset(torch.utils.data.Dataset):
    """
    Grid cells of tag-labelled images. Each sample is a window of
    cell_size + 2 * context pixels (divisible by 32) with the cell at offset
    `context`, surrounded by image context; a validity mask marks the cell
    itself (edge cells can be smaller than cell_size). It also returns the
    tag vector and, when available, the cell pseudo-labels (IGNORE_INDEX
    elsewhere). Only tag-preserving geometric augmentations are applied.
    """

    def __init__(self, images, cells, cell_size=200, context=28,
                 pseudo_labels=None, augmentation=True):
        self.images = images
        self.cells = cells
        self.cell_size = cell_size
        self.context = context
        self.pseudo_labels = pseudo_labels
        self.augmentation = augmentation
        # extra padding after the image so windows of edge cells fit
        after = context + cell_size
        self.padded = [np.pad(image, ((context, after), (context, after), (0, 0)),
                              mode='reflect') for image in images]

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, item):
        index, y, x, h, w, tags = self.cells[item]
        size, c = self.cell_size + 2 * self.context, self.context
        # in padded coordinates the window around the cell starts at (y, x)
        image = self.padded[index][y:y + size, x:x + size]
        target = np.full((size, size), IGNORE_INDEX, dtype=np.uint8)
        if self.pseudo_labels is not None:
            target[c:c + h, c:c + w] = self.pseudo_labels[item]
        valid = np.zeros((size, size), dtype=np.uint8)
        valid[c:c + h, c:c + w] = 1
        if self.augmentation:
            image, target, valid = random_geometric(image, target, valid)
        return {"image": to_tensor(image),
                "target": torch.from_numpy(target.astype(np.int64)),
                "valid": torch.from_numpy(valid.astype(bool)),
                "tags": torch.from_numpy(tags)}


def infinite_loader(dataset, batch_size, num_workers=2, seed=0):
    """Yields batches forever, reshuffling at every pass."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True,
                                         num_workers=num_workers,
                                         generator=generator,
                                         worker_init_fn=_seed_worker,
                                         persistent_workers=num_workers > 0)
    while True:
        yield from loader


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
