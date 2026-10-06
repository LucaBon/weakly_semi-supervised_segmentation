import numpy as np

from wsss.constants import CAR, IGNORE_INDEX
from wsss.data import (PixelCropDataset, TagCellDataset, compute_grid_tags,
                       grid_cells, list_area_ids, load_label, split_data)


def test_split_is_deterministic_disjoint_and_complete():
    ids = [str(i) for i in range(1, 34)]
    a = split_data(ids, 3, 23, seed=0)
    assert a == split_data(list(reversed(ids)), 3, 23, seed=0)
    assert a != split_data(ids, 3, 23, seed=1)
    n1, n2, test = map(set, a)
    assert (len(n1), len(n2), len(test)) == (3, 23, 7)
    assert n1 | n2 | test == set(ids) and not (n1 & n2 or n1 & test or n2 & test)


def test_list_area_ids_sorted_numerically(synthetic_root):
    assert list_area_ids(synthetic_root) == ["1", "2", "3", "4", "5", "6"]


def test_grid_cells_cover_the_whole_image_without_overlap():
    count = np.zeros((450, 610), dtype=int)
    for y, x, h, w in grid_cells(450, 610, 200):
        assert h <= 200 and w <= 200
        count[y:y + h, x:x + w] += 1
    assert (count == 1).all()


def test_grid_tags(synthetic_root):
    label = load_label(synthetic_root, "1")
    cells = compute_grid_tags([label], cell_size=100)
    index, y, x, h, w, tags = cells[0]
    assert (index, y, x, h, w) == (0, 0, 0, 100, 100)
    np.testing.assert_array_equal(tags, [1, 0, 0, 0, 1])


def test_tag_cell_dataset_valid_region_and_targets(synthetic_root):
    label = load_label(synthetic_root, "1")
    image = np.zeros((300, 320, 3), dtype=np.uint8)
    cells = compute_grid_tags([label], cell_size=100)
    pseudo = [np.full((h, w), 2, dtype=np.uint8) for _, _, _, h, w, _ in cells]
    dataset = TagCellDataset([image], cells, cell_size=100, context=14,
                             pseudo_labels=pseudo, augmentation=True)
    sample = dataset[0]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["valid"].sum() == 100 * 100
    target = sample["target"]
    assert ((target == 2) == sample["valid"]).all()
    assert (target[~sample["valid"]] == IGNORE_INDEX).all()


def test_pixel_crop_dataset_car_sampling(synthetic_root):
    label = load_label(synthetic_root, "1")
    image = np.zeros((300, 320, 3), dtype=np.uint8)
    dataset = PixelCropDataset([image], [label], crop_size=64, car_probability=1.0)
    for _ in range(10):
        _, target = dataset[0]
        assert target.shape == (64, 64)
        assert (target == CAR).any()


def test_tag_cell_dataset_edge_cell(synthetic_root):
    label = load_label(synthetic_root, "1")  # 300 x 320: last column cells are 20 px wide
    image = np.zeros((300, 320, 3), dtype=np.uint8)
    cells = compute_grid_tags([label], cell_size=100)
    item = next(i for i, cell in enumerate(cells) if cell[4] == 20)
    dataset = TagCellDataset([image], cells, cell_size=100, context=14,
                             augmentation=False)
    sample = dataset[item]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["valid"].sum() == 100 * 20
    assert sample["valid"][14:114, 14:34].all()
