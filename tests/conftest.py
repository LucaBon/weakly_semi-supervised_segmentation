import os

import numpy as np
import pytest
from PIL import Image
from skimage import io


@pytest.fixture
def synthetic_root(tmp_path):
    """A tiny dataset in the prepared layout: 6 areas of 300x320 px."""
    rng = np.random.default_rng(0)
    for folder in ("images", "labels", "labels_eroded"):
        os.makedirs(tmp_path / folder)
    for area_id in range(1, 7):
        label = np.zeros((300, 320), dtype=np.uint8)
        label[:, 100:200] = 1
        label[150:, :] = 2
        label[:80, 220:] = 3
        label[20:40, 20:60] = 4
        label[260:, 280:] = 5
        image = (rng.random((300, 320, 3)) * 60).astype(np.uint8) + label[..., None] * 30
        io.imsave(str(tmp_path / "images" / "area{}.tif".format(area_id)), image,
                  check_contrast=False)
        Image.fromarray(label).save(tmp_path / "labels" / "area{}.png".format(area_id))
        eroded = label.copy()
        eroded[:, 99:101] = 255
        Image.fromarray(eroded).save(tmp_path / "labels_eroded" / "area{}.png".format(area_id))
    return str(tmp_path)
