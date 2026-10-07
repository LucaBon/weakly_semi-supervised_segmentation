import random

import numpy as np
import torch
import torch.nn.functional as F

from wsss.constants import MEAN, STD


def random_geometric(*arrays):
    """
    Same random flip / 90-degree rotation applied to every (H, W[, C]) array.
    These transforms preserve crop-level tags. Aerial imagery has no canonical
    orientation, so all 8 dihedral transforms are valid.
    """
    k = random.randint(0, 3)
    flip = random.random() < 0.5
    results = []
    for array in arrays:
        array = np.rot90(array, k, axes=(0, 1))
        if flip:
            array = array[:, ::-1]
        results.append(np.ascontiguousarray(array))
    return tuple(results)


def strong_photometric(images, p_jitter=0.8, p_blur=0.5):
    """
    Per-sample color jitter and Gaussian blur on a normalized (B, 3, H, W)
    batch, as in the strong view of FixMatch/UniMatch.
    """
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    x = images * std + mean
    batch = x.shape[0]

    def factor(strength):
        return 1 + (torch.rand(batch, 1, 1, 1, device=x.device) * 2 - 1) * strength

    jitter = (torch.rand(batch, 1, 1, 1, device=x.device) < p_jitter).float()
    brightness = 1 + jitter * (factor(0.5) - 1)
    contrast = 1 + jitter * (factor(0.5) - 1)
    saturation = 1 + jitter * (factor(0.5) - 1)
    x = x * brightness
    gray_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - gray_mean) * contrast + gray_mean
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * saturation + gray
    x = x.clamp(0, 1)

    blur = torch.rand(batch, device=x.device) < p_blur
    if blur.any():
        sigma = random.uniform(0.1, 2.0)
        x[blur] = gaussian_blur(x[blur], sigma).to(x.dtype)
    return (x - mean) / std


def gaussian_blur(x, sigma):
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kernel_h = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_v = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), kernel_h,
                 groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), kernel_v,
                    groups=channels)


def cutmix_masks(batch, height, width, device, p=0.5, area=(0.02, 0.4)):
    """
    One random box per sample (or an empty mask with probability 1 - p).
    Returns:
        torch.Tensor: (B, H, W) bool, True where the partner sample is pasted
    """
    masks = torch.zeros(batch, height, width, dtype=torch.bool, device=device)
    for i in range(batch):
        if random.random() > p:
            continue
        box_area = random.uniform(*area) * height * width
        ratio = random.uniform(0.3, 1 / 0.3)
        box_h = min(height, int(round((box_area * ratio) ** 0.5)))
        box_w = min(width, int(round((box_area / ratio) ** 0.5)))
        y = random.randint(0, height - box_h)
        x = random.randint(0, width - box_w)
        masks[i, y:y + box_h, x:x + box_w] = True
    return masks


def apply_cutmix(images, masks, permutation):
    """Paste images[permutation] into images where masks is True."""
    return torch.where(masks.unsqueeze(1), images[permutation], images)
