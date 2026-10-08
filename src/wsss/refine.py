"""
Image-guided refinement of class probabilities with PAMR (Pixel-Adaptive
Mask Refinement, Araslanov & Roth, "Single-Stage Semantic Segmentation from
Image Labels", CVPR 2020). Probabilities are iteratively averaged over local
neighbourhoods at several dilations, weighted by colour similarity, so that
predicted borders move onto image edges.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsss.data import to_tensor

# 8 neighbour offsets of a 3x3 kernel
NEIGHBOURS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]


def _shift_kernel(with_centre):
    """(P, 1, 3, 3) kernels copying each neighbour (minus the centre if
    `with_centre`, giving differences)."""
    kernel = torch.zeros(len(NEIGHBOURS), 1, 3, 3)
    for i, (y, x) in enumerate(NEIGHBOURS):
        kernel[i, 0, y, x] = 1
        if with_centre:
            kernel[i, 0, 1, 1] = -1
    return kernel


class PAMR(nn.Module):

    def __init__(self, iterations=10, dilations=(1, 2, 4, 8, 12, 24), temperature=0.1):
        super(PAMR, self).__init__()
        self.iterations = iterations
        self.dilations = dilations
        self.temperature = temperature
        self.register_buffer("copy", _shift_kernel(False))
        self.register_buffer("difference", _shift_kernel(True))

    def _neighbours(self, x, kernel, include_centre=False):
        """(B, K, H, W) -> (B, K, P * len(dilations) [+1], H, W)"""
        batch, channels, height, width = x.shape
        flat = x.reshape(batch * channels, 1, height, width)
        out = [F.conv2d(F.pad(flat, [d] * 4, mode="replicate"), kernel, dilation=d)
               for d in self.dilations]
        if include_centre:
            out.append(flat)
        return torch.cat(out, 1).reshape(batch, channels, -1, height, width)

    def forward(self, image, probabilities):
        """
        Args:
            image (torch.Tensor): (B, 3, H, W) normalized image
            probabilities (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W) refined probabilities
        """
        # colour affinities, normalized by the local standard deviation
        std = self._neighbours(image, self.copy, include_centre=True).std(2, keepdim=True)
        affinity = -self._neighbours(image, self.difference).abs() / (1e-8 + self.temperature * std)
        affinity = affinity.mean(1, keepdim=True).softmax(2)
        for _ in range(self.iterations):
            probabilities = (self._neighbours(probabilities, self.copy) * affinity).sum(2)
        return probabilities


@torch.no_grad()
def refine_probabilities(probabilities, image, iterations=10,
                         dilations=(1, 2, 4, 8, 12, 24), tile=256, margin=96,
                         device="cuda"):
    """
    PAMR on a whole image, tile by tile with a context margin (the refinement
    is local, so tiles are independent away from their margin).
    Args:
        probabilities (np.ndarray): (C, H, W)
        image (np.ndarray): (H, W, 3) uint8

    Returns:
        np.ndarray: (C, H, W) refined probabilities
    """
    pamr = PAMR(iterations, tuple(dilations)).to(device)
    _, height, width = probabilities.shape
    image_tensor = to_tensor(image)
    refined = np.empty_like(probabilities, dtype=np.float32)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            y0, x0 = max(0, y - margin), max(0, x - margin)
            y1, x1 = min(height, y + tile + margin), min(width, x + tile + margin)
            image_tile = image_tensor[:, y0:y1, x0:x1][None].to(device)
            prob_tile = torch.from_numpy(
                np.ascontiguousarray(probabilities[:, y0:y1, x0:x1], dtype=np.float32))[None].to(device)
            out = pamr(image_tile, prob_tile)[0].cpu().numpy()
            th, tw = min(tile, height - y), min(tile, width - x)
            refined[:, y:y + th, x:x + tw] = out[:, y - y0:y - y0 + th, x - x0:x - x0 + tw]
    return refined
