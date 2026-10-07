"""
Quality of the N2 pseudo-labels produced by a teacher checkpoint, with and
without the class-label correction and the confidence threshold. The hidden
N2 ground truth is used for this diagnostic only.

Usage:
    python scripts/pseudo_label_quality.py --config configs/m3_self_train_unet_r50.yaml \
        --checkpoint runs/b1_unet_r50/seed0/model.pt --seed 0
"""
import argparse
import json

import numpy as np
import torch

from wsss.config import load_config
from wsss.constants import IGNORE_INDEX
from wsss.data import compute_grid_tags, list_area_ids, load_image, load_label, split_data
from wsss.inference import sliding_window_probabilities
from wsss.metrics import ConfusionMatrix
from wsss.models import build_model
from wsss.pseudo_label import class_thresholds, make_pseudo_labels, tag_masked_probabilities

VARIANTS = {"raw": (False, False), "thresholded": (False, True),
            "corrected": (True, False), "corrected + thresholded": (True, True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed)
    root, device = config["data_root"], config["device"]
    _, n2_ids, _ = split_data(list_area_ids(root), seed=config["seed"], **config["split"])

    model = build_model(dict(config["model"], pretrained=False))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    thresholds = class_thresholds(config["pseudo"]["threshold"], config["pseudo"]["car_threshold"])
    no_threshold = torch.zeros_like(thresholds)

    matrices = {name: ConfusionMatrix() for name in VARIANTS}
    kept = {name: 0 for name in VARIANTS}
    total = 0
    for area_id in n2_ids:
        label = load_label(root, area_id)
        probabilities = torch.from_numpy(sliding_window_probabilities(
            model, load_image(root, area_id), device=device, amp=config["train"]["amp"],
            window=config["eval"]["window"], stride=config["eval"]["stride"]))
        for _, y, x, h, w, tags in compute_grid_tags([label], config["tags"]["cell_size"],
                                                     config["tags"]["min_fraction"]):
            cell = probabilities[:, y:y + h, x:x + w][None]
            gt = label[y:y + h, x:x + w]
            total += gt.size
            for name, (correct, threshold) in VARIANTS.items():
                p = tag_masked_probabilities(cell.clamp(min=1e-8).log(),
                                             torch.from_numpy(tags)[None]) if correct else cell
                pseudo, _ = make_pseudo_labels(p, thresholds if threshold else no_threshold)
                pseudo = pseudo[0].numpy()
                keep = pseudo != IGNORE_INDEX
                matrices[name].update(pseudo[keep], gt[keep])
                kept[name] += int(keep.sum())

    results = {}
    for name in VARIANTS:
        summary = matrices[name].summary()
        results[name] = {"coverage": kept[name] / total, "miou": summary["miou"],
                         "per_class_iou": summary["per_class_iou"]}
        print("{:26s} coverage {:.3f}  mIoU {:.3f}  ({})".format(
            name, kept[name] / total, summary["miou"],
            ", ".join("{:.3f}".format(v) for v in summary["per_class_iou"].values())))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
