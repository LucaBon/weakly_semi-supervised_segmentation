"""
Evaluation-only boundary refinement: test-time augmentation (TTA) and PAMR,
alone and combined, on existing checkpoints. Variants are compared on the
dev split (seed 99) to choose one, and reported on the given seeds.
Metrics: mIoU and car IoU on full and eroded ground truth, and the boundary
F-score (tolerance 2 px = 18 cm) on the full ground truth.

Usage:
    python scripts/boundary_refinement.py --config configs/b1_unet_r50.yaml \
        --run b1_unet_r50 --seeds 99 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from wsss.config import load_config
from wsss.constants import CAR
from wsss.data import list_area_ids, load_image, load_label, split_data
from wsss.inference import decide, sliding_window_probabilities
from wsss.metrics import BoundaryScore, ConfusionMatrix
from wsss.models import build_model
from wsss.refine import refine_probabilities

PAMR_SETTINGS = {"pamr-light": {"iterations": 5, "dilations": (1, 2, 4, 8)},
                 "pamr": {"iterations": 10, "dilations": (1, 2, 4, 8, 12, 24)}}


def evaluate_seed(config_path, run, seed):
    config = load_config(config_path, seed=seed)
    root, device = config["data_root"], config["device"]
    _, _, test_ids = split_data(list_area_ids(root), seed=seed, **config["split"])
    model = build_model(dict(config["model"], pretrained=False))
    model.load_state_dict(torch.load("runs/{}/seed{}/model.pt".format(run, seed),
                                     map_location="cpu"))
    model.to(device)
    variants = ["base", "tta"] + list(PAMR_SETTINGS) + ["tta+" + k for k in PAMR_SETTINGS]
    matrices = {v: {"full": ConfusionMatrix(), "eroded": ConfusionMatrix()} for v in variants}
    boundaries = {v: BoundaryScore(tolerance=2, classes=[CAR]) for v in variants}
    timing = {v: 0.0 for v in variants}
    for area_id in test_ids:
        image = load_image(root, area_id)
        label, eroded = load_label(root, area_id), load_label(root, area_id, eroded=True)
        probabilities = {}
        for name, tta in (("base", False), ("tta", True)):
            start = time.time()
            probabilities[name] = sliding_window_probabilities(
                model, image, window=config["eval"]["window"],
                stride=config["eval"]["stride"], tta=tta, device=device)
            timing[name] += time.time() - start
        for source in ("base", "tta"):
            for key, settings in PAMR_SETTINGS.items():
                name = key if source == "base" else "tta+" + key
                start = time.time()
                probabilities[name] = refine_probabilities(probabilities[source], image,
                                                           device=device, **settings)
                timing[name] += time.time() - start
        for name, p in probabilities.items():
            prediction = decide(p)
            matrices[name]["full"].update(prediction, label)
            matrices[name]["eroded"].update(prediction, eroded)
            boundaries[name].update(prediction, label)
    results = {}
    for name in variants:
        full, ero = matrices[name]["full"].summary(), matrices[name]["eroded"].summary()
        boundary = boundaries[name].summary()
        results[name] = {"miou_full": full["miou"], "miou_eroded": ero["miou"],
                         "car_iou_full": full["per_class_iou"]["Car"],
                         "car_iou_eroded": ero["per_class_iou"]["Car"],
                         "boundary_f1": boundary["all"]["f1"],
                         "car_boundary_f1": boundary[CAR]["f1"],
                         "seconds_per_image": timing[name] / len(test_ids)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[99, 0, 1, 2])
    args = parser.parse_args()
    all_results = {}
    for seed in args.seeds:
        results = evaluate_seed(args.config, args.run, seed)
        all_results[seed] = results
        print("seed {}:".format(seed))
        print("  variant          mIoU full  mIoU ero  car full  car ero  bound F1  car bound F1")
        for name, r in results.items():
            print("  {:15s}  {:.3f}      {:.3f}     {:.3f}     {:.3f}    {:.3f}     {:.3f}".format(
                name, r["miou_full"], r["miou_eroded"], r["car_iou_full"],
                r["car_iou_eroded"], r["boundary_f1"], r["car_boundary_f1"]), flush=True)
    out = os.path.join("runs", args.run, "boundary_refinement.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("written", out)


if __name__ == "__main__":
    main()
