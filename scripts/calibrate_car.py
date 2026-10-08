"""
Post-hoc calibration of the car class. A constant is added to the car
log-probability; it is chosen on the test images of the dev split (seed 99,
never a reported seed) by maximizing mIoU, then applied unchanged to the
reported seeds.

Usage:
    python scripts/calibrate_car.py --config configs/m3_self_train_unet_r50.yaml \
        --run m3_self_train_unet_r50 --dev-seed 99 --apply-seeds 0 1 2
"""
import argparse
import json
import os

import numpy as np
import torch

from wsss.config import load_config
from wsss.constants import CAR
from wsss.data import list_area_ids, load_image, load_label, split_data
from wsss.inference import decide, sliding_window_probabilities
from wsss.metrics import ConfusionMatrix
from wsss.models import build_model

OFFSETS = np.round(np.arange(-3.0, 0.51, 0.25), 2)


def test_probabilities(config_path, run, seed):
    """Probabilities and labels (full, eroded) of the test images of a seed."""
    config = load_config(config_path, seed=seed)
    root = config["data_root"]
    _, _, test_ids = split_data(list_area_ids(root), seed=seed, **config["split"])
    model = build_model(dict(config["model"], pretrained=False))
    model.load_state_dict(torch.load("runs/{}/seed{}/model.pt".format(run, seed),
                                     map_location="cpu"))
    model.to(config["device"])
    items = []
    for area_id in test_ids:
        probabilities = sliding_window_probabilities(
            model, load_image(root, area_id), window=config["eval"]["window"],
            stride=config["eval"]["stride"], device=config["device"])
        items.append((probabilities.astype(np.float16), load_label(root, area_id),
                      load_label(root, area_id, eroded=True)))
    del model
    torch.cuda.empty_cache()
    return items


def score(items, offset):
    matrices = {"full": ConfusionMatrix(), "eroded": ConfusionMatrix()}
    for probabilities, label, eroded in items:
        prediction = decide(probabilities.astype(np.float32), offset)
        matrices["full"].update(prediction, label)
        matrices["eroded"].update(prediction, eroded)
    results = {}
    for gt, matrix in matrices.items():
        m = matrix.matrix
        results[gt] = dict(matrix.summary(),
                           car_precision=float(m[CAR, CAR] / m[:, CAR].sum()),
                           car_recall=float(m[CAR, CAR] / m[CAR].sum()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True, help="runs/<run>/seed<k>/model.pt")
    parser.add_argument("--dev-seed", type=int, default=99)
    parser.add_argument("--apply-seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()

    dev = test_probabilities(args.config, args.run, args.dev_seed)
    sweep = {float(o): score(dev, o) for o in OFFSETS}
    best = max(sweep, key=lambda o: sweep[o]["full"]["miou"])
    print("Dev split (seed {}) sweep, full ground truth:".format(args.dev_seed))
    print("  offset   mIoU   car IoU  car P  car R")
    for offset, r in sweep.items():
        f = r["full"]
        print("  {:+5.2f}   {:.3f}   {:.3f}   {:.3f}  {:.3f}{}".format(
            offset, f["miou"], f["per_class_iou"]["Car"], f["car_precision"],
            f["car_recall"], "   <- chosen" if offset == best else ""))

    applied = {}
    for seed in args.apply_seeds:
        items = test_probabilities(args.config, args.run, seed)
        applied[seed] = {"baseline": score(items, 0.0), "calibrated": score(items, best)}
        for gt in ("full", "eroded"):
            b, c = applied[seed]["baseline"][gt], applied[seed]["calibrated"][gt]
            print("seed {} [{}]: mIoU {:.3f} -> {:.3f}   car IoU {:.3f} -> {:.3f}   "
                  "car P {:.3f} -> {:.3f}   car R {:.3f} -> {:.3f}".format(
                      seed, gt, b["miou"], c["miou"], b["per_class_iou"]["Car"],
                      c["per_class_iou"]["Car"], b["car_precision"], c["car_precision"],
                      b["car_recall"], c["car_recall"]))

    out = os.path.join("runs", args.run, "car_calibration.json")
    with open(out, "w") as f:
        json.dump({"dev_seed": args.dev_seed, "chosen_offset": best,
                   "dev_sweep": sweep, "applied": applied}, f, indent=2)
    print("written", out)


if __name__ == "__main__":
    main()
