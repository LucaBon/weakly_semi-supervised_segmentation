import argparse
import json
import os

import torch

from wsss.config import load_config
from wsss.data import list_area_ids, load_image, load_label, split_data
from wsss.inference import predict
from wsss.metrics import ConfusionMatrix
from wsss.models import build_model


def evaluate(model, data_root, test_ids, window=512, stride=256,
             filter_threshold=None, filter_cell_size=200, device="cuda", amp=True):
    """
    Evaluate on whole test images (sliding window) against the full and the
    eroded ground truth (ISPRS protocol: boundary pixels ignored).
    Metrics come from a single confusion matrix accumulated over the test set.
    Returns:
        dict: {"full": summary, "eroded": summary}
    """
    matrices = {"full": ConfusionMatrix(), "eroded": ConfusionMatrix()}
    for area_id in test_ids:
        prediction = predict(model, load_image(data_root, area_id),
                             window=window, stride=stride,
                             filter_threshold=filter_threshold,
                             filter_cell_size=filter_cell_size,
                             device=device, amp=amp)
        matrices["full"].update(prediction, load_label(data_root, area_id))
        eroded_path = os.path.join(data_root, "labels_eroded",
                                   "area{}.png".format(area_id))
        if os.path.isfile(eroded_path):
            matrices["eroded"].update(prediction,
                                      load_label(data_root, area_id, eroded=True))
    results = {"full": matrices["full"].summary()}
    if matrices["eroded"].matrix.sum() > 0:
        results["eroded"] = matrices["eroded"].summary()
    return results


def format_results(results):
    lines = []
    for gt_type, summary in results.items():
        lines.append("[{}] mIoU {:.3f}  mF1 {:.3f}  OA {:.3f}".format(
            gt_type, summary["miou"], summary["mf1"], summary["oa"]))
        lines.append("    IoU: " + ", ".join("{} {:.3f}".format(k, v) for k, v
                                             in summary["per_class_iou"].items()))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=None, help="split seed")
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="prediction filtering threshold (Bae et al. 2022); "
                             "defaults to eval.filter_threshold of the config")
    args = parser.parse_args()
    config = load_config(args.config, seed=args.seed)
    _, _, test_ids = split_data(list_area_ids(config["data_root"]),
                                seed=config["seed"], **config["split"])
    model = build_model(dict(config["model"], pretrained=False))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(config["device"])
    eval_config = dict(config["eval"])
    if args.filter_threshold is not None:
        eval_config["filter_threshold"] = args.filter_threshold
    results = evaluate(model, config["data_root"], test_ids,
                       filter_cell_size=config["tags"]["cell_size"],
                       device=config["device"], amp=config["train"]["amp"],
                       **eval_config)
    print(format_results(results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
