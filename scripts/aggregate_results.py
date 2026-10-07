"""
Mean +- std over seeds of every experiment in a runs folder, as a Markdown table.

Usage:
    python scripts/aggregate_results.py runs [--gt eroded]
"""
import argparse
import glob
import json
import os

import numpy as np

from wsss.constants import EVAL_CLASSES, LABEL_NAMES


def mean_std(values):
    return "{:.3f} ± {:.3f}".format(np.mean(values), np.std(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("--gt", default="full", choices=["full", "eroded"])
    args = parser.parse_args()

    classes = [LABEL_NAMES[c] for c in EVAL_CLASSES]
    print("| Experiment | seeds | mIoU | mF1 | OA | " + " | ".join(classes) + " |")
    print("|---" * (5 + len(classes)) + "|")
    for experiment in sorted(os.listdir(args.runs)):
        files = sorted(glob.glob(os.path.join(args.runs, experiment, "seed*", "results.json")))
        if not files:
            continue
        summaries = [json.load(open(f)) for f in files]
        for key, suffix in (("test", ""), ("test_filtered", " + filtering")):
            rows = [s[key][args.gt] for s in summaries if key in s and args.gt in s[key]]
            if not rows:
                continue
            cells = [mean_std([r[m] for r in rows]) for m in ("miou", "mf1", "oa")]
            cells += [mean_std([r["per_class_iou"][c] for r in rows]) for c in classes]
            print("| {}{} | {} | {} |".format(experiment, suffix, len(rows), " | ".join(cells)))


if __name__ == "__main__":
    main()
