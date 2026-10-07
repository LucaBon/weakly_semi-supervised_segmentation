"""
Figures for the README, from the results in runs/ and the trained checkpoints.

Usage:
    python scripts/make_figures.py --out figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from wsss.config import load_config
from wsss.constants import CAR, IGNORE_INDEX, LABEL_NAMES
from wsss.data import compute_grid_tags, list_area_ids, load_image, load_label, split_data
from wsss.inference import sliding_window_probabilities
from wsss.labels import convert_to_color
from wsss.models import build_model
from wsss.pseudo_label import class_thresholds, make_pseudo_labels, tag_masked_probabilities

SEEDS = (0, 1, 2)
ROOT = "data/vaihingen"
METHODS = [("B0", "b0_encdec_unpool", "test"),
           ("B1", "b1_unet_r50", "test"),
           ("M1", "m1_tags_unet_r50", "test"),
           ("M2", "m1_tags_unet_r50", "test_filtered"),
           ("M3-B1", "m3_b1_teacher_unet_r50", "test"),
           ("M3", "m3_self_train_unet_r50", "test"),
           ("M4", "m4_unimatch_unet_r50", "test"),
           ("UB", "ub_unet_r50", "test")]

# validated categorical slots (light mode) and text inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, NEUTRAL = "#e4e3df", "#fcfcfb", "#b9b8b3"
IGNORED = (70, 70, 70)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False,
})


def load_metric(run, key, metric="miou", gt="full"):
    values = []
    for seed in SEEDS:
        summary = json.load(open("runs/{}/seed{}/results.json".format(run, seed)))[key][gt]
        values.append(summary[metric] if metric in summary else summary["per_class_iou"][metric])
    return np.array(values)


def style_value_axis(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def miou_by_method(out):
    names = [m[0] for m in METHODS]
    values = np.array([load_metric(run, key) for _, run, key in METHODS])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(names))[::-1]
    colors = [NEUTRAL if n in ("B0", "UB") else SERIES[0] for n in names]
    ax.barh(y, values.mean(1), height=0.55, color=colors, zorder=2)
    markers = ["o", "s", "^"]
    for s, marker in enumerate(markers):
        ax.scatter(values[:, s], y, marker=marker, s=26, color=INK, zorder=3,
                   label="seed {}".format(SEEDS[s]), edgecolors=SURFACE, linewidths=1)
    for yi, mean in zip(y, values.mean(1)):
        ax.text(0.7795, yi, "{:.3f}".format(mean), va="center", ha="left", color=INK_2, fontsize=9)
    b1 = values[names.index("B1")].mean()
    ax.axvline(b1, color=INK_3, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(y, names)
    ax.set_xlim(0.66, 0.79)
    ax.text(0.7795, len(names) - 0.35, "mean", ha="left", color=INK_3, fontsize=8.5)
    ax.set_xlabel("test mIoU (full ground truth)")
    ax.set_title("mIoU per method: mean (bar) and each split seed (markers)", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, borderaxespad=0.2)
    style_value_axis(ax)
    fig.text(0.01, 0.01, "Blue: Task (ii) methods and B1.  Grey: B0 and the upper bound (UB).  "
             "Dashed line: B1 mean.", color=INK_3, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "miou_by_method.png"), dpi=150)
    plt.close(fig)


def gain_per_seed(out):
    rows = [m for m in METHODS if m[0] not in ("B1", "M2", "B0")]
    b1 = load_metric("b1_unet_r50", "test")
    gains = np.array([load_metric(run, key) - b1 for _, run, key in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(rows))
    width = 0.26
    for s in range(len(SEEDS)):
        ax.bar(x + (s - 1) * (width + 0.02), gains[:, s], width=width, color=SERIES[s],
               label="seed {}".format(SEEDS[s]), zorder=2)
    ax.set_xticks(x, ["{}\nmean {:+.3f}".format(r[0], m) for r, m in zip(rows, gains.mean(1))])
    ax.set_ylim(0, gains.max() + 0.012)
    ax.set_ylabel("mIoU gain over B1")
    ax.set_title("Gain over the N1-only baseline, per split seed", loc="left")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    style_value_axis(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gain_per_seed.png"), dpi=150)
    plt.close(fig)


def per_class_iou(out):
    rows = [("B1", "b1_unet_r50"), ("M3", "m3_self_train_unet_r50"),
            ("M4", "m4_unimatch_unet_r50"), ("UB", "ub_unet_r50")]
    classes = LABEL_NAMES[:5]
    values = np.array([[load_metric(run, "test", c).mean() for c in classes] for _, run in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(classes))
    width = 0.19
    for i, (name, _) in enumerate(rows):
        ax.bar(x + (i - 1.5) * (width + 0.015), values[i], width=width, color=SERIES[i],
               label=name, zorder=2)
    ax.set_xticks(x, ["Impervious", "Building", "Low veg.", "Tree", "Car"])
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("test IoU (mean of 3 seeds)")
    ax.set_title("Per-class IoU: baseline, best Task (ii) methods, upper bound", loc="left")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    style_value_axis(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_class_iou.png"), dpi=150)
    plt.close(fig)


def load_model(config_name, run, seed):
    config = load_config("configs/{}.yaml".format(config_name), seed=seed)
    model = build_model(dict(config["model"], pretrained=False))
    model.load_state_dict(torch.load("runs/{}/seed{}/model.pt".format(run, seed),
                                     map_location="cpu"))
    return model.cuda().eval()


def color_labels(labels):
    rgb = convert_to_color(labels)
    rgb[labels == IGNORE_INDEX] = IGNORED
    return rgb


def class_legend(fig, with_ignored=False):
    handles = [Patch(facecolor=np.array(c) / 255, edgecolor=INK_3, linewidth=0.5, label=n)
               for n, c in zip(LABEL_NAMES, [(255, 255, 255), (0, 0, 255), (0, 255, 255),
                                             (0, 255, 0), (255, 255, 0), (255, 0, 0)])]
    if with_ignored:
        handles.append(Patch(facecolor=np.array(IGNORED) / 255, label="ignored"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5)


def show_panels(fig, axes, panels):
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)


def pseudo_label_example(out, seed=0):
    """One N2 cell: the hidden ground truth, the teacher's mask, and the
    pseudo-label after the class-label correction and the confidence threshold."""
    config = load_config("configs/m3_b1_teacher_unet_r50.yaml", seed=seed)
    _, n2_ids, _ = split_data(list_area_ids(ROOT), seed=seed)
    teacher = load_model("b1_unet_r50", "b1_unet_r50", seed)
    thresholds = class_thresholds(config["pseudo"]["threshold"], config["pseudo"]["car_threshold"])
    best = None
    for area_id in n2_ids[:8]:
        image, label = load_image(ROOT, area_id), load_label(ROOT, area_id)
        probabilities = torch.from_numpy(sliding_window_probabilities(teacher, image))
        for _, y, x, h, w, tags in compute_grid_tags([label], 200):
            if h < 200 or w < 200 or tags[CAR] == 0:
                continue
            cell = probabilities[:, y:y + h, x:x + w][None]
            raw = cell.argmax(1)[0].numpy()
            corrected_p = tag_masked_probabilities(cell.clamp(min=1e-8).log(),
                                                   torch.from_numpy(tags)[None])
            corrected = corrected_p.argmax(1)[0].numpy()
            changed = (raw != corrected).mean()
            if best is None or changed > best[0]:
                final, _ = make_pseudo_labels(corrected_p, thresholds)
                best = (changed, image[y:y + h, x:x + w], label[y:y + h, x:x + w], raw,
                        corrected, final[0].numpy(), tags)
    _, image, gt, raw, corrected, final, tags = best
    present = ", ".join(LABEL_NAMES[c] for c in range(5) if tags[c])
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.3))
    show_panels(fig, axes, [
        ("N2 cell (IRRG)", image),
        ("Hidden ground truth", color_labels(gt)),
        ("1. Teacher prediction", color_labels(raw)),
        ("2. Absent classes removed", color_labels(corrected)),
        ("3. Unsure pixels ignored", color_labels(final))])
    fig.suptitle("Pseudo-label construction on one 200×200 N2 cell.  Tags: " + present,
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    class_legend(fig, with_ignored=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "pseudo_label_example.png"), dpi=150)
    plt.close(fig)


def qualitative(out, seed=2, size=600):
    """A test crop rich in cars: image, ground truth, B1, M3 and M4."""
    _, _, test_ids = split_data(list_area_ids(ROOT), seed=seed)
    area_id = max(test_ids, key=lambda i: (load_label(ROOT, i) == CAR).mean())
    image, label = load_image(ROOT, area_id), load_label(ROOT, area_id)
    cars = (label == CAR).astype(np.float32)
    step = size // 2
    _, y, x = max((cars[y:y + size, x:x + size].sum(), y, x)
                  for y in range(0, label.shape[0] - size, step)
                  for x in range(0, label.shape[1] - size, step))
    panels = [("Test image (IRRG), area {}".format(area_id), image[y:y + size, x:x + size]),
              ("Ground truth", color_labels(label[y:y + size, x:x + size]))]
    for name, config_name, run in [("B1 (N1 only)", "b1_unet_r50", "b1_unet_r50"),
                                   ("M3 (self-training)", "m3_self_train_unet_r50",
                                    "m3_self_train_unet_r50"),
                                   ("M4 (UniMatch-style)", "m4_unimatch_unet_r50",
                                    "m4_unimatch_unet_r50")]:
        model = load_model(config_name, run, seed)
        prediction = sliding_window_probabilities(model, image).argmax(0).astype(np.uint8)
        panels.append((name, color_labels(prediction[y:y + size, x:x + size])))
        del model
        torch.cuda.empty_cache()
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.3))
    show_panels(fig, axes, panels)
    fig.suptitle("Test predictions, split seed {}".format(seed), x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    class_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "qualitative.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    miou_by_method(args.out)
    gain_per_seed(args.out)
    per_class_iou(args.out)
    pseudo_label_example(args.out)
    qualitative(args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
