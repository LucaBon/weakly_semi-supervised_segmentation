"""
Training entry point for every experiment:

    supervised   B0/B1: pixel labels of N1 only (Task i)
    upper_bound  UB: pixel labels of N1 + N2 (N2 ground truth revealed)
    tags         M1: N1 pixel labels + tag loss on N2 crops (Task ii)
    self_train   M3: N1 + tag-filtered pseudo-labels of N2 from a teacher
    unimatch     M4: N1 + online weak-to-strong consistency on N2 crops with
                 tag-constrained pseudo-labels + tag loss

Usage:
    python -m wsss.train --config configs/b1_unet_r50.yaml --seed 0
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from wsss.augment import apply_cutmix, cutmix_masks, strong_photometric
from wsss.config import load_config
from wsss.constants import IGNORE_INDEX
from wsss.data import (PixelCropDataset, TagCellDataset, compute_grid_tags,
                       infinite_loader, list_area_ids, load_image, load_label,
                       split_data)
from wsss.evaluate import evaluate, format_results
from wsss.inference import sliding_window_probabilities
from wsss.losses import (class_weights_from_labels, masked_cross_entropy,
                         segmentation_loss, tag_loss)
from wsss.metrics import ConfusionMatrix
from wsss.models import build_model
from wsss.pseudo_label import (class_thresholds, make_pseudo_labels,
                               tag_masked_probabilities)

UNLABELED_METHODS = ("tags", "self_train", "unimatch")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, train_config, lr_scale=1.0):
    """Two parameter groups: the pre-trained encoder is trained at
    lr * encoder_lr_mult, the decoder at the nominal lr."""
    lr = train_config["lr"] * lr_scale
    groups = [{"params": list(model.encoder_parameters()),
               "lr": lr * train_config["encoder_lr_mult"]},
              {"params": list(model.decoder_parameters()), "lr": lr}]
    if train_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(groups, weight_decay=train_config["weight_decay"])
    elif train_config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(groups, weight_decay=train_config["weight_decay"])
    else:
        optimizer = torch.optim.SGD(groups, momentum=0.9,
                                    weight_decay=train_config["weight_decay"])
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def poly_lr(optimizer, iteration, total, power=0.9):
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (1 - iteration / total) ** power


@torch.no_grad()
def ema_update(teacher, student, decay):
    for t, s in zip(teacher.state_dict().values(), student.state_dict().values()):
        if t.dtype.is_floating_point:
            t.mul_(decay).add_(s.detach(), alpha=1 - decay)
        else:
            t.copy_(s)


def generate_pseudo_labels(teacher, images, labels, cells, config):
    """
    Offline pseudo-labels for the N2 grid cells (M3). The teacher predicts the
    whole image with a sliding window; in each cell the classes absent from
    the tags are removed, and low-confidence pixels are ignored.
    The hidden N2 ground truth is used only to report pseudo-label quality.
    Returns:
        tuple(list, dict): one (cell, cell) uint8 map per cell, diagnostics
    """
    thresholds = class_thresholds(config["pseudo"]["threshold"],
                                  config["pseudo"]["car_threshold"])
    matrix = ConfusionMatrix()
    pseudo_labels = [None] * len(cells)
    kept, total = 0, 0
    for index, image in enumerate(images):
        probabilities = torch.from_numpy(sliding_window_probabilities(
            teacher, image, device=config["device"], amp=config["train"]["amp"],
            window=config["eval"]["window"], stride=config["eval"]["stride"]))
        for item, (cell_index, y, x, h, w, tags) in enumerate(cells):
            if cell_index != index:
                continue
            cell = probabilities[:, y:y + h, x:x + w]
            # masking renormalized probabilities == masking logits before softmax
            masked = tag_masked_probabilities(cell.clamp(min=1e-8).log()[None],
                                              torch.from_numpy(tags)[None])
            pseudo, _ = make_pseudo_labels(masked, thresholds)
            pseudo = pseudo[0].numpy().astype(np.uint8)
            pseudo_labels[item] = pseudo
            gt = labels[index][y:y + h, x:x + w]
            keep = pseudo != IGNORE_INDEX
            matrix.update(pseudo[keep], gt[keep])
            kept += keep.sum()
            total += keep.size
    diagnostics = {"coverage": float(kept / total), "quality": matrix.summary()}
    return pseudo_labels, diagnostics


def train_loop(model, config, labeled, unlabeled, iterations, mode,
               class_weights, writer, lr_scale=1.0, tag=""):
    """
    Args:
        model (nn.Module): network
        config (dict): experiment config
        labeled (iterator): batches (image, target) of pixel-labelled crops
        unlabeled (iterator): batches of TagCellDataset dicts, or None
        iterations (int): number of iterations
        mode (str): supervised | tags | self_train | unimatch
        class_weights (torch.Tensor): weights of the supervised CE
        writer (SummaryWriter): logger
        lr_scale (float): learning rate multiplier (fine-tuning)
        tag (str): prefix of the logged scalars
    """
    train_config = config["train"]
    device = config["device"]
    optimizer = build_optimizer(model, train_config, lr_scale)
    device_type = torch.device(device).type
    scaler = torch.amp.GradScaler(device_type, enabled=train_config["amp"])
    tag_weight = config["tags"]["weight"]
    unimatch_config = config["unimatch"]
    thresholds = class_thresholds(unimatch_config["threshold"],
                                  unimatch_config["car_threshold"]).to(device)
    teacher = None
    if mode == "unimatch" and unimatch_config["ema_decay"] > 0:
        teacher = copy.deepcopy(model)
        teacher.requires_grad_(False)

    start = time.time()
    for iteration in range(iterations):
        poly_lr(optimizer, iteration, iterations)
        model.train()
        image_x, target_x = (t.to(device, non_blocking=True) for t in next(labeled))
        losses = {}
        with torch.autocast(device_type=device_type, enabled=train_config["amp"]):
            if mode == "supervised":
                logits_x = model(image_x)
            else:
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in next(unlabeled).items()}
                image_u, valid, tags = batch["image"], batch["valid"], batch["tags"]
                logits = model(torch.cat([image_x, image_u]))
                logits_x, logits_u = logits.split([len(image_x), len(image_u)])
                if tag_weight > 0:
                    losses["tag"] = tag_weight * tag_loss(
                        logits_u, tags, valid, r=config["tags"]["r"])
                if mode == "self_train":
                    losses["pseudo"] = masked_cross_entropy(logits_u, batch["target"])
                elif mode == "unimatch":
                    losses.update(unimatch_losses(
                        model, model if teacher is None else teacher, image_u,
                        valid, tags, thresholds, unimatch_config["weight"]))
            losses["supervised"] = segmentation_loss(
                logits_x, target_x, class_weights, train_config["dice_weight"])
            loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if teacher is not None:
            ema_update(teacher, model, unimatch_config["ema_decay"])

        if iteration % train_config["log_every"] == 0 or iteration == iterations - 1:
            for name, value in losses.items():
                writer.add_scalar("{}loss/{}".format(tag, name), value.item(), iteration)
            print("{}iter {}/{}  loss {:.4f}  ({})  {:.0f}s".format(
                tag, iteration, iterations, loss.item(),
                ", ".join("{} {:.4f}".format(k, v.item()) for k, v in losses.items()),
                time.time() - start), flush=True)


def unimatch_losses(model, teacher, image_w, valid, tags, thresholds, weight):
    """
    UniMatch (Yang et al., CVPR 2023) unlabeled losses with tag constraints:
    pseudo-labels from the weak view (classes absent from the tags removed,
    low-confidence pixels ignored) supervise two strong views (photometric +
    CutMix) and a feature-perturbed view of the weak image.
    """
    with torch.no_grad():
        teacher.eval()
        logits_w = teacher(image_w)
        model.train()
        probabilities = tag_masked_probabilities(logits_w, tags)
        pseudo, _ = make_pseudo_labels(probabilities, thresholds, valid)
        batch, _, height, width = image_w.shape
        permutation = torch.randperm(batch, device=image_w.device)
        strong_images, strong_targets = [], []
        for _ in range(2):
            masks = cutmix_masks(batch, height, width, image_w.device)
            strong_images.append(apply_cutmix(strong_photometric(image_w), masks,
                                              permutation))
            strong_targets.append(torch.where(masks, pseudo[permutation], pseudo))
    logits_strong = model(torch.cat(strong_images))
    logits_fp = model(image_w, perturb=True)
    loss_strong = masked_cross_entropy(logits_strong, torch.cat(strong_targets))
    loss_fp = masked_cross_entropy(logits_fp, pseudo)
    return {"strong": weight * 0.5 * loss_strong,
            "feature_perturbation": weight * 0.5 * loss_fp}


def run(config):
    set_seed(config["seed"])
    device = config["device"]
    method = config["method"]
    out_dir = os.path.join(config["output_dir"], config["name"],
                           "seed{}".format(config["seed"]))
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir)

    data_root = config["data_root"]
    pixel_ids, image_ids, test_ids = split_data(list_area_ids(data_root),
                                                seed=config["seed"], **config["split"])
    print("N1 (pixel labels):", pixel_ids)
    print("N2 (crop tags):", image_ids)
    print("Test:", test_ids)
    n1_images = [load_image(data_root, i) for i in pixel_ids]
    n1_labels = [load_label(data_root, i) for i in pixel_ids]
    n2_images, n2_labels = [], []
    if method in UNLABELED_METHODS + ("upper_bound",):
        n2_images = [load_image(data_root, i) for i in image_ids]
        n2_labels = [load_label(data_root, i) for i in image_ids]

    train_config = config["train"]
    supervised_images, supervised_labels = n1_images, n1_labels
    if method == "upper_bound":
        supervised_images, supervised_labels = n1_images + n2_images, n1_labels + n2_labels
    class_weights = None
    if train_config["class_weights"]:
        class_weights = class_weights_from_labels(supervised_labels).to(device)
        print("Class weights:", class_weights.tolist())
    labeled = infinite_loader(
        PixelCropDataset(supervised_images, supervised_labels,
                         crop_size=train_config["crop_size"],
                         car_probability=train_config["car_probability"]),
        train_config["batch_size"], train_config["num_workers"], config["seed"])

    model = build_model(config["model"]).to(device)
    summary = {"config": config, "split": {"n1": pixel_ids, "n2": image_ids,
                                           "test": test_ids}}
    unlabeled = None
    if method in UNLABELED_METHODS:
        tags_config = config["tags"]
        cells = compute_grid_tags(n2_labels, tags_config["cell_size"],
                                  tags_config["min_fraction"])
        pseudo_labels = None
        if method == "self_train":
            teacher = build_model(dict(config["model"], pretrained=False)).to(device)
            teacher_path = config["pseudo"]["teacher"].format(seed=config["seed"])
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            pseudo_labels, diagnostics = generate_pseudo_labels(
                teacher, n2_images, n2_labels, cells, config)
            summary["pseudo_labels"] = diagnostics
            print("Pseudo-labels: coverage {:.3f}, mIoU on kept pixels {:.3f}".format(
                diagnostics["coverage"], diagnostics["quality"]["miou"]))
            del teacher
        unlabeled = infinite_loader(
            TagCellDataset(n2_images, cells, tags_config["cell_size"],
                           tags_config["context"], pseudo_labels=pseudo_labels),
            train_config["unlabeled_batch_size"], train_config["num_workers"],
            config["seed"] + 1)

    train_loop(model, config, labeled, unlabeled, train_config["iters"],
               "supervised" if method == "upper_bound" else method,
               class_weights, writer)
    if method == "self_train" and config["pseudo"]["finetune_iters"] > 0:
        # refine on the clean N1 pixel labels only
        train_loop(model, config, labeled, None, config["pseudo"]["finetune_iters"],
                   "supervised", class_weights, writer,
                   lr_scale=config["pseudo"]["finetune_lr_mult"], tag="finetune/")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    # The final checkpoint is evaluated: no model selection on the test set
    eval_config = dict(config["eval"])
    filter_threshold = eval_config.pop("filter_threshold")
    eval_config["filter_cell_size"] = config["tags"]["cell_size"]
    summary["test"] = evaluate(model, data_root, test_ids, device=device,
                               amp=train_config["amp"], **eval_config)
    print(format_results(summary["test"]))
    if filter_threshold is not None:
        # M2: prediction filtering with the pooled tag scores
        summary["test_filtered"] = evaluate(model, data_root, test_ids,
                                            filter_threshold=filter_threshold,
                                            device=device, amp=train_config["amp"],
                                            **eval_config)
        print("With prediction filtering:")
        print(format_results(summary["test_filtered"]))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    writer.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None,
                        help="override train.iters (e.g. for smoke tests)")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    overrides = {}
    if args.iters is not None:
        overrides["train"] = {"iters": args.iters}
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    run(load_config(args.config, seed=args.seed, overrides=overrides))


if __name__ == "__main__":
    main()
