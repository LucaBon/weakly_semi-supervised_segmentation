import copy

import yaml

DEFAULTS = {
    "name": "experiment",
    "data_root": "data/vaihingen",
    "output_dir": "runs",
    "device": "cuda",
    "seed": 0,
    "split": {"n_pixel": 3, "n_image": 23},
    "model": {"architecture": "unet", "encoder": "resnet50", "pretrained": True},
    # supervised | upper_bound | tags | self_train | unimatch
    "method": "supervised",
    "train": {
        "iters": 10000,
        "batch_size": 8,
        "unlabeled_batch_size": 8,
        "crop_size": 256,
        "optimizer": "adamw",
        "lr": 2.0e-4,
        "encoder_lr_mult": 0.5,
        "weight_decay": 1.0e-4,
        "car_probability": 0.3,
        "class_weights": True,
        "dice_weight": 1.0,
        "amp": True,
        "num_workers": 4,
        "log_every": 100,
    },
    "tags": {"cell_size": 200, "context": 28, "min_fraction": 0.0,
             "weight": 0.5, "r": 20.0},
    "pseudo": {"teacher": None, "threshold": 0.9, "car_threshold": 0.7,
               "finetune_iters": 1000, "finetune_lr_mult": 0.1},
    "unimatch": {"threshold": 0.95, "car_threshold": 0.8, "weight": 1.0,
                 "ema_decay": 0.0},
    "eval": {"window": 512, "stride": 256, "filter_threshold": None},
}


def deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, seed=None, overrides=None):
    """
    Defaults <- YAML file <- explicit overrides.
    Args:
        path (str): YAML config path
        seed (int): split/training seed override
        overrides (dict): nested overrides

    Returns:
        dict: config
    """
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            deep_update(config, yaml.safe_load(f) or {})
    if overrides:
        deep_update(config, overrides)
    if seed is not None:
        config["seed"] = seed
    return config
