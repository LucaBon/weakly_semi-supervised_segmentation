import json
import os

import sys

import pytest
import yaml

from wsss.config import load_config
from wsss.train import run


def smoke_config(root, tmp_path, method, **extra):
    overrides = {
        "name": method, "data_root": root, "output_dir": str(tmp_path / "runs"),
        "device": "cpu", "method": method, "split": {"n_pixel": 2, "n_image": 2},
        "model": {"architecture": "unet", "encoder": "resnet18", "pretrained": False},
        "train": {"iters": 2, "batch_size": 2, "unlabeled_batch_size": 2,
                  "crop_size": 128, "amp": False, "num_workers": 0, "log_every": 1},
        "tags": {"cell_size": 100, "context": 14},
        "eval": {"window": 128, "stride": 96, "filter_threshold": 0.5},
    }
    overrides.update(extra)
    return load_config(overrides=overrides)


@pytest.mark.parametrize("method", ["supervised", "upper_bound", "tags", "unimatch"])
def test_methods_run_end_to_end(synthetic_root, tmp_path, method):
    summary = run(smoke_config(synthetic_root, tmp_path, method))
    assert set(summary["test"]) == {"full", "eroded"}
    assert 0 <= summary["test"]["full"]["miou"] <= 1
    assert "test_filtered" in summary
    with open(os.path.join(tmp_path, "runs", method, "seed0", "results.json")) as f:
        assert json.load(f)["split"]["test"] == summary["split"]["test"]


def test_self_training_with_teacher(synthetic_root, tmp_path):
    run(smoke_config(synthetic_root, tmp_path, "tags"))
    teacher = str(tmp_path / "runs" / "tags" / "seed{seed}" / "model.pt")
    summary = run(smoke_config(synthetic_root, tmp_path, "self_train",
                               pseudo={"teacher": teacher, "finetune_iters": 1}))
    assert 0 <= summary["pseudo_labels"]["coverage"] <= 1


def test_evaluate_cli(synthetic_root, tmp_path, monkeypatch, capsys):
    from wsss import evaluate
    config = smoke_config(synthetic_root, tmp_path, "tags")
    run(config)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "runs" / "tags" / "seed0" / "model.pt"
    for extra in ([], ["--filter-threshold", "0.3"]):
        monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config_path),
                                          "--checkpoint", str(checkpoint)] + extra)
        evaluate.main()
        assert "mIoU" in capsys.readouterr().out
