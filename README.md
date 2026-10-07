# Weakly semi-supervised segmentation

### Problem description

Pixel-level labels are expensive, but image-level class labels are much cheaper and more
plentiful. The goal is to use these cheap labels to improve accuracy on the pixel-level task.

Semantic segmentation of the ISPRS **Vaihingen** dataset: 33 aerial IRRG orthophotos
(about 1900×2500 px at 9 cm) with 6 classes:

1. Impervious surfaces - WHITE
2. Building - BLUE
3. Low vegetation - TURQUOISE
4. Tree - GREEN
5. Car - YELLOW
6. Clutter/background - RED (predicted, but excluded from tags and metrics)

Split (3 random split seeds, `wsss.data.split_data`):

* N1 = 3 images with pixel-level labels (10%)
* N2 = 23 images with **crop-level tags** only (70%): each 200×200 cell of a fixed grid is
  tagged with the classes it contains
* Test = 7 images (20%)

Tasks:

* Task (i): N1 pixel-level labels only
* Task (ii): N1 pixel-level labels + N2 crop-level tags

### Methods

| ID | Data | Method | Config |
|---|---|---|---|
| B0 | N1 | Original EncDecUnpool (VGG16-BN SegNet/DeconvNet-like), fixed | `configs/b0_encdec_unpool.yaml` |
| B1 | N1 | U-Net, ImageNet ResNet-50, CE + Dice, car-aware crop sampling | `configs/b1_unet_r50.yaml` |
| M1 | N1 + N2 tags | B1 + multi-label loss on LSE-pooled softmax probabilities of each tagged crop | `configs/m1_tags_unet_r50.yaml` |
| M2 | N1 + N2 tags | M1 + prediction filtering at inference (Bae et al. 2022) | M1 run, `test_filtered` |
| M3 | N1 + N2 tags | Offline self-training: M1 teacher pseudo-labels N2, restricted to each crop's tags, thresholded; then fine-tuning on N1 | `configs/m3_self_train_unet_r50.yaml` |
| M3-B1 | N1 + N2 tags | As M3, but the teacher is B1 (N1 only): the original Task (ii) idea | `configs/m3_b1_teacher_unet_r50.yaml` |
| M4 | N1 + N2 tags | UniMatch-style weak-to-strong consistency (2 strong views + feature perturbation) with tag-constrained pseudo-labels + tag loss | `configs/m4_unimatch_unet_r50.yaml` |
| UB | N1 + N2 pixels | Upper bound: B1 with the N2 ground truth revealed | `configs/ub_unet_r50.yaml` |

How the crop tags are used:
* **Tag loss (M1).** Per-pixel class probabilities are pooled per crop with Log-Sum-Exp
  pooling, which sits between average and max pooling. The pooled scores are trained with
  BCE against the tags. An absent class must have low probability everywhere in the crop;
  a present class, even a small car, must be confident somewhere.
* **Tag-constrained pseudo-labels (M3, M4).** Classes absent from the crop's tags are removed
  from the teacher's softmax, then low-confidence pixels are ignored. The car threshold is lower.
* **Prediction filtering (M2).** At test time no tags are available, so classes whose pooled
  score in a window is below 0.5 are suppressed.

### Evaluation protocol

* The **final checkpoint** is evaluated after a fixed iteration budget. No model selection on
  the test set.
* Whole test images, sliding window (512 px, stride 256), every pixel included.
* Metrics come from one confusion matrix accumulated over the test set: per-class IoU/F1,
  mIoU, mF1, OA, over the 5 classes with clutter excluded. They are computed on both the
  full and the **eroded** ground truth (ISPRS protocol).
* Reported as mean ± std over split seeds 0, 1, 2 (`scripts/aggregate_results.py`).

### Results

**Preliminary: split seed 0 only.** The final table will be mean ± std over seeds 0, 1, 2
(`scripts/run_all.sh`). Seed 0 split: N1 = areas 2, 29, 31; test = areas 1, 6, 8, 12, 24, 27, 28.
U-Net ResNet-50, 10k iterations, final checkpoint.

Bold = best method that uses only Task (ii) data (UB excluded).

Full ground truth:

| Experiment | mIoU | mF1 | OA | Impervious | Building | Low veg. | Tree | Car |
|---|---|---|---|---|---|---|---|---|
| B0 (original net, fixed) | 0.669 | 0.799 | 0.819 | 0.707 | 0.774 | 0.599 | 0.716 | 0.549 |
| B1 (N1 only) | 0.710 | 0.828 | 0.844 | 0.737 | 0.812 | 0.638 | 0.732 | **0.629** |
| M1 (tag loss) | 0.714 | 0.831 | 0.847 | 0.756 | 0.835 | 0.635 | 0.720 | 0.624 |
| M2 (M1 + filtering) | 0.715 | 0.831 | 0.847 | 0.758 | 0.835 | 0.636 | 0.720 | 0.625 |
| M3 (self-training, M1 teacher) | **0.726** | **0.839** | 0.855 | 0.768 | 0.847 | 0.657 | 0.731 | 0.624 |
| M3-B1 (self-training, B1 teacher) | 0.721 | 0.836 | 0.852 | 0.748 | 0.825 | **0.659** | **0.744** | 0.628 |
| M4 (UniMatch-style) | 0.725 | 0.838 | **0.856** | **0.772** | **0.856** | 0.648 | 0.730 | 0.621 |
| UB (N1 + N2 pixels) | 0.762 | 0.863 | 0.877 | 0.801 | 0.884 | 0.674 | 0.757 | 0.695 |

Eroded ground truth (ISPRS protocol):

| Experiment | mIoU | mF1 | OA | Impervious | Building | Low veg. | Tree | Car |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.716 | 0.832 | 0.849 | 0.756 | 0.805 | 0.644 | 0.758 | 0.617 |
| B1 | 0.764 | 0.865 | 0.874 | 0.786 | 0.844 | 0.686 | 0.778 | 0.726 |
| M1 | 0.769 | 0.868 | 0.878 | 0.811 | 0.869 | 0.682 | 0.765 | 0.717 |
| M2 | 0.770 | 0.868 | 0.878 | 0.813 | 0.869 | 0.683 | 0.765 | 0.720 |
| M3 | **0.781** | **0.875** | 0.887 | 0.823 | 0.881 | 0.706 | 0.778 | 0.715 |
| M3-B1 | 0.776 | 0.873 | 0.882 | 0.797 | 0.857 | **0.709** | **0.791** | **0.728** |
| M4 | **0.781** | **0.875** | **0.888** | **0.829** | **0.891** | 0.697 | 0.776 | 0.713 |
| UB | 0.823 | 0.902 | 0.908 | 0.857 | 0.919 | 0.726 | 0.805 | 0.809 |

Observations (one split, so differences below about 0.01 may be noise):
* **Architecture matters most at this label budget.** B0, the original network with the fixed
  VGG loading, reaches 0.669 mIoU (0.603 in v0.1). B1's U-Net with a ResNet-50 encoder adds
  +0.041.
* **Tag loss alone (M1) barely helps** (+0.004 mIoU), and prediction filtering (M2) adds at
  most 0.001 to any model. The tags carry little information: a class counts as present with a
  single pixel, so an average cell is tagged with 3.6 of the 5 classes, and 35% of cells are
  tagged "car".
* **Self-training gives the clearest gain.** The teacher's predicted masks on N2 are corrected
  with the class labels (absent classes removed), and low-confidence pixels are dropped.
  * With M1 as teacher (M3): +0.016 mIoU, recovering about 30% of the 0.052 gap between B1
    and the upper bound.
  * With B1 as teacher (M3-B1, the original idea): +0.011. It is best on low vegetation and
    trees, and it is the only Task (ii) method that does not lose ground on cars.
* **UniMatch-style training (M4) ties with M3** (0.725 against 0.726). It is best on impervious
  surfaces and buildings, but takes about 38 minutes against 16 for M3.
* **Cars are the main open problem.** No Task (ii) method improves them, and the upper bound
  gains most there (0.695 against 0.629). The tag loss slightly hurts cars in every model that
  uses it.

#### Pseudo-label quality on N2

`scripts/pseudo_label_quality.py` scores the N2 pseudo-labels of a teacher against the hidden
N2 ground truth (diagnostic only, never used for training). Seed 0, mIoU on the kept pixels:

| Teacher | Raw masks | + class-label correction | + confidence threshold | Both (used by M3) |
|---|---|---|---|---|
| B1 (N1 only) | 0.690 | 0.715 | 0.765 (85% kept) | 0.787 (86% kept) |
| M1 (N1 + tag loss) | 0.718 | 0.729 | 0.816 (81% kept) | 0.824 (82% kept) |

The class-label correction alone adds +0.025 (B1) and +0.011 (M1) at full coverage. For B1 it
gives the largest gain on cars (0.612 → 0.648). The confidence threshold adds the most
(+0.08 to +0.10).

#### Overfitting

The N1-only models are scored on their 3 training images and on the 23 N2 images, which are
fully held out for them (seed 0, final checkpoints):

| Model | N1 (train) | N2 (held-out) | Test |
|---|---|---|---|
| B0 EncDecUnpool | 0.819 (car 0.703) | 0.653 (car 0.539) | 0.669 |
| B1 U-Net R50 | 0.919 (car 0.850) | 0.690 (car 0.612) | 0.710 |

Both overfit the 3 training images. B1 has the larger gap (0.23 against 0.17) but still
generalises better, and cars overfit the most. These numbers do not show whether stopping
earlier would help: only final checkpoints are kept, and the test set is not tracked during
training. Learning curves on the dev split (seed 99) would answer that.

**About the previous results (v0.1, mIoU 0.531 / 0.603).** These numbers are not reliable,
for three reasons:
1. The VGG16-BN weights were matched to the model by key *position*. The checkpoint has no
   `num_batches_tracked` keys, so keys misaligned after the first layer. The 38 resulting
   size-mismatch errors were swallowed by a bare `except`, and only `conv1_1` received
   ImageNet weights.
2. The best epoch was selected on the test set.
3. mIoU was averaged over batches instead of computed from an accumulated confusion matrix.

### Setup

Requires an NVIDIA GPU (a 6 GB laptop GPU is enough: M4 peaks at about 2.3 GiB with batch 4+4
and AMP) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # Python 3.12, torch 2.x
uv run pytest           # unit + smoke tests on synthetic data
```

Docker: `docker build -t wsss .` then
`docker run --gpus all --rm -it -v "$PWD/data":/app/data -v "$PWD/runs":/app/runs wsss python -m wsss.train --config configs/b1_unet_r50.yaml`

### Data

1. Request the dataset from the
   [ISPRS 2D Semantic Labeling – Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
   page. Download the orthophotos (`top/`), the complete ground truth and the eroded complete
   ground truth.
2. Convert them to the layout used here (`data/vaihingen/{images,labels,labels_eroded}`):

```bash
uv run python scripts/prepare_vaihingen.py \
    --images <raw>/top \
    --labels <raw>/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE \
    --eroded <raw>/ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE \
    --out data/vaihingen
```

### Training and evaluation

```bash
uv run python -m wsss.train --config configs/b1_unet_r50.yaml --seed 0
# M3 needs the M1 model of the same seed as teacher
uv run python -m wsss.train --config configs/m1_tags_unet_r50.yaml --seed 0
uv run python -m wsss.train --config configs/m3_self_train_unet_r50.yaml --seed 0
# everything, 3 seeds, then the results tables
scripts/run_all.sh
# re-evaluate a checkpoint
uv run python -m wsss.evaluate --config configs/m1_tags_unet_r50.yaml \
    --checkpoint runs/m1_tags_unet_r50/seed0/model.pt --seed 0 --filter-threshold 0.5
```

Outputs are written to `runs/<name>/seed<k>/`: `model.pt`, `results.json` (config, split,
metrics and pseudo-label quality) and TensorBoard logs. Hyper-parameters should be tuned on a
separate dev split (`--seed 99`), never on the reported seeds.

### Code layout

```
src/wsss/
  config.py        defaults + YAML configs
  data.py          splits, random pixel crops (car-aware), fixed-grid tag cells
  augment.py       dihedral transforms, strong photometric views, CutMix
  models/          EncDecUnpool (VGG16-BN) and segmentation_models_pytorch wrappers
  losses.py        CE + Dice, class weights, LSE tag pooling and tag loss
  pseudo_label.py  tag-constrained, thresholded pseudo-labels
  inference.py     sliding-window prediction, prediction filtering
  metrics.py       accumulated confusion matrix (IoU, F1, OA, kappa)
  train.py         all methods
  evaluate.py      test-set evaluation (full + eroded GT)
```

### References

* Papandreou et al., *Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation*, ICCV 2015
* Pinheiro & Collobert, *From Image-level to Pixel-level Labeling with Convolutional Networks*, CVPR 2015 (LSE pooling)
* Ouali et al., *Semi-Supervised Semantic Segmentation with Cross-Consistency Training*, CVPR 2020
* Bae et al., *One Weird Trick to Improve Your Semi-Weakly Supervised Semantic Segmentation Model*, IJCAI 2022
* Yang et al., *Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation* (UniMatch), CVPR 2023
* Noh et al., *Learning Deconvolution Network for Semantic Segmentation*, ICCV 2015
* Wang et al., *UNetFormer*, ISPRS J. P&RS 2022 (fully supervised Vaihingen reference)
