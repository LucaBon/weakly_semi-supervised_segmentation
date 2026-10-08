import numpy as np
import torch

from wsss.inference import sliding_window_probabilities
from wsss.metrics import BoundaryScore, boundary_map
from wsss.refine import PAMR, refine_probabilities


def test_pamr_keeps_a_valid_distribution():
    torch.manual_seed(0)
    image = torch.randn(1, 3, 40, 50)
    probabilities = torch.rand(1, 6, 40, 50).softmax(1)
    refined = PAMR(iterations=3, dilations=(1, 2))(image, probabilities)
    assert refined.shape == probabilities.shape
    torch.testing.assert_close(refined.sum(1), torch.ones(1, 40, 50))


def test_pamr_moves_a_blurry_border_onto_the_image_edge():
    # image: left half dark, right half bright; prediction border is 4 px too far right
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255
    probabilities = np.zeros((6, 32, 32), dtype=np.float32)
    probabilities[0, :, :20] = 0.9
    probabilities[1, :, :20] = 0.1
    probabilities[0, :, 20:] = 0.1
    probabilities[1, :, 20:] = 0.9
    refined = refine_probabilities(probabilities, image, iterations=20, dilations=(1, 2),
                                   device="cpu")
    before = (probabilities.argmax(0)[:, 16:20] == 1).mean()
    after = (refined.argmax(0)[:, 16:20] == 1).mean()
    assert after > before


def test_tiled_refinement_matches_the_untiled_one_away_from_tile_borders():
    rng = np.random.default_rng(0)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    probabilities = rng.random((6, 64, 64)).astype(np.float32)
    probabilities /= probabilities.sum(0)
    tiled = refine_probabilities(probabilities, image, iterations=2, dilations=(1,),
                                 tile=32, margin=8, device="cpu")
    whole = refine_probabilities(probabilities, image, iterations=2, dilations=(1,),
                                 tile=64, margin=0, device="cpu")
    np.testing.assert_allclose(tiled, whole, atol=1e-5)


def test_tta_matches_plain_inference_for_a_pointwise_model():
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 6, 1)
    image = (np.random.default_rng(0).random((70, 90, 3)) * 255).astype(np.uint8)
    plain = sliding_window_probabilities(model, image, window=64, stride=32, device="cpu", amp=False)
    tta = sliding_window_probabilities(model, image, window=64, stride=32, device="cpu",
                                       amp=False, tta=True)
    np.testing.assert_allclose(tta, plain, atol=1e-5)


def test_boundary_score():
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[:, 10:] = 1
    assert boundary_map(gt).sum() == 20
    perfect = BoundaryScore(tolerance=2, classes=[1])
    perfect.update(gt, gt)
    assert perfect.summary()["all"]["f1"] == 1.0
    shifted = np.zeros_like(gt)
    shifted[:, 11:] = 1  # border 1 px to the right
    near = BoundaryScore(tolerance=2)
    near.update(shifted, gt)
    assert near.summary()["all"]["f1"] == 1.0
    far = np.zeros_like(gt)
    far[:, 16:] = 1
    wrong = BoundaryScore(tolerance=2)
    wrong.update(far, gt)
    assert wrong.summary()["all"]["f1"] == 0.0
