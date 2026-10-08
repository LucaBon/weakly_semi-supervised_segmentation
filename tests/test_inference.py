import numpy as np
import torch

from wsss.data import to_tensor
from wsss.constants import CAR
from wsss.inference import decide, sliding_window_probabilities, window_origins


def test_window_origins_cover_everything():
    for length in (100, 128, 300, 1999):
        covered = np.zeros(length, dtype=bool)
        for origin in window_origins(length, 128, 96):
            covered[origin:origin + 128] = True
        assert covered.all()


def test_sliding_window_matches_full_image_for_pointwise_model():
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 6, 1)
    image = (np.random.default_rng(0).random((150, 233, 3)) * 255).astype(np.uint8)
    probabilities = sliding_window_probabilities(model, image, window=64, stride=40,
                                                 device="cpu", amp=False)
    expected = model(to_tensor(image)[None]).softmax(1)[0].detach().numpy()
    assert probabilities.shape == (6, 150, 233)
    np.testing.assert_allclose(probabilities, expected, atol=1e-5)


def test_sliding_window_on_image_smaller_than_window():
    model = torch.nn.Conv2d(3, 6, 1)
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    probabilities = sliding_window_probabilities(model, image, window=64, stride=32,
                                                 device="cpu", amp=False)
    assert probabilities.shape == (6, 40, 50)


def test_prediction_filtering_removes_absent_class():
    class Constant(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 6, *x.shape[2:])
            logits[:, 1] = 1.0  # weak building everywhere
            logits[:, 0] = 1.5
            return logits
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    probabilities = sliding_window_probabilities(Constant(), image, window=64, stride=64,
                                                 device="cpu", amp=False,
                                                 filter_threshold=0.5)
    # pooled building probability ~0.32 < 0.5: building is suppressed
    assert probabilities[1].max() == 0


def test_car_offset_trades_car_for_the_runner_up():
    probabilities = np.zeros((6, 1, 2), dtype=np.float32)
    probabilities[CAR, 0] = [0.55, 0.9]
    probabilities[0, 0] = [0.45, 0.1]
    assert decide(probabilities).tolist() == [[CAR, CAR]]
    # log(0.55) - 0.5 < log(0.45), but log(0.9) - 0.5 > log(0.1)
    assert decide(probabilities, car_offset=-0.5).tolist() == [[0, CAR]]
