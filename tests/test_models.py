import torch
import torchvision

from wsss.models.encdec_unpool import EncDecUnpoolNet, load_vgg_weights


def test_vgg_weights_are_fully_loaded():
    vgg = torchvision.models.vgg16_bn(weights=None)
    state_dict = vgg.state_dict()
    net = load_vgg_weights(EncDecUnpoolNet(), state_dict)
    torch.testing.assert_close(net.encoder[0][0][0].weight, state_dict["features.0.weight"])
    torch.testing.assert_close(net.encoder[0][0][1].running_var,
                               state_dict["features.1.running_var"])
    torch.testing.assert_close(net.encoder[4][2][0].weight, state_dict["features.40.weight"])
    torch.testing.assert_close(net.encoder[4][2][1].bias, state_dict["features.41.bias"])


def test_legacy_checkpoint_without_num_batches_tracked():
    """The original vgg16_bn-6c64b313 checkpoint has no num_batches_tracked keys:
    the old positional key matching misaligned on it."""
    state_dict = {k: v for k, v in torchvision.models.vgg16_bn(weights=None).state_dict().items()
                  if not k.endswith("num_batches_tracked")}
    net = load_vgg_weights(EncDecUnpoolNet(), state_dict)
    torch.testing.assert_close(net.encoder[2][1][0].weight, state_dict["features.17.weight"])


def test_encdec_unpool_output_shape_and_perturbation():
    net = EncDecUnpoolNet().eval()
    x = torch.randn(2, 3, 64, 96)
    assert net(x).shape == (2, 6, 64, 96)
    assert net(x, perturb=True).shape == (2, 6, 64, 96)
