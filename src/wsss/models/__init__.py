import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

from wsss.constants import N_CLASSES
from wsss.models.encdec_unpool import EncDecUnpoolNet, load_vgg_weights

SMP_ARCHITECTURES = {"unet": smp.Unet,
                     "deeplabv3plus": smp.DeepLabV3Plus,
                     "fpn": smp.FPN}


class SmpSegmenter(nn.Module):
    """
    Wrapper around a segmentation_models_pytorch model exposing the same
    interface as EncDecUnpoolNet: forward(x, perturb) -> logits, where
    `perturb` applies dropout to the encoder features (UniMatch feature
    perturbation).
    """

    def __init__(self, architecture, encoder_name, pretrained=True,
                 n_classes=N_CLASSES):
        super(SmpSegmenter, self).__init__()
        self.net = SMP_ARCHITECTURES[architecture](
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=n_classes)

    def encoder_parameters(self):
        return self.net.encoder.parameters()

    def decoder_parameters(self):
        return list(self.net.decoder.parameters()) + \
            list(self.net.segmentation_head.parameters())

    def forward(self, x, perturb=False):
        features = self.net.encoder(x)
        if perturb:
            features = [F.dropout2d(f, 0.5, training=True) for f in features]
        return self.net.segmentation_head(self.net.decoder(features))


def build_model(config):
    """
    Args:
        config (dict): `model` section of the experiment config

    Returns:
        nn.Module: segmentation network returning logits
    """
    if config["architecture"] == "encdec_unpool":
        net = EncDecUnpoolNet(dropout=config.get("dropout", 0.5))
        if config.get("pretrained", True):
            net = load_vgg_weights(net)
        return net
    return SmpSegmenter(config["architecture"], config["encoder"],
                        pretrained=config.get("pretrained", True))
