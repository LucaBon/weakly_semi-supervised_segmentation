import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from wsss.constants import N_CLASSES

# (block, number of convs, out channels) of the VGG16 encoder
VGG16_BLOCKS = [(1, 2, 64), (2, 2, 128), (3, 3, 256), (4, 3, 512), (5, 3, 512)]


def conv_bn(in_channels, out_channels):
    # conv -> BN -> ReLU, the same order as VGG16-BN, so its weights are reusable
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class EncDecUnpoolNet(nn.Module):
    """
    EncDecUnpool network based on VGG16-BN. It is inspired by Deconvnet
    "Learning Deconvolution Network for Semantic Segmentation", H. Noh et al.
    and SegNet: the decoder upsamples with the max-pooling indices of the
    encoder. It returns pixel-wise class logits.
    """

    def __init__(self, in_channels=3, n_classes=N_CLASSES, dropout=0.5):
        super(EncDecUnpoolNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.encoder = nn.ModuleList()
        channels = in_channels
        for _, n_convs, out_channels in VGG16_BLOCKS:
            layers = []
            for _ in range(n_convs):
                layers.append(conv_bn(channels, out_channels))
                channels = out_channels
            self.encoder.append(nn.Sequential(*layers))

        # The decoder mirrors the encoder; the last conv of each block reduces
        # the channels to those of the next (shallower) block
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(VGG16_BLOCKS))):
            _, n_convs, out_channels = VGG16_BLOCKS[i]
            next_channels = VGG16_BLOCKS[i - 1][2] if i > 0 else 64
            layers = [nn.Dropout(dropout)]
            for j in range(n_convs):
                is_last = j == n_convs - 1
                if i == 0 and is_last:
                    layers.append(nn.Conv2d(out_channels, n_classes, 3, padding=1))
                else:
                    layers.append(conv_bn(out_channels,
                                          next_channels if is_last else out_channels))
            self.decoder.append(nn.Sequential(*layers))

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def forward(self, x, perturb=False):
        indices, sizes = [], []
        for block in self.encoder:
            x = block(x)
            sizes.append(x.size())
            x, index = self.pool(x)
            indices.append(index)
        if perturb:
            # feature perturbation (UniMatch): dropout on the bottleneck
            x = F.dropout2d(x, 0.5, training=True)
        for block, index, size in zip(self.decoder, reversed(indices), reversed(sizes)):
            x = self.unpool(x, index, output_size=size)
            x = block(x)
        return x


def vgg16_bn_key_mapping(net):
    """
    Map torchvision vgg16_bn `features.*` keys to EncDecUnpoolNet encoder keys
    by layer structure (not by position in the state dict).
    Returns:
        dict: vgg key prefix -> encoder key prefix
    """
    mapping = {}
    vgg_index = 0
    for block_index, (_, n_convs, _) in enumerate(VGG16_BLOCKS):
        for conv_index in range(n_convs):
            prefix = "encoder.{}.{}".format(block_index, conv_index)
            mapping["features.{}".format(vgg_index)] = prefix + ".0"  # conv
            mapping["features.{}".format(vgg_index + 1)] = prefix + ".1"  # bn
            vgg_index += 3  # conv, bn, relu
        vgg_index += 1  # max pooling
    return mapping


def load_vgg_weights(net, vgg_state_dict=None):
    """
    Initialize the encoder with ImageNet VGG16-BN weights. Every encoder tensor
    must be loaded, otherwise an error is raised (no silent partial loading).
    Args:
        net (EncDecUnpoolNet): network
        vgg_state_dict (dict): optional vgg16_bn state dict; downloaded from
            torchvision if None

    Returns:
        EncDecUnpoolNet: network with initialized encoder
    """
    if vgg_state_dict is None:
        weights = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
        vgg_state_dict = weights.get_state_dict(progress=True)
    mapping = vgg16_bn_key_mapping(net)
    mapped = {}
    for key, value in vgg_state_dict.items():
        prefix, _, name = key.rpartition(".")
        if prefix in mapping:
            mapped["{}.{}".format(mapping[prefix], name)] = value
    encoder_keys = {k for k in net.state_dict() if k.startswith("encoder.")
                    and not k.endswith("num_batches_tracked")}
    missing = encoder_keys - set(mapped)
    if missing:
        raise RuntimeError("VGG weights missing for: {}".format(sorted(missing)))
    net.load_state_dict(mapped, strict=False)
    return net
