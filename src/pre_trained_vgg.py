import os
from urllib.request import URLopener

import torch

from constants import VGG16_URL


def download_vgg(vgg_url=VGG16_URL):
    # Download VGG-16 weights from PyTorch
    if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
        print("Downloading VGG16 weights...")
        URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')
        print("Download completed!")


def load_vgg_weights(net, vgg16_weights_path='./vgg16_bn-6c64b313.pth'):
    vgg16_weights = torch.load(vgg16_weights_path)
    mapped_weights = {}
    for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
    try:
        # strict False for allowing keys missing exception
        net.load_state_dict(mapped_weights, strict=False)
    except:
        # size mismatch errors
        pass
    return net
