import torch.optim as optim

from model import EncDecUnpoolNet
from pre_trained_vgg import download_vgg, load_vgg_weights
from data import \
    split_data, \
    load_train_pixel_ids, \
    load_train_image_ids, \
    load_test_ids
from training import train_with_pixel_labels, train_with_image_labels
from constants import EPOCHS, EPOCHS_IMAGE_LABELS, LR_IMAGE_LABELS


def run_training_with_pixel_labels(net,
                                   train_pixel_loader,
                                   test_loader,
                                   load_pretrained_path,
                                   base_lr = 0.001,
                                   epochs=EPOCHS,
                                   loading_vgg_pre_trained=False,
                                   saved_pixel_model_path='./EncDecUnpool_pixel_labels'
                                   ):
    if loading_vgg_pre_trained:
        net = load_vgg_pretrained(base_lr, net)

    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    train_with_pixel_labels(net=net,
                            optimizer=optimizer,
                            train_loader=train_pixel_loader,
                            load_pretrained_path=load_pretrained_path,
                            epochs=epochs,
                            test_loader=test_loader,
                            saved_model_path=saved_pixel_model_path)


def run_training_with_image_labels(net,
                                   train_image_loader,
                                   test_loader,
                                   base_lr,
                                   epochs=EPOCHS,
                                   loading_vgg_pre_trained=False,
                                   saved_image_model_path='./EncDecUnpool_image_final'):
    if loading_vgg_pre_trained:
        net = load_vgg_pretrained(base_lr, net)

    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    train_with_image_labels(net=net,
                            optimizer=optimizer,
                            train_loader=train_image_loader,
                            epochs=epochs,
                            test_loader=test_loader,
                            saved_model_path=saved_image_model_path
                            )


def load_vgg_pretrained(base_lr, net):
    """
    It loads vgg16 weights and set learning rate for encoder parameters to half
    the base learning rate
    Args:
        base_lr (flaot): base learning rate
        net: network

    Returns:

    """
    download_vgg()
    net = load_vgg_weights(net=net)
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params': [value], 'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2
            # (VGG-16 weights are used as initialization)
            params += [{'params': [value], 'lr': base_lr / 2}]
    return net


if __name__ == "__main__":
    net = EncDecUnpoolNet()

    train_pixel_ids, train_image_ids, test_ids = split_data()

    train_pixel_loader = load_train_pixel_ids(train_pixel_ids=train_pixel_ids,
                                              )
    train_image_loader = load_train_image_ids(train_image_ids=train_image_ids,
                                              batch_size=20)
    test_loader = load_test_ids(test_ids=test_ids)

    image_model_path = './EncDecUnpool_image_final'
    pixel_model_path = './EncDecUnpool_pixel_labels'

    run_training_with_image_labels(net=net,
                                   train_image_loader=train_image_loader,
                                   test_loader=test_loader,
                                   base_lr=LR_IMAGE_LABELS,
                                   epochs=EPOCHS_IMAGE_LABELS,
                                   loading_vgg_pre_trained=True,
                                   saved_image_model_path=image_model_path)
