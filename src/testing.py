import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import calculate_iou
from constants import LABEL_NAMES


def evaluate_one_batch(net, data_loader):
    """
    It evaluates the loss and IoU for each class on a single batch
    Args:
        net (model): pytorch model
        data_loader (torch.utils.data.dataloader.DataLoader):

    Returns:

    """
    net.eval()
    with torch.no_grad():
        data_test_batch, target_test_batch = next(
            iter(data_loader))
        test_batch_loss, test_batch_ious = \
            evaluate_batch(net,
                           data_test_batch,
                           target_test_batch)
        print('Test on batch\tLoss: {:.6f}\tIoU: {}'
              ''.format(test_batch_loss.item(), test_batch_ious))
        print('\n')


def evaluate_batch(net, data_batch, target_batch):
    """

    Args:
        net (model):
        data_batch ():
        target_batch:

    Returns:

    """
    test_data, test_target = Variable(data_batch.cuda()), \
                             Variable(target_batch.cuda())
    # calculate outputs by running images through the network
    test_output_seg, test_output_class = net(test_data)
    loss_test = F.nll_loss(test_output_seg,
                           test_target
                           )

    test_pred = np.argmax(test_output_seg.data.cpu().numpy(),
                          axis=1)
    test_gt = test_target.data.cpu().numpy()
    ious = calculate_iou(test_pred,
                         test_gt,
                         label_values=LABEL_NAMES)
    return loss_test, ious

def evaluate_test_set(net, test_loader, load_pretrained_path=None):
    """

    Args:
        net (model): pytorch model
        test_loader (torch.utils.data.dataloader.DataLoader):
        load_pretrained_path (str): path to checkpoint to load

    Returns:
        tuple(float, list): mean loss, mean per class IoUs
    """
    if load_pretrained_path is not None:
        weights = torch.load(load_pretrained_path)
        net.load_state_dict(weights)
    loss_test_list = []
    ious_test_list = []
    net.eval()
    with torch.no_grad():
        for (test_data, test_target) in test_loader:
            loss_test, ious = evaluate_batch(net,
                                             test_data,
                                             test_target)
            loss_test_list.append(loss_test.item())
            ious_test_list.append(ious)
    # clutter is neglected
    mean_ious_test = np.nanmean(ious_test_list, axis=0)[:-1]
    mean_loss_test = np.mean(loss_test_list)
    print('\n')
    print('Test [Number of tiles in test set: {}]\tLoss: {:.6f}'
          '\tPer class mean IoU: {}\tMean IoU: {:.3f}'.format(
        len(test_loader.dataset.tiles),
        mean_loss_test,
        mean_ious_test,
        np.mean(mean_ious_test)))
    print('\n\n')
    return mean_loss_test, mean_ious_test