from torch.autograd import Variable
import numpy as np

from utils import cross_entropy_2d, calculate_iou
from constants import LABEL_NAMES


def evaluate_test_batch(net, data_test_batch, target_test_batch):
    test_data, test_target = Variable(data_test_batch.cuda()), \
                             Variable(target_test_batch.cuda())
    # calculate outputs by running images through the network
    test_output_seg, test_output_class = net(test_data)
    loss_test = cross_entropy_2d(test_output_seg,
                                 test_target,
                                 )

    test_pred = np.argmax(test_output_seg.data.cpu().numpy(),
                          axis=1)
    test_gt = test_target.data.cpu().numpy()
    ious = calculate_iou(test_pred,
                         test_gt,
                         label_values=LABEL_NAMES)
    return loss_test, ious