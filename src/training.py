import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import cross_entropy_2d, \
    per_label_accuracy, \
    per_label_precision, \
    per_label_recall, \
    calculate_iou, \
    calculate_accuracy, \
    calculate_f1_scores

from testing import evaluate_test_batch

from constants import PIXEL_WEIGHTS, \
    IMAGE_WEIGHTS, \
    MULTI_LABEL_THRESHOLD, \
    LABEL_NAMES


def train_with_pixel_labels(net,
                            optimizer,
                            train_loader,
                            epochs,
                            test_loader,
                            saved_model_path,
                            load_pretrained_path=None,
                            scheduler=None,
                            pixel_weights=PIXEL_WEIGHTS,
                            save_epoch=1):
    if load_pretrained_path is not None:
        weights = torch.load(load_pretrained_path)
        net.load_state_dict(weights)

    losses = []
    moving_average_losses = []
    pixel_weights = pixel_weights.cuda()

    iter_ = 0

    for e in range(1, epochs + 1):

        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output_seg, output_class = net(data)
            loss = cross_entropy_2d(output_seg, target, weight=pixel_weights)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            # moving average calculated over 100 iterations
            moving_average_losses.append(
                np.mean(losses[max(0, iter_ - 100):iter_]))

            if iter_ % 100 == 0:
                pred = np.argmax(output_seg.data.cpu().numpy(), axis=1)
                gt = target.data.cpu().numpy()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      '\tIoU: {}'
                      ''.format(e,
                                epochs,
                                batch_idx,
                                len(train_loader),
                                100. * batch_idx / len(train_loader),
                                loss.item(),
                                calculate_iou(pred,
                                              gt,
                                              label_values=LABEL_NAMES)))
                with torch.no_grad():
                    data_test_batch, target_test_batch = next(
                        iter(test_loader))
                    test_batch_loss, test_batch_ious = \
                        evaluate_test_batch(net,
                                            data_test_batch,
                                            target_test_batch)
                    print('Test on batch\tLoss: {:.6f}\tIoU: {}'
                          ''.format(test_batch_loss.item(), test_batch_ious))

            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            loss_test_list = []
            ious_test_list = []
            with torch.no_grad():
                for (test_data, test_target) in test_loader:
                    loss_test, ious = evaluate_test_batch(net,
                                                          test_data,
                                                          test_target)
                    loss_test_list.append(loss_test.item())
                    ious_test_list.append(ious)
            mean_ious_test = np.nanmean(ious_test_list, axis=0)
            mean_loss_test = np.mean(loss_test_list)
            print('Test [Number of tiles in test set: {}]\tLoss: {:.6f}'
                  '\tPer class mean IoU: {}\tMean IoU: {:.3f}'.format(
                                     len(test_loader.dataset.tiles),
                                     mean_loss_test,
                                     mean_ious_test,
                                     np.mean(mean_ious_test)))

            torch.save(net.state_dict(),
                       './EncDecUnpool_pixel_labels_epoch{}_loss{}'
                       ''.format(e,
                                 mean_loss_test))
    torch.save(net.state_dict(), saved_model_path)


def train_with_image_labels(net,
                            optimizer,
                            train_loader,
                            epochs,
                            test_loader,
                            saved_model_path,
                            load_pretrained_path=None,
                            scheduler=None,
                            image_weights=IMAGE_WEIGHTS,
                            save_epoch=1,
                            multi_label_threshold=MULTI_LABEL_THRESHOLD):
    if load_pretrained_path is not None:
        weights = torch.load(load_pretrained_path)
        net.load_state_dict(weights)

    losses = []
    moving_average_losses = []

    image_weights = image_weights.cuda()

    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output_seg, output_class = net(data)
            # for class car a weight 10 is used to increase recall
            # (TP / TP + FN)
            loss = nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=image_weights)(output_class, target)

            loss.sum().backward()
            optimizer.step()

            losses.append(loss.sum().item())
            # moving average calculated over 100 iterations
            moving_average_losses.append(np.mean(losses[max(0, iter_ - 100):iter_]))

            if iter_ % 100 == 0:
                pred = output_class.data.cpu().numpy()
                pred = np.where(pred > multi_label_threshold, 1, 0)
                gt = target.data.cpu().numpy()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                      '\tPer label accuracy: {}\tPer label precision: {}'
                      '\tPer label recall: {}'
                      ''.format(e,
                                epochs,
                                batch_idx,
                                len(train_loader),
                                100. * batch_idx / len(train_loader),
                                loss.mean().item(),
                                per_label_accuracy(pred, gt),
                                per_label_precision(pred, gt),
                                per_label_recall(pred, gt)))

            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            loss_test_list = []
            ious_test_list = []
            with torch.no_grad():
                for (data, target) in test_loader:
                    loss_test, ious = evaluate_test_batch(net=net,
                                                          data_test_batch=data,
                                                          target_test_batch=target)
                    loss_test_list.append(loss_test.item())
                    ious_test_list.append(ious)
            mean_ious_test = np.nanmean(ious_test_list, axis=0)
            mean_loss_test = np.mean(loss_test_list)
            print('Test [Number of tiles in test set: {}]\tLoss: {:.6f}'
                  '\tPer class mean IoU: {}\tMean IoU: {:.3f}'
                  ''.format(len(test_loader.dataset.tiles),
                            mean_loss_test,
                            mean_ious_test,
                            np.mean(mean_ious_test)))
            torch.save(net.state_dict(),
                       './EncDecUnpool_image_labels_epoch{}_loss{}'
                       ''.format(e, mean_loss_test))
    torch.save(net.state_dict(), saved_model_path)
