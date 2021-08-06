import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import \
    per_label_accuracy, \
    per_label_precision, \
    per_label_recall, \
    calculate_iou, \
    denorm, \
    mask_to_tensor, \
    calculate_batch_image_labels

from testing import evaluate_test_set, evaluate_one_batch

from constants import PIXEL_WEIGHTS, \
    IMAGE_WEIGHTS, \
    MULTI_LABEL_THRESHOLD, \
    LABEL_NAMES, \
    PRETRAIN_EPOCHS, \
    THRESHOLD_IMAGE_LABELS


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
        net.load_state_dict(weights, strict=False)

    losses = []
    moving_average_losses = []
    pixel_weights = pixel_weights.cuda()

    iter_ = 0

    for e in range(1, epochs + 1):
        train_one_epoch_with_pixel_labels(e, epochs, iter_, losses, moving_average_losses, net, optimizer,
                                          pixel_weights, test_loader, train_loader)

        mean_loss_test, mean_ious_test = evaluate_test_set(net=net, test_loader=test_loader)
        if e % save_epoch == 0:
            torch.save(net.state_dict(),
                       './EncDecUnpool_pixel_labels_epoch{}_loss_{:.6f}'
                       ''.format(e,
                                 mean_loss_test))
        if scheduler is not None:
            scheduler.step()
    torch.save(net.state_dict(), saved_model_path)


def train_one_epoch_with_pixel_labels(e,
                                      epochs,
                                      iter_,
                                      losses,
                                      moving_average_losses,
                                      net,
                                      optimizer,
                                      pixel_weights,
                                      test_loader,
                                      train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output_seg = net(data)
        loss = F.nll_loss(output_seg, target, weight=pixel_weights)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        # moving average calculated over 100 iterations
        moving_average_losses.append(
            np.mean(losses[max(0, iter_ - 100):iter_]))

        if iter_ % 100 == 0:
            pred = np.argmax(output_seg.data.cpu().numpy(), axis=1)
            gt = target.data.cpu().numpy()
            ious = calculate_iou(pred,
                                 gt,
                                 label_values=LABEL_NAMES)
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  '\tIoU: {}'
                  ''.format(e,
                            epochs,
                            batch_idx,
                            len(train_loader),
                            100. * batch_idx / len(train_loader),
                            loss.item(),
                            ious
                            ))
            evaluate_one_batch(net, test_loader)
        iter_ += 1


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
        net.load_state_dict(weights, strict=False)

    losses = []
    moving_average_losses = []

    image_weights = image_weights.cuda()

    iter_ = 0
    net.cuda()

    for e in range(1, epochs + 1):
        train_one_epoch_with_image_labels(e,
                                          epochs,
                                          image_weights,
                                          iter_,
                                          losses,
                                          moving_average_losses,
                                          multi_label_threshold,
                                          net,
                                          optimizer,
                                          test_loader,
                                          train_loader)

        # mean_loss_test, mean_ious_test = \
        #     evaluate_test_set(net=net,
        #                       test_loader=test_loader)
        if e % save_epoch == 0:
            torch.save(net.state_dict(),
                       './ResNet_image_labels_epoch{}'
                       ''.format(e))
        if scheduler is not None:
            scheduler.step()
    torch.save(net.state_dict(), saved_model_path)


def train_one_epoch_with_image_labels(e,
                                      epochs,
                                      image_weights,
                                      iter_,
                                      losses,
                                      moving_average_losses,
                                      multi_label_threshold,
                                      net,
                                      optimizer,
                                      test_loader,
                                      train_loader):
    pretrain = e < PRETRAIN_EPOCHS
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        # denorm image
        data_raw = denorm(data)
        data, data_raw, target = data.cuda(), data_raw.cuda(), target.cuda()
        optimizer.zero_grad()
        output_class, cls_fg, masks, mask_logits, pseudo_gt, loss_mask = \
            net(data, data_raw, target)

        pseudo_gt_one_channel = mask_to_tensor(pseudo_gt).cpu().numpy()

        classification_loss = nn.MultiLabelSoftMarginLoss()(output_class,
                                                            target)
        classification_loss = classification_loss * image_weights
        classification_loss = classification_loss.mean()
        loss = classification_loss.clone()
        loss.cuda()
        all_losses = {"loss_cls": classification_loss.item(),
                      "loss_fg": cls_fg.mean().item()}

        if "dec" in masks:
            loss_mask = loss_mask.mean()

            if not pretrain:
                loss += loss_mask

            assert not "pseudo" in masks
            masks["pseudo"] = pseudo_gt
            all_losses["loss_mask"] = loss_mask.item()

        loss.backward()
        optimizer.step()

        for mask_key, mask_val in masks.items():
            masks[mask_key] = masks[mask_key].detach()

        mask_logits = mask_logits.detach()

        losses.append(all_losses["loss_cls"])
        # moving average calculated over 100 iterations
        moving_average_losses.append(np.mean(losses[max(0, iter_ - 100):iter_]))

        if iter_ % 1 == 0:
            pred = calculate_batch_image_labels(pseudo_gt_one_channel,
                                                THRESHOLD_IMAGE_LABELS)
            print("PRED: ", pred)
            gt = target.data.cpu().numpy()
            print("GT:", gt)
            per_label_accuracy_result = per_label_accuracy(pred, gt)
            per_label_precision_result = per_label_precision(pred, gt)
            per_label_recall_result = per_label_recall(pred, gt)
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  '\tPer label accuracy: {}\tPer label precision: {}'
                  '\tPer label recall: {}'
                  ''.format(e,
                            epochs,
                            batch_idx,
                            len(train_loader),
                            100. * batch_idx / len(train_loader),
                            classification_loss,
                            per_label_accuracy_result,
                            per_label_precision_result,
                            per_label_recall_result))

            # evaluate_one_batch(net, test_loader)

        iter_ += 1

