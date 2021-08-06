import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import INPUT_CHANNELS,\
    N_CLASSES, \
    BG_SCORE, \
    PAMR_ITER, \
    PAMR_KERNEL, \
    SG_PSI, \
    FOCAL_P, \
    FOCAL_LAMBDA

from utils import rescale_as, focal_loss, pseudo_gtmask, balanced_mask_loss_ce
from aspp import ASPP
from pamr import PAMR
from sg import StochasticGate
from gci import GCI
from resnet38 import ResNet38


class EncDecUnpoolNet(nn.Module):
    """
    EncDecUnpool network based on VGG16. It is inspired by Deconvnet
    "Learning Deconvolution Network for Semantic Segmentation", H. Noh et al.

    The network returns two outputs: one for the pixel-wise classification and
    one for the multiclass image classification
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)

    def __init__(self,
                 in_channels=INPUT_CHANNELS,
                 pixel_out_channels=N_CLASSES,
                 ):
        super(EncDecUnpoolNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, pixel_out_channels, 3, padding=1)

        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.apply(self.weight_init)

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        size1 = x.size()

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        size2 = x.size()

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        size3 = x.size()

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        size4 = x.size()

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5, output_size=size4)
        x = self.dropout_1(x)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4, output_size=size3)
        x = self.dropout_2(x)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3, output_size=size2)
        x = self.dropout_3(x)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2, output_size=size1)
        x = self.dropout_4(x)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.dropout_5(x)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D(x)

        x = F.log_softmax(x, dim=1)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaselineCAM(ResNet38):

    def __init__(self,
                 num_classes=5,
                 dropout=True):
        super().__init__()

        self.fc8 = nn.Conv2d(self.fan_out(), num_classes, 1,
                             bias=False)
        nn.init.xavier_uniform_(self.fc8.weight)

        cls_modules = [nn.AdaptiveAvgPool2d((1, 1)), self.fc8, Flatten()]
        if dropout:
            cls_modules.insert(0, nn.Dropout2d(0.5))

        self.cls_branch = nn.Sequential(*cls_modules)
        self.mask_branch = nn.Sequential(self.fc8, nn.ReLU())

        self.from_scratch_layers = [self.fc8]

        self._mask_logits = None

        self._fix_running_stats(self,
                                fix_params=True)  # freeze backbone BNs

    def forward_backbone(self, x):
        self._mask_logits = super().forward(x)
        return self._mask_logits

    def forward_cls(self, x):
        return self.cls_branch(x)

    def forward_mask(self, x, size):
        logits = self.fc8(x)
        masks = F.interpolate(logits,
                              size=size,
                              mode='bilinear',
                              align_corners=True)
        masks = F.relu(masks)

        # CAMs are unbounded
        # so let's normalised it first
        # (see jiwoon-ahn/psa)
        b, c, h, w = masks.size()
        masks_ = masks.view(b, c, -1)
        z, _ = masks_.max(-1, keepdim=True)
        masks_ /= (1e-5 + z)
        masks = masks.view(b, c, h, w)

        # bg = torch.ones_like(masks[:, :1])
        # masks = torch.cat([BG_SCORE * bg, masks], 1)

        # note, that the masks contain the background as the first channel
        return logits, masks

    def forward(self, y, _, labels=None):
        test_mode = labels is None

        x = self.forward_backbone(y)

        cls = self.forward_cls(x)
        logits, masks = self.forward_mask(x, y.size()[-2:])

        if test_mode:
            return cls, masks

        # foreground stats
        b, c, h, w = masks.size()
        masks_ = masks.view(b, c, -1)
        masks_ = masks_[:, 1:]
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        # upscale the masks & clean
        masks = self._rescale_and_clean(masks, y, labels)

        return cls, cls_fg, {"cam": masks}, logits, None, None

    def _rescale_and_clean(masks, image, labels):
        masks = F.interpolate(masks,
                              size=image.size()[-2:],
                              mode='bilinear',
                              align_corners=True)
        masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        return masks


class SoftMaxAE(ResNet38):

    def __init__(self,
                 num_classes=5,
                 dropout=True):
        super().__init__()

        self.num_classes = num_classes
        self._fix_running_stats(self,
                                fix_params=True)  # freeze backbone BNs

        # Decoder
        self._init_aspp()
        self._init_decoder(num_classes)

        self._backbone = None
        self._mask_logits = None

    def _init_aspp(self):
        self.aspp = ASPP(self.fan_out(), 8, self.NormLayer)

        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, self.NormLayer):
                self.from_scratch_layers.append(m)

        self._fix_running_stats(self.aspp)  # freeze BN

    def _init_decoder(self, num_classes):

        self._aff = PAMR(PAMR_ITER, PAMR_KERNEL)

        def conv2d(*args, **kwargs):
            conv = nn.Conv2d(*args, **kwargs)
            self.from_scratch_layers.append(conv)
            torch.nn.init.kaiming_normal_(conv.weight)
            return conv

        def bnorm(*args, **kwargs):
            bn = self.NormLayer(*args, **kwargs)
            self.from_scratch_layers.append(bn)
            if not bn.weight is None:
                bn.weight.data.fill_(1)
                bn.bias.data.zero_()
            return bn

        # pre-processing for shallow features
        self.shallow_mask = GCI(self.NormLayer)
        self.from_scratch_layers += self.shallow_mask.from_scratch_layers

        # Stochastic Gate
        self.sg = StochasticGate()
        self.fc8_skip = nn.Sequential(conv2d(256, 48, 1, bias=False),
                                      bnorm(48), nn.ReLU())
        self.fc8_x = nn.Sequential(
            conv2d(304, 256, kernel_size=3, stride=1, padding=1,
                   bias=False),
            bnorm(256), nn.ReLU())

        # decoder
        self.last_conv = nn.Sequential(
            conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                   bias=False),
            bnorm(256), nn.ReLU(),
            nn.Dropout(0.5),
            conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                   bias=False),
            bnorm(256), nn.ReLU(),
            nn.Dropout(0.1),
            conv2d(256, num_classes, kernel_size=1, stride=1))

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear",
                           align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec

    def forward_backbone(self, x):
        self._backbone = super().forward_as_dict(x)
        return self._backbone['conv6']

    def forward(self, y, y_raw=None, labels=None):
        test_mode = y_raw is None and labels is None

        # 1. backbone pass
        x = self.forward_backbone(y)

        # 2. ASPP modules
        x = self.aspp(x)

        #
        # 3. merging deep and shallow features
        #

        # 3.1 skip connection for deep features
        x2_x = self.fc8_skip(self._backbone['conv3'])
        x_up = rescale_as(x, x2_x)
        x = self.fc8_x(torch.cat([x_up, x2_x], 1))

        # 3.2 deep feature context for shallow features
        x2 = self.shallow_mask(self._backbone['conv3'], x)

        # 3.3 stochastically merging the masks
        x = self.sg(x, x2, alpha_rate=SG_PSI)

        # 4. final convs to get the masks
        x = self.last_conv(x)

        #
        # 5. Finalising the masks and scores
        #

        # constant BG scores
        # bg = torch.ones_like(x[:, :1])
        # x = torch.cat([bg, x], 1)

        bs, c, h, w = x.size()

        masks = F.softmax(x, dim=1)

        # reshaping
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)

        # classification loss
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1),
                           p=FOCAL_P,
                           c=FOCAL_LAMBDA)

        # adding the losses together
        # cls = cls_1[:, 1:] + cls_2[:, 1:]
        cls = cls_1[:, :] + cls_2[:, :]

        if test_mode:
            # if in test mode, not mask
            # cleaning is performed
            return cls, rescale_as(masks, y)

        self._mask_logits = x

        # foreground stats
        # masks_ = masks_[:, 1:]
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        # mask refinement with PAMR
        masks_dec = self.run_pamr(y_raw, masks.detach())

        # upscale the masks & clean
        masks = self._rescale_and_clean(masks, y, labels)
        masks_dec = self._rescale_and_clean(masks_dec, y, labels)

        # create pseudo GT
        pseudo_gt = pseudo_gtmask(masks_dec).detach()
        loss_mask = balanced_mask_loss_ce(self._mask_logits,
                                          pseudo_gt,
                                          labels)

        return cls, cls_fg, {"cam": masks,
                             "dec": masks_dec}, self._mask_logits, pseudo_gt, loss_mask

    @staticmethod
    def _rescale_and_clean(masks, image, labels):
        """Rescale to fit the image size and remove any masks
        of labels that are not present"""
        masks = F.interpolate(masks, size=image.size()[-2:],
                              mode='bilinear', align_corners=True)
        # masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        masks[:, :] *= labels[:, :, None, None].type_as(masks)
        return masks
