import math
from typing import List
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet101

from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss

from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets
import matplotlib.pyplot as plt


class LaneATT(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchor_feat_channels=64,
                 topk_anchors=None,
                 anchors_freq_path=None):
        super(LaneATT, self).__init__()
        # Some definitions
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w, self.img_h = img_w, img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = int(math.ceil(img_h / self.stride))
        self.fmap_w = int(math.ceil(img_w / self.stride))
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)  # we want to sample from bottom to top in image.
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.anchor_angles = [15., 22., 30., 39., 49., 60., 72., 80., 90., 100, 108., 120., 131., 141., 150., 158., 165.]

        self.n_angles = len(self.anchor_angles)  # 17
        self.n_anchor_pos = self.fmap_w + 2 * (self.fmap_h-1)

        # Generate anchors Anchor vec = [p(pos), p(neg), y_o, x_o, L, d1, ..., dS]
        self.n_anchor_properties = 2 + 2 + 1 + self.n_offsets

        # n_angles, n_anchor_properties, height, width
        self.anchors = self.generate_anchors(self.anchor_angles, self.fmap_h, self.fmap_w)
        
        # Now keep another anchor tensor that holds only the stacked anchors from the sides and bottom
        # n_angles, n_anchor_properties, 2*height + width
        self.edge_anchors = torch.cat([self.anchors[:, :, :-1, 0], self.anchors[:, :, :-1, -1], self.anchors[:, :, -1, :]], -1)

        # n_angles * (2 * (height-1) + width), n_anchor_properties
        self.anchors_flat = self.edge_anchors.permute([0, 2, 1]).reshape([-1, self.n_anchor_properties])

        # Setup and initialize layers
        self.conv_bottom = nn.Conv2d(in_channels=backbone_nb_channels, out_channels=self.anchor_feat_channels, kernel_size=1)
        self.head_conv_bottom = nn.Conv2d(in_channels=self.anchor_feat_channels, out_channels=self.n_angles * (self.n_anchor_properties - 2), kernel_size=1, bias=False)

        self.conv_left = nn.Conv2d(in_channels=backbone_nb_channels, out_channels=self.anchor_feat_channels, kernel_size=1)
        self.head_conv_left = nn.Conv2d(in_channels=self.anchor_feat_channels, out_channels=self.n_angles * (self.n_anchor_properties - 2), kernel_size=1, bias=False)

        self.conv_right = nn.Conv2d(in_channels=backbone_nb_channels, out_channels=self.anchor_feat_channels, kernel_size=1)
        self.head_conv_right = nn.Conv2d(in_channels=self.anchor_feat_channels, out_channels=self.n_angles * (self.n_anchor_properties - 2), kernel_size=1, bias=False)

        self.initialize_layer(self.conv_bottom)
        self.initialize_layer(self.head_conv_bottom)
        self.initialize_layer(self.conv_left)
        self.initialize_layer(self.head_conv_left)
        self.initialize_layer(self.conv_right)
        self.initialize_layer(self.head_conv_right)

    def generate_anchors(self, angles, fmap_h, fmap_w):
        anchors = torch.zeros((self.n_angles, self.n_anchor_properties, fmap_h, fmap_w))  # n_angles, n_anchor_properties, height, width

        left_anchors = self.generate_side_anchors(angles, x=0., n_origins=fmap_h)  # n_angles, n_anchor_properties, height
        right_anchors = self.generate_side_anchors(angles, x=1., n_origins=fmap_h)  # n_angles, n_anchor_properties, height
        bottom_anchors = self.generate_side_anchors(angles, y=1., n_origins=fmap_w)  # n_angles, n_anchor_properties, width

        anchors[:, :, :-1, 0] = left_anchors[..., :-1]
        anchors[:, :, :-1, -1] = right_anchors[..., :-1]
        anchors[:, :, -1, :] = bottom_anchors

        return anchors  # n_angles, n_anchor_properties, height, width

    def generate_side_anchors(self, angles, n_origins, x=None, y=None):
        if x is None and y is not None:
            origins = [(x, y) for x in np.linspace(0., 1., num=n_origins)]
        elif x is not None and y is None:
            origins = [(x, y) for y in np.linspace(0., 1., num=n_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 length, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((self.n_angles, self.n_anchor_properties, n_origins))
        for j, origin in enumerate(origins):
            for i, angle in enumerate(angles):
                anchors[i, :, j] = self.generate_anchor(origin, angle)  # num_anchor_properties

        return anchors  # n_angles, n_anchor_properties, n_origins

    def generate_anchor(self, start, angle) -> torch.Tensor:
        anchor_ys = self.anchor_ys  # between [1,0]. starts from 1 and has n_offsets
        anchor = torch.zeros(self.n_anchor_properties)
        angle_rad = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        # anchor[:2] are the probabilities of positive/negative anchor
        anchor[2] = start_y
        anchor[3] = start_x * (self.img_w - 1)
        # anchor[4] is the length
        anchor[5:] = (start_x + (anchor_ys - start_y) / math.tan(angle_rad)) * (self.img_w - 1)
        return anchor  # [num_anchor_properties, ]

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=6000):
        resnet_features = self.feature_extractor(x)  # B, Cr, Hf, Wf

        features = self.conv_bottom(resnet_features)  # B, Cf, Hf, Wf
        lane_features = self.head_conv_bottom(features)  # B, A*(2 + 1 + S), Hf, Wf

        features_left = self.conv_left(resnet_features)  # B, Cf, Hf, Wf
        lane_features_left = self.head_conv_left(features_left)  # B, A*(2 + 1 + S), Hf, Wf

        features_right = self.conv_right(resnet_features)  # B, Cf, Hf, Wf
        lane_features_right = self.head_conv_right(features_right)  # B, A*(2 + 1 + S), Hf, Wf

        sh = lane_features.shape
        # B, A, (2 + 1 + S), Hf, Wf
        lane_features_map = torch.reshape(lane_features, (sh[0], self.n_angles, self.n_anchor_properties - 2, sh[-2], sh[-1]))
        lane_features_map_left = torch.reshape(lane_features_left, (sh[0], self.n_angles, self.n_anchor_properties - 2, sh[-2], sh[-1]))
        lane_features_map_right = torch.reshape(lane_features_right, (sh[0], self.n_angles, self.n_anchor_properties - 2, sh[-2], sh[-1]))
        lane_features_map[:, :, :, :-1, 0] = lane_features_map_left[:, :, :, :-1, 0]
        lane_features_map[:, :, :, :-1, -1] = lane_features_map_right[:, :, :, :-1, -1]

        # Now select only the lane features from the left, right and bottom.
        # B, num_angles * (2 + 1 + S), 2 * (Hf-1) + Wf
        lane_proposals = torch.cat([lane_features_left[:, :, :-1, 0], lane_features_right[:, :, :-1, -1], lane_features[:, :, -1, :]], -1)

        # B, 2 * (Hf-1) + Wf, num_angles * (2 + 1 + S)
        lane_proposals = lane_proposals.permute([0, 2, 1])

        batch_size = lane_proposals.shape[0]
        # B, 2 * (Hf-1) + Wf, num_angles, (2 + 1 + S)
        lane_proposals = torch.reshape(lane_proposals, [batch_size, self.n_anchor_pos, self.n_angles, -1])

        # B, num_angles, 2 * (Hf-1) + Wf, (2 + 1 + S)
        lane_proposals = lane_proposals.permute([0, 2, 1, 3])

        # B, (num_angles*(2 * (Hf-1) + Wf)), (2 + 1 + S)
        lane_proposals = torch.reshape(lane_proposals, [batch_size, self.n_angles * self.n_anchor_pos, -1])

        # BS, n_angles*(2 * (Hf-1) + Wf), n_anchor_properties
        anchors_flat = torch.cat([self.anchors_flat[None, ...]] * batch_size, axis=0)

        # BS, n_angles * (2*(Hf-1) + Wf), n_anchor_properties
        lane_proposals = torch.cat([lane_proposals[..., :2],
                                    anchors_flat[..., 2:4],
                                    lane_proposals[..., 2:] + anchors_flat[..., 4:]], axis=-1)

        lane_proposal_list = self.nms(lane_proposals, nms_thres, nms_topk, conf_threshold)
        return lane_proposal_list, lane_features_map

    def nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold, is_training=False) -> List[List[torch.Tensor]]:
        softmax = nn.Softmax(dim=1)
        proposals_list = []

        if False:
            print("Running NMS for evaluations.")
            for proposals in batch_proposals:
                # The gradients do not have to (and can't) be calculated for the NMS procedure
                with torch.no_grad():
                    scores = softmax(proposals[:, :2])[:, 1]
                    if conf_threshold is not None:
                        # apply confidence threshold
                        above_threshold = scores > conf_threshold
                        proposals = proposals[above_threshold]
                        scores = scores[above_threshold]
                    if proposals.shape[0] == 0:
                        print("All proposals were removed.")
                        proposals_list.append(proposals[[]])
                        continue

                    *_, distances = match_proposals_with_targets(self, proposals, proposals)

                    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                    keep = keep[:num_to_keep]

                proposals = proposals[keep]
                proposals_list.append((proposals, self.anchors_flat[keep]))
        else:
            for proposals in batch_proposals:
                anchors_flat = self.anchors_flat
                if conf_threshold is not None:
                    scores = softmax(proposals[:, :2])[:, 1]
                    keep = scores >= conf_threshold
                    proposals = proposals[keep]
                    anchors_flat = anchors_flat[keep]
                proposals_list.append((proposals, anchors_flat))
        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10.0, reg_loss_weight=30.0):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices, distances = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(num_negatives).long()
                cls_pred = proposals[negatives_mask, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum() / num_negatives
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]
            cls_loss += (focal_loss(cls_pred[:num_positives], cls_target[:num_positives]).sum() / num_positives) + \
                        10. * \
                        (focal_loss(cls_pred[num_positives:], cls_target[num_positives:]).sum() / num_negatives)

            # Regression targets
            reg_pred = positives[:, 4:]
            reg_pred[:, 0] = reg_pred[:, 0] / self.n_offsets
            reg_pred[:, 1:] = reg_pred[:, 1:] / self.img_w
            with torch.no_grad():
                target = target[target_positives_indices]

                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()

                target[:, 4] = target[:, 4] + (positive_starts - target_starts)

                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts - (target[:, 4] - 1)).long()

                valid_offsets_mask = torch.zeros((num_positives, self.n_offsets), dtype=torch.int16)  # S
                valid_offsets_mask[all_indices, self.n_strips - positive_starts] = 1
                valid_offsets_mask[all_indices, self.n_strips - ends] -= 1
                valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) > 0
                valid_offsets_mask[all_indices[target[:, 4] > 0], (self.n_strips - ends)[target[:, 4] > 0]] = True

                invalid_offsets_mask = ~valid_offsets_mask
                invalid_offsets_mask = torch.cat((torch.zeros_like(invalid_offsets_mask[:, :1], dtype=torch.bool), invalid_offsets_mask), dim=1)

                reg_target = target[:, 4:]
                reg_target[:, 0] = reg_target[:, 0] / self.n_offsets
                reg_target[:, 1:] = reg_target[:, 1:] / self.img_w
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += (smooth_l1_loss(reg_pred, reg_target) / num_positives)

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss
        return loss, {'cls_loss': cls_loss_weight * cls_loss, 'reg_loss': reg_loss_weight * reg_loss, 'batch_positives': total_positives}

    def draw_anchors(self, img_w, img_h, k=None):
        from IPython.display import clear_output
        import time
        base_ys = self.anchor_ys.numpy()
        img = Image.open("/home/vincentmayer/repos/LaneATT/datasets/tusimple/clips/0531/1492626274615008344/2.jpg").resize((self.img_w, self.img_h))
        i = -1
        for anchor in self.anchors_flat:
            i += 1
            # if k is not None and i != k:
            #     continue
            anchor = anchor.numpy()
            anchor_start_x_img = anchor[3]*img_w
            anchor_start_y_img = (1-anchor[2])*img_h
            print(f"Start_x {anchor_start_x_img} Start_y: {anchor_start_y_img}")
            # get the anchor x,y coordinates and coords in image
            xs = anchor[5:]
            ys = base_ys * img_h
            xs_in_img = xs[(xs<=img_w) * (xs>=0)]
            ys_in_img = ys[(xs<=img_w) * (xs>=0)]
            plt.imshow(img, origin="upper")
            # plot anchor origin and anchor line
            plt.plot(anchor_start_x_img, anchor_start_y_img, color='r', markersize=10, marker="o")
            plt.plot(xs_in_img, ys_in_img, color='g', markersize=3, marker='o')
            plt.show()
            # loop visualuization in jupyter notebook
            clear_output(wait=True)
            time.sleep(0.2)
        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = self.n_strips - int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, self.n_strips)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            # mask = ~((((lane_xs[:start] >= 0.) &
            #            (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[:start] = -2
            if end < self.n_strips:
                lane_xs[end + 1:] = -2

            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]

            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)

            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, *_ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                print("Proposals were empty.")
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.anchors_anchor_dim = cuda_self.anchors_anchor_dim.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.anchors_flat = device_self.anchors_flat.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    if backbone == 'resnet101':
        backbone = torch.nn.Sequential(*list(resnet101(pretrained=pretrained).children())[:-2])
        fmap_c = 2048
        stride = 32
    elif backbone == 'resnet34':
        backbone = torch.nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride