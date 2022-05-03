import math

from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34

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
        self.fmap_h = img_h // self.stride
        self.fmap_w = img_w // self.stride
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.anchor_angles = [15., 22., 30., 39., 49., 60., 72, 80., 90., 100, 108., 120., 131., 141., 150., 158., 165.]
        self.n_anchors_per_pos = len(self.anchor_angles) # 17
        self.n_anchor_pos_edges = self.fmap_w + 2*self.fmap_h

        # Generate anchors Anchor vec = [p(pos), p(neg), y_o, x_o, L, d1, ..., dS]
        self.n_pred_per_anchor = 2 + 2 + 1 + self.n_offsets
        # 1309x90x160
        self.anchors = self.generate_anchors(self.anchor_angles, self.fmap_h, self.fmap_w) # (N_pred_per_anchor*N_angles)xHfxWf
        
        # Now keep another anchor tensor that holds only the stacked anchors from the sides and bottom
        # (A*(2+2+S+1))x(2*Hf+Wf) = (77*17)1309x340
        self.edge_anchors = torch.cat([self.anchors[:, :, 0], self.anchors[:, :, -1], self.anchors[: , -1, :]], 1)

        # A*N_pred*Hf*Wf -> A*(2*Hf+Wf)xN_pred = 5780x77
        self.anchors_anchor_dim = torch.zeros((self.n_anchors_per_pos*self.n_anchor_pos_edges, self.n_pred_per_anchor))
        for pos_ix in range(self.n_anchor_pos_edges):
            for a_ix in range(self.n_anchors_per_pos):
                single_anchor = self.edge_anchors[a_ix*self.n_pred_per_anchor:(a_ix+1)*self.n_pred_per_anchor, pos_ix]
                self.anchors_anchor_dim[pos_ix*self.n_anchors_per_pos + a_ix, :] = single_anchor

        # Setup and initialize layers
        conv_out_channels = 1024
        self.conv = nn.Conv2d(in_channels=backbone_nb_channels, out_channels=conv_out_channels, kernel_size=1)
        self.conv_cls = nn.Conv2d(in_channels=conv_out_channels, out_channels=2*self.n_anchors_per_pos, kernel_size=1)
        self.conv_reg = nn.Conv2d(in_channels=conv_out_channels, out_channels=self.n_anchors_per_pos*(self.n_offsets+1), kernel_size=1)

        self.initialize_layer(self.conv)
        self.initialize_layer(self.conv_cls)
        self.initialize_layer(self.conv_reg)


    def generate_anchors(self, angles, fmap_h, fmap_w):
        n_pred = self.n_pred_per_anchor * self.n_anchors_per_pos
        anchors = torch.zeros((n_pred, fmap_h, fmap_w))

        left_anchors = self.generate_side_anchors(angles, x=0., n_origins=fmap_h)
        right_anchors = self.generate_side_anchors(angles, x=1., n_origins=fmap_h)
        bottom_anchors = self.generate_side_anchors(angles, y=1., n_origins=fmap_w)

        anchors[:,:,0] = left_anchors
        anchors[:,:,-1] = right_anchors
        anchors[:,-1,:] = bottom_anchors

        return anchors
    def generate_side_anchors(self, angles, n_origins, x=None, y=None):
        if x is None and y is not None:
            origins = [(x, y) for x in np.linspace(1., 0., num=n_origins)]
        elif x is not None and y is None:
            origins = [(x, y) for y in np.linspace(1., 0., num=n_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((self.n_pred_per_anchor * self.n_anchors_per_pos, n_origins))
        for i, origin in enumerate(origins):
            anchors_at_origin = torch.empty((0))
            for angle in angles:
                anchor_with_angle = self.generate_anchor(origin, angle) # shape 77x1
                anchors_at_origin = torch.cat([anchors_at_origin, anchor_with_angle], 0)
            anchors[:, i] = anchors_at_origin # shape (77*17=n_pred_per_anchor*n_angles)

        return anchors

    def generate_anchor(self, start, angle):
        anchor_ys = self.anchor_ys
        anchor = torch.zeros(self.n_pred_per_anchor)
        angle_rad = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        # anchor[:2] are the probabilities of positive/negative anchor
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        # anchor[4] is the length
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle_rad)) * self.img_w
        return anchor

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=6000):
        resnet_features = self.feature_extractor(x) # BxCrxHfxWf
        features = self.conv(resnet_features) # BxCfxHfxWf

        cls_features = self.conv_cls(features) # Bx2*AxHfxWf
        reg_features = self.conv_reg(features) # BxA*(S+1)xHfxWf

        # fill the lane feature tensor. This is still the same shape as the feature map 
        # BxA*(2+2+S+1)xHfxWf
        batch_size = cls_features.shape[0]
        lane_features = torch.zeros((batch_size, self.n_pred_per_anchor*self.n_anchors_per_pos, self.fmap_h, self.fmap_w), device=x.device)
        lane_features += self.anchors # self.anchors.shape = A*(2+2+S+1)xHfxWf
        for i in range(self.n_anchors_per_pos):
            lane_features[:, i*self.n_pred_per_anchor:i*self.n_pred_per_anchor+2, :, :] = cls_features[:, i*2:(i+1)*2,: ,:]
            lane_features[:, i*(self.n_pred_per_anchor)+4:(i+1)*(self.n_pred_per_anchor), :, :] += reg_features[:, i*(self.n_offsets+1):(i+1)*(self.n_offsets+1), :, :]

        # Now select only the lane features from the left, bottom and right. Bx(A*(2+2+S+1))x(2*Hf+Wf)
        lane_proposals = torch.cat([lane_features[:, :, :, 0], lane_features[:, :, :, -1], lane_features[: ,: , -1, :]], 2)

        # Bx(A*(2+2+S+1))x(2*Hf+Wf) -> BxA*(2*Hf+Wf)x(2+2+S+1)
        lane_proposals_anch_dim = torch.zeros((lane_proposals.shape[0], self.n_anchors_per_pos*self.n_anchor_pos_edges, self.n_pred_per_anchor), device=x.device)
        for pos_ix in range(self.n_anchor_pos_edges):
            for a_ix in range(self.n_anchors_per_pos):
                proposal = lane_proposals[:, a_ix*self.n_pred_per_anchor:(a_ix+1)*self.n_pred_per_anchor, pos_ix]
                lane_proposals_anch_dim[:, pos_ix*self.n_anchors_per_pos + a_ix, :] = proposal

        lane_proposal_list = self.nms(lane_proposals_anch_dim, nms_thres, nms_topk, conf_threshold)
        return lane_proposal_list

    def nms(self, batch_proposals, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals in batch_proposals:
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors_anchor_dim[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            proposals_list.append((proposals, self.anchors_anchor_dim[keep], anchor_inds))

        return proposals_list

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _), target in zip(proposals_list, targets):
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
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def draw_anchors(self, img_w, img_h, k=None):
        from IPython.display import clear_output
        import time
        base_ys = self.anchor_ys.numpy()
        img = Image.open("/home/vincentmayer/repos/LaneATT/datasets/tusimple/clips/0531/1492626274615008344/2.jpg").resize((self.img_w, self.img_h))
        i = -1
        for anchor in self.anchors_anchor_dim:
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
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
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
        for proposals, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
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
        device_self.anchors_anchor_dim = device_self.anchors_anchor_dim.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
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