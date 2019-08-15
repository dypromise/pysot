from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head
from pysot.models.neck import get_neck
from pysot.models.head.proposal_layer import _ProposalLayer
from pysot.models.head.proposal_target_layer import _ProposalTargetLayer
from faster_rcnn_lib.model.utils.config import cfg as cfg_rcnn
from faster_rcnn_lib.model.roi_crop.modules.roi_crop import _RoICrop
from faster_rcnn_lib.model.utils.net_utils import _smooth_l1_loss, \
    _affine_grid_gen, _crop_pool_layer


def _center2corner_batch(center):
    """
    center: (b,4) ndarray
    """
    x, y, w, h = center[:, 0], center[:, 1], center[:, 2], center[:, 3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return np.vstack((x1, y1, x2, y2)).transpose()


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.RCNN_base = get_backbone(cfg.BACKBONE.TYPE,
                                      **cfg.BACKBONE.KWARGS)
        self.RCNN_neck = get_neck(cfg.ADJUST.TYPE,
                                  **cfg.ADJUST.KWARGS)
        self.RCNN_rpn = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)
        self.RCNN_proposal = _ProposalLayer()
        self.RCNN_proposal_target = _ProposalTargetLayer()
        self.RCNN_roi_crop = _RoICrop()
        self.RCNN_head_hidden = nn.Linear(512, 256)
        self.RCNN_cls_score = nn.Linear(256, 2)
        self.RCNN_bbox_pred = nn.Linear(256, 4)
        self.grid_size = cfg_rcnn.POOLING_SIZE * 2 if \
            cfg_rcnn.CROP_RESIZE_WITH_MAX_POOL else cfg_rcnn.POOLING_SIZE
        self.training = True

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if cfg.MASK.MASK else None
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        batch_size = search.size(0)

        bbox = data['bbox'].numpy()  # b,4
        bbox = torch.from_numpy(_center2corner_batch(bbox)).long()
        gt_boxes = torch.cat([bbox.unsqueeze(1), torch.ones(
            batch_size, 1, 1).long()], 2).cuda()  # b, 1, 5

        # get backbone feature
        zf = self.RCNN_base(template)
        xf = self.RCNN_base(search)

        # get neck feature
        zf = self.RCNN_neck(zf)
        xf = self.RCNN_neck(xf)

        # rpn result
        cls, loc = self.RCNN_rpn(zf, xf)

        # roi proposals
        rois = self.RCNN_proposal(cls, loc)  # b, num_rois, 5

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            (rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws
             ) = self.RCNN_proposal_target(rois, gt_boxes)

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(
                rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        base_feat = xf.clone()
        if cfg_rcnn.POOLING_MODE == 'crop':
            pooled_feat, _ = _crop_pool_layer(
                base_feat, rois.view(-1, 5), max_pool=False, stride=8.0)
            # grid_xy = _affine_grid_gen(
            #     rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            # grid_yx = torch.stack([grid_xy.data[:, :, :, 1],
            #                        grid_xy.data[:, :, :, 0]], 3).contiguous()
            # pooled_feat = self.RCNN_roi_crop(
            #     base_feat, Variable(grid_yx).detach())
            # if cfg_rcnn.CROP_RESIZE_WITH_MAX_POOL:
            #     pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg_rcnn.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg_rcnn.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # global average pool
        pooled_feat = pooled_feat.mean([2, 3])

        # pooling zf feat
        zf_size = zf.size(3)
        zf_crop_l = zf_size // 4
        zf_crop_r = zf_crop_l + zf_size // 2
        template_feat = zf[:, :, zf_crop_l:zf_crop_r, zf_crop_l:zf_crop_r]
        template_feat = template_feat.mean([2, 3]).expand(
            pooled_feat.size(0), template_feat.size(1))

        # tail hidden
        hidden_feat = torch.cat([pooled_feat, template_feat], 1)
        hidden_feat = self.RCNN_head_hidden(hidden_feat)

        # compute bbox offset and classification probability
        bbox_pred = self.RCNN_bbox_pred(hidden_feat)
        cls_score = self.RCNN_cls_score(hidden_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_cls_loss = 0
        RCNN_bbox_loss = 0

        if self.training:
            # classification loss
            RCNN_cls_loss = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_bbox_loss = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # get rpn loss
        cls = self.log_softmax(cls)
        RPN_cls_loss = select_cross_entropy_loss(cls, label_cls)
        RPN_loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * RPN_cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * RPN_loc_loss + \
            cfg.TRAIN.CLS_WEIGHT * RCNN_cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * RCNN_bbox_loss

        outputs['rpn_cls_loss'] = RPN_cls_loss
        outputs['rpn_loc_loss'] = RPN_loc_loss
        outputs['rcnn_cls_loss'] = RCNN_cls_loss
        outputs['rcnn_loc_loss'] = RCNN_bbox_loss

        return outputs
