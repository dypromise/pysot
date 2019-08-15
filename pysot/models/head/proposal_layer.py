from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from faster_rcnn_lib.model.rpn.bbox_transform import bbox_transform_inv2,\
    clip_boxes
from model.nms.nms_wrapper import nms
from pysot.utils.anchor import Anchors
from pysot.core.config import cfg

DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self):
        super(_ProposalLayer, self).__init__()
        self._feat_size = cfg.TRAIN.OUTPUT_SIZE
        self._feat_stride = cfg.ANCHOR.STRIDE
        self._anchors = torch.from_numpy(
            self._generate_anchor(self._feat_size)).float()
        self._num_anchors = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        self._im_size = cfg.TRAIN.SEARCH_SIZE
        self._pre_nms_topN = 200
        self._post_nms_topN = 30
        self._nms_thresh = 0.7

    def _generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                           x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid(
            [ori + total_stride * dx for dx in range(score_size)],
            [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(
            np.float32), yy.astype(np.float32)
        return anchor  # A*K,4

    def _log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4).permute(
            0, 4, 1, 2, 3).contiguous().view(b, a2, h, w)
        return cls

    def forward(self, pred_cls, pred_reg):
        """
        Take RPN's output feature map as input, outputs roi proposals by using
        anchors
        Params:
            pred_cls: b, 2*A, h, w
            pred_reg: b, 4*A, h, w
        """

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        pred_scores = self._log_softmax(pred_cls)
        scores = pred_scores[:, self._num_anchors:, :, :]  # b, 1*A, h, w
        bbox_deltas = pred_reg  # b, 4*A, h, w

        batch_size, fh, fw = scores.size(0), scores.size(2), scores.size(3)
        assert fh == self._feat_size, "proposal layer: feat_size don't match!"
        K = fh * fw
        A = self._num_anchors

        # expand anchors
        anchors = self._anchors.view(1, A * K, 4).expand(batch_size, A * K, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        bbox_deltas = bbox_deltas.view(batch_size, 4, A, fh, fw).permute(
            0, 2, 3, 4, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # b, A*K, 4

        # Same story for the scores:
        scores = scores.view(batch_size, 2, A, fh, fw).permute(
            0, 2, 3, 4, 1).contiguous()
        scores = scores.view(batch_size, -1)  # b, A*K

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv2(anchors, bbox_deltas, batch_size)
        proposals = clip_boxes(proposals, [self._im_size, self._im_size])

        proposals_keep = proposals
        scores_keep = scores  # b, A*K
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, self._post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # 3. remove predicted boxes with either height or width <
            # threshold. (NOTE: convert min_size to input image scale
            # stored in im_info[2])
            proposals_single = proposals_keep[i]  # A*K, 4
            scores_single = scores_keep[i]  # A*K

            # 4. sort all (proposal, score) pairs by score from highest
            # to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]
            if self._pre_nms_topN > 0 and self._pre_nms_topN < \
                    scores_keep.numel():
                order_single = order_single[:self._pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(
                torch.cat((proposals_single, scores_single), 1),
                self._nms_thresh, force_cpu=False
            )
            keep_idx_i = keep_idx_i.long().view(-1)

            if self._post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self._post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i, :num_proposal, 0] = scores_single
            output[i, :num_proposal, 1:] = proposals_single
        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) &
                (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep
