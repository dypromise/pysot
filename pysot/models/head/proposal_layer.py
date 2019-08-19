from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from faster_rcnn_lib.model.rpn.bbox_transform import bbox_transform_inv2,\
    clip_boxes
from faster_rcnn_lib.model.nms.nms_wrapper import nms
from faster_rcnn_lib.model.utils.config import cfg as cfg_rcnn
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
        self._feat_stride = cfg.ANCHOR.STRIDE
        self._anchor_generator = Anchors(cfg.ANCHOR.STRIDE,
                                         cfg.ANCHOR.RATIOS,
                                         cfg.ANCHOR.SCALES)
        self.init_size(cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        self._num_anchors = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        self._pre_nms_topN = cfg_rcnn.TRAIN.RPN_PRE_NMS_TOP_N
        self._post_nms_topN = cfg_rcnn.TRAIN.RPN_POST_NMS_TOP_N
        self._nms_thresh = cfg_rcnn.TRAIN.RPN_NMS_THRESH

    def _log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4).permute(
            0, 4, 1, 2, 3).contiguous().view(b, a2, h, w)
        return cls

    def init_size(self, search_size, output_size):
        self._anchor_generator.generate_all_anchors(
            im_c=search_size // 2,
            size=output_size
        )
        self._anchors = torch.from_numpy(  # centor anchors
            self._anchor_generator.all_anchors[1]).float().cuda()
        self._im_size = search_size

    def forward(self, pred_cls, pred_reg, post_nms_num=None):
        """
        Take RPN's output feature map as input, outputs roi proposals by using
        anchors
        Params:
            pred_cls: b, 2*A, h, w
            pred_reg: b, 4*A, h, w
        """
        if not post_nms_num:
            post_nms_num = self._post_nms_topN

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        pred_scores = self._log_softmax(pred_cls)
        scores = pred_scores[:, self._num_anchors:, :, :]  # b, 1*A, h, w
        bbox_deltas = pred_reg  # b, 4*A, h, w

        batch_size, fh, fw = scores.size(0), scores.size(2), scores.size(3)
        # print(scores.size())
        # assert fh == self._feat_size, "proposal layer: feat_size don't match!"
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
        scores = scores.view(batch_size, -1)  # b, A*K

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv2(anchors, bbox_deltas, batch_size)
        proposals = clip_boxes(proposals, [self._im_size, self._im_size])

        proposals_keep = proposals
        scores_keep = scores  # b, A*K
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_num, 5).zero_()
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

            if post_nms_num > 0:
                keep_idx_i = keep_idx_i[:post_nms_num]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :].view(-1)

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
