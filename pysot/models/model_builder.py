# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
# import time


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # branch specific conv
        if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
            self.sconv_z = get_neck(cfg.SPECIFIC_CONV.TYPE,
                                    **cfg.SPECIFIC_CONV.KWARGS)
            self.sconv_x = get_neck(cfg.SPECIFIC_CONV.TYPE,
                                    **cfg.SPECIFIC_CONV.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)
            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]

        if cfg.ADJUST.ADJUST:
            z_crop = True
            if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
                z_crop = False
            zf = self.neck(zf, crop=z_crop)

        if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
            zf = self.sconv_z(zf, crop=True)

        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]

        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf, crop=False)

        if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
            xf = self.sconv_x(xf, crop=False)

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
        label_obj_cls = data['label_obj_cls'].cuda()  # b

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]

        if cfg.ADJUST.ADJUST:
            z_crop = True
            if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
                z_crop = False
            zf = self.neck(zf, crop=z_crop)
            xf = self.neck(xf, crop=False)

        if cfg.SPECIFIC_CONV.SPECIFIC_CONV:
            zf = self.sconv_z(zf, crop=True)
            xf = self.sconv_x(xf, crop=False)

        if cfg.RPN.WITH_OBJ:
            cls, loc, obj_cls = self.rpn_head(zf, xf)
        else:
            cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.RPN.WITH_OBJ:
            obj_cls = F.log_softmax(obj_cls, dim=1)
            obj_cls_loss = F.nll_loss(obj_cls, label_obj_cls)
            outputs['obj_loss'] = obj_cls_loss
            outputs['total_loss'] += cfg.TRAIN.OBJ_CLS_WEIGHT * obj_cls_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
