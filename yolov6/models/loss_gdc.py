#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class ComputeLoss_gdc:
    def __init__(self,
                 loss_weight=0.1
                 ):
        self.loss_weight = loss_weight
        self.loss_fn = nn.BCEWithLogitsLoss()

    def __call__(self,
                 gdc_logits,
                 target):
        # 0: target domain, 1: source domain
        assert target == 0 or target == 1

        loss = []
        for logit in gdc_logits:
            logit_target = torch.full(logit.shape, target, dtype=logit.dtype, device=logit.device)
            loss.append(self.loss_fn(logit, logit_target))

        return sum(loss)*self.loss_weight
