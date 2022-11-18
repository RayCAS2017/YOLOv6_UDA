#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.models.effidehead import Detect, build_effidehead_layer
from yolov6.models.global_domain_classifier import *


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        #self.mode = config.training_mode
        self.backbone, self.neck, self.detect, self.gdc = build_network(config, channels, num_classes, anchors, num_layers)

        # Init Detect head
        begin_indices = config.model.head.begin_indices
        out_indices_head = config.model.head.out_indices
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        # if self.gdc is not None and self.backbone.training is True:
        #     gdc_logits = self.gdc(x)
        # else:
        #     gdc_logits = None
        gdc_logits = self.gdc(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)

        if export_mode:
            return x, gdc_logits
        else:
            return [x, featmaps, gdc_logits]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_anchors = config.model.head.anchors
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
    if 'gdc' in config.model.keys():
        num_convs = config.model.gdc.num_convs
        feat_in_channels = [channels_list[i+2] for i in range(3)]
        dis_conv_channels = config.model.gdc.dis_conv_channels
        grl_lambdas = config.model.gdc.grl_lambdas
        grl_applied_domain = config.model.gdc.grl_applied_domain
        GDC = eval(config.model.gdc.type)
        gdc = GDC(
            num_convs=num_convs,
            feat_in_channels=feat_in_channels,
            dis_conv_channels=dis_conv_channels,
            grl_lambdas=grl_lambdas,
            grl_applied_domains = [grl_applied_domain]*3
        )
    else:
        gdc = None

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head, gdc


def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    return model
