import math
from torch import nn
from yolov6.layers.common import GradientReversal
from collections import OrderedDict
from yolov6.utils.torch_utils import initialize_weights


class SingleGlobalDomainClassifer(nn.Module):
    def __init__(self, num_conv=2, feat_in_channel=256, dis_conv_channel=256, grl_lambda=1.0, grl_applied_domain='both'):
        """
        Args:
            feat_in_channels        : number of the input feature channels
            dis_conv_channels       : number of the middle conv channels
            grl_lambda         : dx = -lambda*grads, the multiplier of the reversal grads
            grl_applied_domain : which domain will use the grl
        """

        super(SingleGlobalDomainClassifer, self).__init__()

        self.prior_prob = 1e-2

        self.dis_stems = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels=feat_in_channel, out_channels=dis_conv_channel, kernel_size=1, stride=1, padding=0)),
            ('batchnorm_1', nn.BatchNorm2d(num_features=dis_conv_channel)),
            ('relu_1', nn.ReLU())
        ]))

        for i in range(num_conv):
            self.dis_stems.add_module('conv_%d' % (i + 2), nn.Conv2d(
                in_channels=dis_conv_channel,
                out_channels=dis_conv_channel,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            self.dis_stems.add_module('batchnorm_%d'%(i+2), nn.BatchNorm2d(num_features=dis_conv_channel))
            self.dis_stems.add_module('relu_%d'%(i+2), nn.ReLU())

        self.cls_logits = nn.Conv2d(
            in_channels=dis_conv_channel,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.initialize_cls_logits()

        self.grl = GradientReversal(grl_lambda)

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

    def initialize_cls_logits(self):
        for m in self.cls_logits.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature,  domain='source'):
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grl(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grl(feature)

        x = self.dis_stems(feature)
        x = self.cls_logits(x)

        return x


class GlobalDomainClassifer(nn.Module):
    def __init__(
            self,
            num_convs=[2]*3,
            feat_in_channels=[256]*3,
            dis_conv_channels=[256]*3,
            grl_lambdas=[1.0]*3,
            grl_applied_domains=['both']*3
    ):
        super(GlobalDomainClassifer, self).__init__()

        self.gdcs = nn.ModuleList()
        for i in range(3):
            self.gdcs.append(
                SingleGlobalDomainClassifer(
                    num_conv=num_convs[i],
                    feat_in_channel=feat_in_channels[i],
                    dis_conv_channel=dis_conv_channels[i],
                    grl_lambda=grl_lambdas[i],
                    grl_applied_domain=grl_applied_domains[i]
                )
            )

        # Init weights
        initialize_weights(self)

    def forward(self, features):
        return tuple([self.gdcs[i](features[i]) for i in range(3)])


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network_gdc(config):
    width_mul = config.model.width_multiple
    channels_list_backbone = config.model.backbone.out_channels
    channels_list_backbone = [make_divisible(i * width_mul, 8) for i in channels_list_backbone]
    channels_list_neck = config.model.neck.out_channels
    channels_list_neck = [make_divisible(i * width_mul, 8) for i in channels_list_neck]
    if config.gdc.get('featuremaps', False):
        if config.gdc.featuremaps == 'backbone':
            feat_in_channels = [channels_list_backbone[i + 2] for i in range(3)]
        elif config.gdc.featuremaps == 'fpn':
            feat_in_channels = [channels_list_neck[2 * i + 1] for i in range(3)]
        else:
            feat_in_channels = [channels_list_backbone[i + 2] for i in range(3)]
    else:
        feat_in_channels = [channels_list_backbone[i + 2] for i in range(3)]
    
    num_convs = config.gdc.num_convs
    dis_conv_channels = config.gdc.dis_conv_channels
    grl_lambdas = config.gdc.grl_lambdas
    grl_applied_domain = config.gdc.grl_applied_domain
    GDC = eval(config.gdc.type)
    gdc = GDC(
        num_convs=num_convs,
        feat_in_channels=feat_in_channels,
        dis_conv_channels=dis_conv_channels,
        grl_lambdas=grl_lambdas,
        grl_applied_domains=[grl_applied_domain] * 3
    )

    return gdc


def build_gdc_net(cfg, device):
    gdc_net = build_network_gdc(cfg).to(device)
    return gdc_net




