'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
This MoE design is based on the implementation of Yerlan Idelbayev.
'''

from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out

class ShallowExpert(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(depth)]))

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class ResNet_MoE_cls(nn.Module):

    def __init__(self, num_blocks, num_experts=3, num_classes=10, use_norm=False,fead_in = 64):
        super(ResNet_MoE_cls, self).__init__()
        self.s = 1
        self.num_experts = num_experts
        self.use_norm = use_norm

        if use_norm:
            self.s = 30
            self.classifiers = nn.ModuleList(
                    [NormedLinear(fead_in, num_classes) for _ in range(self.num_experts)])
            self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(fead_in, num_classes) for _ in range(self.num_experts)])
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(fead_in, num_classes, bias=True) for _ in range(self.num_experts)])

        self.apply(_weights_init)

        self.depth = list(
            reversed([i + 1 for i in range(len(num_blocks) - 1)]))  # [2, 1]
        
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(
            self.num_experts)]  # [2, 1, 2]
        
        feat_dim = 16
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])


    def call_last_layer(self,block):
        import copy
        self.layer3s = nn.ModuleList([copy.deepcopy(block) for _ in range(self.num_experts - 1)])

    def forward(self, x, crt=False):
        shallow_outs = x[:-1]
        out3s = [x[-1]] + [ self.layer3s[_](shallow_outs[-1]) for _ in range(self.num_experts - 1)]

        shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

        exp_outs = [out3s[i] * shallow_expe_outs[i] for i in range(self.num_experts)]
        exp_outs = [F.avg_pool2d(output, output.size()[3]).view(output.size(0), -1) 
                        for output in exp_outs]
            
        if self.use_norm and crt:
            outs = [self.s * self.rt_classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
        else:
            outs = [self.s * self.classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
            
        if not self.training:                
            return sum(outs)/self.num_experts
        return outs

def resnet32_moe_cls(num_class,use_norm):
    return ResNet_MoE_cls([5,5,5],num_classes=num_class , use_norm=use_norm)


def resnet50_moe_cls(num_class,use_norm):
    return ResNet_MoE_cls([3, 4, 6, 3],num_classes=num_class , use_norm=use_norm,fead_in=2048)