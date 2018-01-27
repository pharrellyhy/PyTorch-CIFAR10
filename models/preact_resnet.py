'''
Pytorch implementation for pre-activation ResNet.

Original paper: https://arxiv.org/abs/1603.05027
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
