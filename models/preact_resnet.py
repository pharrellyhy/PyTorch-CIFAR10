'''
Pytorch implementation for pre-activation ResNet.

Original paper: https://arxiv.org/abs/1603.05027
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
