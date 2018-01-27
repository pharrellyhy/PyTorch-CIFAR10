from __future__ import absolute_import

from .resnet import *
from .preact_resnet import *
from .densenet import *
from .inception_v4 import *
from .inception_resnet_v2 import *
from .resnext import *


__models = {
  'resnet18': resnet18,
  'resnet34': resnet34,
  'resnet50': resnet50,
  'resnet101': resnet101,
  'resnet152': resnet152,
  'densenet_cifar': densenet_cifar,
  'preact_resnet18': preact_resnet18,
  'preact_resnet34': preact_resnet34,
  'preact_resnet50': preact_resnet50,
  'preact_resnet101': preact_resnet101,
  'preact_resnet152': preact_resnet152,
}


def construct(model, **kwargs):
  if model not in __models:
    raise KeyError('*** model {} not recognized'.format(model))

  return __models[model](**kwargs)
