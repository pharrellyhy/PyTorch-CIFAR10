'''
Pytorch implementation for pre-activation ResNet.

Original paper: https://arxiv.org/abs/1603.05027
'''

import torch
import torch.nn as nn

from torch.autograd import Variable


__all__ = ['PreActResNet', 'preact_resnet18', 'preact_resnet34', 'preact_resnet50',
           'preact_resnet101', 'preact_resnet152']


def preact_resnet18():
    model = PreActResNet(BasicPreAct, [2, 2, 2, 2])

    return model


def preact_resnet34():
    model = PreActResNet(BasicPreAct, [3, 4, 6, 3])

    return model


def preact_resnet50():
    model = PreActResNet(FullPreAct, [3, 4, 6, 3])

    return model


def preact_resnet101():
    model = PreActResNet(FullPreAct, [3, 4, 23, 3])

    return model


def preact_resnet152():
    model = PreActResNet(FullPreAct, [3, 8, 36, 3])

    return model


class BasicPreAct(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicPreAct, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual

        return out


class FullPreAct(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(FullPreAct, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride,
                    bias=False)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))

        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


def test_cifar():
    print('----- Testing cifar[PreactResNet] -----')
    model = preact_resnet152()
    # model = model.cuda()
    print(model)
    X = torch.randn(1, 3, 32, 32)
    out = model(Variable(X))
    print('out:', out)
    print('out size:', out.size())


if __name__ == '__main__':
    test_cifar()
