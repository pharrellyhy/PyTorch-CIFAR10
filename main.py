from __future__ import absolute_import

import sys
import os
import argparse
import time
import datetime
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

import models
from trainer import Trainer
from validator import Validator
from tester import Tester
from utils.logger import Logger


best_prec1 = 0
np.random.seed(40)
torch.manual_seed(40)


def get_parser():
    parser = argparse.ArgumentParser(description='Training multi networks on CIFAR10')
    parser.add_argument('--arch', '-a', default='resnet18', type=str, help='network architecture')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer for updating weights and biases (default: adam)')
    parser.add_argument('--num-workers', '-j', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--epochs', '-ep', default=200, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini batch size')
    parser.add_argument('--learning-rate', '-lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', '-m', default=0., type=float, help='momentum')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency')
    parser.add_argument('--start-epoch', default=0, type=int, help='epoch to resume from')
    parser.add_argument('--resume', default='', type=str, help='path to resumed checkpoint')
    parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true', help='evaluate on test set')
    parser.add_argument('--train-dir', default='', type=str, help='training set directory')
    # parser.add_argument('--val-dir', default='', type=str, help='validation set directory')
    parser.add_argument('--test-dir', default='', type=str, help='test set directory')
    parser.add_argument('--log-dir', default='', type=str, help='directory to save log')

    return parser


def command_line_runner():
    parser = get_parser()
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    print(args)

    return args


def main():
    global best_prec1

    args = command_line_runner()
    saved_name = '{}_{}_mnt{}_lr{}_b{}_ep{}'.format(args.arch, args.optimizer, args.momentum,
                                                    args.learning_rate, args.batch_size, args.epochs)

    if not args.evaluate:
        if not os.path.exists(args.log_dir):
            os.makedir(args.log_dir)

        log_dir = os.path.join(args.log_dir, saved_name)
        logger = Logger(log_dir)

    # creates model
    model = models.construct(args.arch)
    print(model)

    # defines cost function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = choose_optimizer(model, args)

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # creates trainer, validator and tester
    trainer = Trainer(model, criterion)
    validator = Validator(model, criterion)
    tester = Tester(model, criterion)

    # # resumes from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print('\n===> loading checkpoint {} ...\n'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('\n**** checkpoint {} loaded at epoch {} ...\n'.format(args.resume,
                checkpoint['epoch']))
        else:
            raise Exception('\n===> No checkpoint found at {} ...\n'.format(args.resume))

    # data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    }

    # loads datasets
    print('\n===> loading dataset...\n')
    train_set = torchvision.datasets.CIFAR10(root=args.train_dir, train=True, download=False,
                        transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    test_set = torchvision.datasets.CIFAR10(root=args.test_dir, train=False, download=False,
                       transform=data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)
    print('\n===> dataset loaded...\n')

    cudnn.benchmark = True

    if args.evaluate:
        tester.test(test_loader)

        return

    start = datetime.datetime.now().replace(microsecond=0)
    print('\n===> Training starts at: {}\n'.format(start))

    t = tqdm(range(args.start_epoch, args.epochs), desc='Training Process', ncols=100, leave=True)
    for epoch in t:
        adjust_lr(optimizer, epoch, args, logger)

        trainer.train(optimizer, epoch, train_loader, logger, args.print_freq)
        prec1 = validator.validate(epoch, test_loader, logger, args.print_freq)  # same as test loader

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer': optimizer.state_dict()
                         },
                        is_best, path='./checkpoint', filename=saved_name + '.pth.tar')
    end = datetime.datetime.now().replace(microsecond=0)
    print('\n===> Training Done!!!\n')
    print('\n===> Training Duration: {}\n'.format(end - start))
    tester.test(test_loader, args.print_freq)


def adjust_lr(optimizer, epoch, args, logger):
    """Decays the initial learning rate by order of 10 after every 100 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logger.scalar_summary('learning_rate', lr, epoch + 1)


def choose_optimizer(model, args):
    updated_params = model.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(updated_params, lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(updated_params, lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)

    return optimizer


def save_checkpoint(state, is_best, path, filename='./checkpoint/checkpoint.pth.tar'):
  torch.save(state, os.path.join(path, filename))
  if is_best:
    shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'best_model_' + filename))


if __name__ == '__main__':
    main()
