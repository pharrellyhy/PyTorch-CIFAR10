from __future__ import absolute_import

from time import sleep
from tqdm import tqdm, trange

from torch.autograd import Variable

from utils.meter import RuntimeMeter
from utils.utils import accuracy


class Trainer():

    def __init__(self, model, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, optimizer, epoch, train_loader, logger, print_freq=10):
        losses = RuntimeMeter()
        top1 = RuntimeMeter()

        self.model.train()

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc='', leave=True)
        for i, (inputs, targets) in t:
            targets = targets.cuda(async=True)

            inputs_var = Variable(inputs)
            targets_var = Variable(targets)

            outputs = self.model(inputs_var)

            loss = self.criterion(outputs, targets_var)

            prec = accuracy(outputs.data, targets)
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec[0][0], inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                t.set_description('*Train: [{0}][{1}/{2}] | '
                        '*Loss=({loss.val:.3f})/({loss.avg:3f}) | '
                        '*Prec@1=({top1.val:.3f})/({top1.avg:3f})'.format(
                          epoch, i + 1, len(train_loader), loss=losses, top1=top1))
                t.refresh()
                sleep(0.01)

            # if (i + 1) % print_freq == 0:
            #     print(
            #         '*Train: [{0}][{1}/{2}]\n'
            #         '*Loss {loss.val:.4f} ({loss.avg:4f})\n'
            #         '*Precision@1 {top1.val:.4f} ({top1.avg:4f})\n'.format(
            #           epoch, i + 1, len(train_loader), loss=losses, top1=top1)
            #     )

            logger.scalar_summary('train_loss', losses.avg, epoch * len(train_loader) + i + 1)

        logger.scalar_summary('train_acc', top1.avg, epoch + 1)
