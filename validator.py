from __future__ import absolute_import

from time import sleep
from tqdm import tqdm, trange

from torch.autograd import Variable

from utils.meter import RuntimeMeter
from utils.utils import accuracy


class Validator():

    def __init__(self, model, criterion):
        super(Validator, self).__init__()
        self.model = model
        self.criterion = criterion

    def validate(self, epoch, val_loader, logger, print_freq=10):
        losses = RuntimeMeter()
        top1 = RuntimeMeter()

        self.model.eval()

        t = tqdm(enumerate(val_loader), total=len(val_loader), desc='', leave=True)
        for i, (inputs, targets) in t:
            targets = targets.cuda(async=True)
            inputs_var = Variable(inputs, volatile=True)
            targets_var = Variable(targets, volatile=True)

            outputs = self.model(inputs_var)

            loss = self.criterion(outputs, targets_var)

            prec = accuracy(outputs.data, targets)
            losses.update(loss.data[0], targets.size(0))
            top1.update(prec[0][0], targets.size(0))

            if i % print_freq == 0:
                t.set_description('*Val: [{0}][{1}/{2}] | '
                        '*Loss=({loss.val:.3f})/({loss.avg:3f}) | '
                        '*Prec@1=({top1.val:.3f})/({top1.avg:3f})'.format(
                          epoch, i + 1, len(val_loader), loss=losses, top1=top1))
                t.refresh()
                sleep(0.01)
            # if (i + 1) % print_freq == 0:
            #     print(
            #         '*Val: [{0}][{1}/{2}]\n'
            #         '*Loss {loss.val:.4f} ({loss.avg:4f})\n'
            #         '*Precision@1 {top1.val:.4f} ({top1.avg:4f})\n'.format(
            #           epoch, i + 1, len(val_loader), loss=losses, top1=top1)
            #     )

            logger.scalar_summary('val_loss', losses.avg, epoch * len(val_loader) + i + 1)

        logger.scalar_summary('val_acc', top1.avg, epoch + 1)

        return top1.avg
