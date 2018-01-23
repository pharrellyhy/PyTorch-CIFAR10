from __future__ import absolute_import

from time import sleep
from tqdm import tqdm, trange

from torch.autograd import Variable

from utils.meter import RuntimeMeter
from utils.utils import accuracy


class Tester():

    def __init__(self, model, criterion):
        super(Tester, self).__init__()
        self.model = model
        self.criterion = criterion

    def test(self, test_loader, print_freq=10):
        losses = RuntimeMeter()
        top1 = RuntimeMeter()

        self.model.eval()

        t = tqdm(enumerate(test_loader), total=len(test_loader), desc='', ncols=140, leave=True)
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
                t.set_description('*Test: [{0}/{1}] | '
                        '*Loss=({loss.val:.3f})/({loss.avg:.3f}) | '
                        '*Prec@1=({top1.val:.3f})/({top1.avg:.3f})'.format(
                          i + 1, len(test_loader), loss=losses, top1=top1))
                t.refresh()
                sleep(0.01)

        print('*Test precision@1 = {:.4f}'.format(top1.avg))
