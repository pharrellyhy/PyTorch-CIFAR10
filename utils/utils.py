def accuracy(out, target, topk=(1,)):
  """Calculates the top k precision"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = out.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  result = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    result.append(correct_k.mul(100.0 / batch_size))

  return result
