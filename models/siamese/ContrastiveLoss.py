import sys

import torch
from torch.nn import Module, PairwiseDistance


class ContrastiveLoss(Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(self, x1, x2, y_true, **kwargs):
        """
        :param margin:
        :param x1: N * D
        :param x2: N * D
        :param y_true: 0 for un-similar, 1 for similar
        :return:
        """
        assert len(x1.shape) == 2
        assert len(x2.shape) == 2
        assert len(y_true.shape) == 1
        assert y_true.shape[0] == x1.shape[0]

        pdist = PairwiseDistance(p=2)
        dist = pdist(x1, x2)
        loss = y_true * torch.pow(dist, 2) \
               + (1-y_true) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return torch.sum(loss)


if __name__ == '__main__':
    contrastive_loss = ContrastiveLoss(margin=1.0)
    import numpy as np
    np.random.seed(0)
    N = 10
    x1 = torch.tensor(np.random.rand(N, 16), dtype=torch.float32)
    x2 = torch.tensor(np.random.rand(N, 16), dtype=torch.float32)
    y_pred = [x1, x2]

    y1 = torch.tensor(np.random.rand(N, 1))
    y_zeros = torch.zeros((N, 1))
    y_ones = torch.ones((N, 1))

    y_true = torch.where(y1 > 0, y_ones, y_zeros)
    y_true = torch.squeeze(y_true)

    loss = contrastive_loss(x1, x2, y_true)
    print(loss.shape)
    print(loss)

    sys.exit()
