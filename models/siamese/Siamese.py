
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, Flatten
from torch.nn import LeakyReLU, ReLU
from torch.nn.functional import normalize


class Siamese(Module):

    def __init__(self):
        super(Siamese, self).__init__()
        layers = []
        in_channels = 1
        layers += [Conv2d(in_channels, 4, kernel_size=7, stride=2, padding=3)]
        layers += [LeakyReLU(0.1, inplace=True)]

        layers += [Conv2d(4, 8, kernel_size=5, stride=2, padding=2), ReLU(inplace=True)]

        layers += [Conv2d(8, 16, kernel_size=3, stride=2, padding=1), ReLU(inplace=True)]

        layers += [Conv2d(16, 32, kernel_size=3, stride=2, padding=1), ReLU(inplace=True)]

        layers += [Conv2d(32, 16, kernel_size=3, stride=2, padding=1), ReLU(inplace=True)]

        self.branch = Sequential(*layers)
        self.fc = Sequential(*[Linear(6 * 10 * 16, 16)])

    def _forward_one_branch(self, x):
        x = self.branch(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        x = normalize(x, p=2)
        return x

    def forward(self, x1, x2):
        x1 = self._forward_one_branch(x1)
        x2 = self._forward_one_branch(x2)
        return x1, x2

    def feature_numpy(self, x):
        feat = self._forward_one_branch(x)
        feat = feat.data
        feat = feat.cpu()
        feat = feat.numpy()
        if len(feat.shape) == 4:
            # N x C x 1 x 1
            feat = np.squeeze(feat, axis=(2, 3))
        else:
            # N x C
            assert len(feat.shape) == 2
        return feat

    @staticmethod
    def load_model(filename, model=None, optimizer=None, history=None):
        """Load trained model along with its optimizer and training, plottable history."""
        model_components = torch.load(filename)
        if model:
            model.load_state_dict(model_components['model'])
        if optimizer:
            optimizer.load_state_dict(model_components['optimizer'])
        if history:
            history = model_components['history']
        return model, optimizer, history

if __name__ == '__main__':
    import sys
    from ContrastiveLoss import ContrastiveLoss

    siamese = Siamese()
    criterion = ContrastiveLoss(margin=1.0)

    import numpy as np
    np.random.seed(0)
    N = 2
    x1 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)
    x2 = torch.tensor(np.random.rand(N, 1, 180, 320), dtype=torch.float32)

    y1 = torch.tensor(np.random.rand(N, 1), dtype=torch.float32)
    y_zeros = torch.zeros(N, 1)
    y_ones = torch.ones(N, 1)

    y_true = torch.where(y1 > 0, y_ones, y_zeros)
    y_true = torch.squeeze(y_true)

    f1, f2 = siamese(x1, x2)
    loss = criterion(f1, f2, y_true)
    print(f1)
    print(f2)
    print(loss)
    sys.exit()
