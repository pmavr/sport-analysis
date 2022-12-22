import sys
from time import time
import scipy.io as sio
import seaborn as sns

import torch
from torch.nn import PairwiseDistance
from torch.optim import Adam
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.backends.cudnn as cudnn

from models.siamese.SiameseDataset import SiameseDataset
from models.siamese.Siamese import Siamese
from models.siamese.ContrastiveLoss import ContrastiveLoss
from util import utils, plot

sns.set()


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def fit_model(model, opt_func, loss_func, train_loader,
              num_of_epochs=10, num_of_epochs_until_save=20, silent=False, history=None):
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        loss_func = loss_func.to(device)
        optimizer_to(opt_func, device)
        cudnn.benchmark = True

    l2_distance = PairwiseDistance(p=2)

    if history:
        hist = history
        num_of_trained_epochs = len(history['train_loss'])
    else:
        hist = {
            'train_loss': [],
            'val_loss': [],
            'positive_distance': [],
            'negative_distance': [],
            'distance_ratio': []}
        num_of_trained_epochs = 0
    num_of_epochs += num_of_trained_epochs

    start_time = time()
    epoch_train_data = train_loader.total_dataset_size()

    model.train()
    for epoch in range(num_of_trained_epochs, num_of_epochs):
        epoch_start_time = time()

        epoch_train_loss = 0.
        positive_distance = 0.
        negative_distance = 0.

        for i in range(train_loader.__len__()):
            x1, x2, y_true = train_loader[i]
            x1, x2, y_true = x1.to(device), x2.to(device), y_true.to(device)

            f1, f2 = model(x1, x2)

            opt_func.zero_grad()
            loss = loss_func(f1, f2, y_true)
            loss.backward()
            opt_func.step()

            epoch_train_loss += loss.item()

            distance = l2_distance(f1, f2)
            for j in range(len(y_true)):
                if y_true[j] == 1:
                    positive_distance += distance[j]
                elif y_true[j] == 0:
                    negative_distance += distance[j]
                else:
                    assert 0

        epoch_train_loss /= epoch_train_data
        positive_distance /= epoch_train_data
        negative_distance /= epoch_train_data
        distance_ratio = negative_distance / (positive_distance + 0.000001)

        hist['train_loss'].append(epoch_train_loss)
        hist['positive_distance'].append(float(positive_distance))
        hist['negative_distance'].append(float(negative_distance))
        hist['distance_ratio'].append(float(distance_ratio))

        epoch_duration = time() - epoch_start_time

        if (epoch + 1) % num_of_epochs_until_save == 0:
            model_components = {
                'model': model,
                'opt_func': optimizer}
            utils.save_model(model_components, hist,
                             f"{utils.get_siamese_model_path()}siamese_{len(hist[next(iter(hist))])}.pth")

        train_loader.shuffle_data()

        if not silent:
            print(f"Epoch {epoch + 1}/{num_of_epochs}: Duration: {epoch_duration:.2f} "
                  f"| Train Loss: {hist['train_loss'][-1]:.5f} "
                  f"| Positive Dist.: {hist['positive_distance'][-1]:.3f} "
                  f"| Negative Dist.: {hist['negative_distance'][-1]:.3f} "
                  f"| Dist. Ratio: {hist['distance_ratio'][-1]:.3f}")
        else:
            print('.', end='')

    training_time = time() - start_time
    print('\nTotal training time: {:.2f}s'.format(training_time))

    return model, opt_func, hist


if __name__ == '__main__':

    print('[INFO] Loading training data..')
    data = sio.loadmat(f'{utils.get_camera_estimator_files_path()}train_data_10k.mat')
    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    train_dataset = SiameseDataset(
        pivot_images,
        positive_images,
        batch_size=64,
        num_of_batches=128,
        data_transform=transform,
        is_train=True)

    siamese = Siamese()
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = Adam(filter(lambda p: p.requires_grad, siamese.parameters()),
                     lr=.01,
                     weight_decay=0.000001)

    # siamese, optimizer, history = Siamese.load_model(f'{utils.get_generated_models_path()}siamese_400.pth',
    #                  siamese, optimizer, history=True)

    network, optimizer, history = fit_model(
        model=siamese,
        opt_func=optimizer,
        loss_func=criterion,
        train_loader=train_dataset,
        num_of_epochs=10,
        num_of_epochs_until_save=20)
    # history=history)

    plot.plot_siamese_results(history, info='')

    model_components = {
        'model': network,
        'opt_func': optimizer}
    utils.save_model(model_components, history,
                     f"{utils.get_siamese_model_path()}siamese_{len(history[next(iter(history))])}.pth")

    sys.exit()
