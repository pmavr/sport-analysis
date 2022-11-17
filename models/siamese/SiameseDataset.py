import random
import scipy.io as sio

import torch
from torch.utils.data.dataset import Dataset

from util import utils


class SiameseDataset(Dataset):

    def __init__(self,
                 pivot_data,
                 positive_data,
                 batch_size,
                 num_of_batches,
                 data_transform,
                 is_train=True):

        self.pivot_data = pivot_data
        self.positive_data = positive_data
        self.batch_size = batch_size
        self.num_of_batches = num_of_batches
        self.data_transform = data_transform
        self.num_camera = pivot_data.shape[0]

        self.positive_index = []
        self.negative_index = []
        self.is_train = is_train

        if self.is_train:
            self.shuffle_data()
        else:
            # in testing, loop over all pivot cameras
            self.num_of_batches = self.num_camera // batch_size
            # if self.num_camera % batch_size != 0:
            #     self.num_of_batches += 1

    def shuffle_data(self):
        self.positive_index = []
        self.negative_index = []
        num = self.batch_size * self.num_of_batches
        c_set = set([i for i in range(self.num_camera)])
        for i in range(num):
            idx1, idx2 = random.sample(c_set, 2)  # select two indices in random
            self.positive_index.append(idx1)
            self.negative_index.append(idx2)

    def _get_train_item(self, index):
        """
        :param index:
        :return:
        """
        assert index < self.num_of_batches

        n, c, h, w = self.pivot_data.shape

        start_index = self.batch_size * index
        end_index = start_index + self.batch_size
        positive_index = self.positive_index[start_index:end_index]
        negative_index = self.negative_index[start_index:end_index]

        x1, x2, label = [], [], []

        for i in range(self.batch_size):
            idx1, idx2 = positive_index[i], negative_index[i]
            pivot = self.pivot_data[idx1].squeeze()
            pos = self.positive_data[idx1].squeeze()
            neg = self.pivot_data[idx2].squeeze()

            pivot = self.data_transform(pivot)
            pos = self.data_transform(pos)
            neg = self.data_transform(neg)

            x1.append(pivot)
            x1.append(pivot)
            x2.append(pos)
            x2.append(neg)

            label.append(1.)
            label.append(0.)

        return torch.stack(x1), torch.stack(x2), torch.tensor(label)

    def _get_test_item(self, index):
        """
        In testing, the label is hole-fill value, not used in practice.
        :param index:
        :return:
        """
        assert index < self.num_of_batches

        n, c, h, w = self.pivot_data.shape

        start_index = self.batch_size * index
        end_index = start_index + self.batch_size
        pivot_data = self.pivot_data[start_index:end_index]

        x, label_dummy = [], []

        for i in range(self.batch_size):
            pivot = pivot_data[i].squeeze()
            pivot = self.data_transform(pivot)
            x.append(pivot)

        return torch.stack(x)

    def total_dataset_size(self):
        return self.num_of_batches * self.batch_size

    def __getitem__(self, index):
        if self.is_train:
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def __len__(self):
        return self.num_of_batches


if __name__ == '__main__':
    import sys
    from torchvision.transforms import ToTensor, Normalize, Compose

    world_cup_2014_dataset_path = utils.get_world_cup_2014_dataset_path()
    data = sio.loadmat(f'{world_cup_2014_dataset_path}train_data_10k.mat')

    pivot_images = data['pivot_images']
    positive_images = data['positive_images']

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.0188], std=[0.128])])

    train_dataset = SiameseDataset(
        pivot_images,
        positive_images,
        batch_size=32,
        num_of_batches=64,
        data_transform=transform,
        is_train=True)

    for i in range(train_dataset.__len__()):
        [x1, x2], label1 = train_dataset[i]
        print(f'{x1.shape} {x2.shape} {label1.shape}')
        break

    # test_dataset = SiameseDataset(pivot_images, positive_images, batch_size, num_batch, normalize, is_train=False)

    # for i in range(len(train_dataset)):
    #     x, _ = test_dataset[i]
    #     print(f'{x.shape}')
    #     break

    # print('train, test dataset size {} {}'.format(len(train_dataset), len(test_dataset)))
    sys.exit()
