# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import random
import torch
from torch.utils.data import Sampler, DataLoader, ConcatDataset

from models.foot_and_ball.data.issia_dataset import create_issia_dataset, IssiaDataset
from models.foot_and_ball.data.spd_bmvc2017_dataset import create_spd_dataset


def make_dataloaders(params):

    train_issia_dataset = create_issia_dataset(
        params['issia_path'], params['issia_train_cameras'], mode='train', only_ball_frames=False,
        train_image_size=params['image_size'], val_image_size=params['image_size'])

    if len(params['spd_train_cameras']) == 0:
        train_spd_dataset = None
    else:
        train_spd_dataset = create_spd_dataset(
            params['spd_path'], params['spd_train_cameras'], mode='train', image_size=params['image_size'])

    if len(params['issia_val_cameras']) == 0:
        val_issia_dataset = None
    else:
        val_issia_dataset = create_issia_dataset(
            params['issia_path'], params['issia_val_cameras'], mode='val', only_ball_frames=True,
            train_image_size=params['image_size'], val_image_size=params['image_size'])

    if train_spd_dataset:
        train_dataset = ConcatDataset([train_issia_dataset, train_spd_dataset])
    else:
        train_dataset = train_issia_dataset

    batch_sampler = BalancedSampler(train_dataset)
    dataloaders = {}
    dataloaders['train'] = DataLoader(train_dataset, sampler=batch_sampler, batch_size=params['batch_size'],
                                      num_workers=params['num_workers'], pin_memory=True, collate_fn=my_collate)
    if val_issia_dataset is not None:
        dataloaders['val'] = DataLoader(val_issia_dataset, batch_size=2, num_workers=params['num_workers'],
                                        pin_memory=True, collate_fn=my_collate)
    return dataloaders


def my_collate(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return images, boxes, labels


class BalancedSampler(Sampler):
    # Sampler sampling the same number of frames with and without the ball
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()

    def generate_samples(self):
        # Sample generation function expects concatenation of 2 datasets: one is ISSIA CNR and the other is SPD
        # or only one ISSIA CNR dataset.
        assert len(self.data_source.datasets) <= 2
        issia_dataset_ndx = None
        spd_dataset_ndx = None
        for ndx, ds in enumerate(self.data_source.datasets):
            if isinstance(ds, IssiaDataset):
                issia_dataset_ndx = ndx
            else:
                spd_dataset_ndx = ndx

        assert issia_dataset_ndx is not None, 'Training data must contain ISSIA CNR dataset.'

        issia_ds = self.data_source.datasets[issia_dataset_ndx]
        n_ball_images = len(issia_ds.ball_images_ndx)
        # no_ball_images = 0.5 * ball_images
        n_no_ball_images = min(len(issia_ds.no_ball_images_ndx), int(0.5 * n_ball_images))
        issia_samples_ndx = list(issia_ds.ball_images_ndx) + random.sample(issia_ds.no_ball_images_ndx,
                                                                           n_no_ball_images)
        if issia_dataset_ndx > 0:
            # Add sizes of previous datasets to create cummulative indexes
            issia_samples_ndx = [e + self.data_source.cummulative_sizes[issia_dataset_ndx-1] for e in issia_samples_ndx]

        if spd_dataset_ndx is not None:
            spd_dataset = self.data_source.datasets[spd_dataset_ndx]
            n_spd_images = min(len(spd_dataset), int(0.5 * n_ball_images))
            spd_samples_ndx = random.sample(range(len(spd_dataset)), k=n_spd_images)
            if spd_dataset_ndx > 0:
                # Add sizes of previous datasets to create cummulative indexes
                spd_samples_ndx = [e + self.data_source.cummulative_sizes[spd_dataset_ndx - 1] for e in spd_samples_ndx]
        else:
            n_spd_images = 0
            spd_samples_ndx = []

        self.sample_ndx = issia_samples_ndx + spd_samples_ndx
        random.shuffle(self.sample_ndx)

    def __iter__(self):
        self.generate_samples()         # Re-generate samples every epoch
        for ndx in self.sample_ndx:
            yield ndx

    def __len(self):
        return len(self.sample_ndx)


if __name__ == '__main__':
    from util import utils

    data_params = {
        'issia_path': utils.get_issia_dataset_path(),
        'spd_path': utils.get_soccer_player_detection_dataset_path(),
        'issia_train_cameras': [1],
        'spd_train_cameras': [1],
        'issia_val_cameras': [],
        'batch_size': 8,
        'num_workers': 0,
        'image_size': (720, 1280)
    }
    dataloaders = make_dataloaders(data_params)
    count = 0
    for ndx, (images, boxes, labels) in enumerate(dataloaders['train']):
        count += 1
        print(count)