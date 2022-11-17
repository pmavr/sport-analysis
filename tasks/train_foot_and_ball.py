from time import time

import torch
import torch.optim as optim

from models.foot_and_ball.network.footandball import FootAndBall
from models.foot_and_ball.data.data_reader import make_dataloaders
from models.foot_and_ball.network.ssd_loss import SSDLoss
from models.foot_and_ball.network.iou_loss import IouLoss

from util import utils


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


def fit_model(model_params, opt_func, dataloaders,
              sched_func=None, num_of_epochs_until_save=1, history=None, save_path=''):
    # Weight for components of the loss function.
    # Ball-related loss and player-related loss are mean losses (loss per one positive example)
    alpha_l_player = 0.01
    alpha_c_player = 1.
    alpha_c_ball = 5.

    # Normalize weights
    total = alpha_l_player + alpha_c_player + alpha_c_ball
    alpha_l_player /= total
    alpha_c_player /= total
    alpha_c_ball /= total

    model = model_params['model']['method']
    num_of_epochs = model_params['train_epochs']

    # Loss function
    criterion = model_params['loss']['method']
    criterion = criterion.to(model.device)

    # Optimizer
    optimizer = opt_func['method']
    optimizer_to(optimizer, model.device)

    if history:
        hist = history
        num_of_trained_epochs = len(history['train_loss'])
    else:
        hist = {
            'train_loss': [],
            'train_loss_ball_c': [],
            'train_loss_player_c': [],
            'train_loss_player_l': [],
            'val_loss': [],
            'val_loss_ball_c': [],
            'val_loss_player_c': [],
            'val_loss_player_l': []}
        num_of_trained_epochs = 0
    num_of_epochs += num_of_trained_epochs

    # Training statistics
    training_stats = {'train': [], 'val': []}

    print('Training...')
    for epoch in range(num_of_trained_epochs, num_of_epochs):
        epoch_start_time = time()

        epoch_train_loss = 0.
        epoch_train_loss_ball_c = 0.
        epoch_train_loss_player_c = 0.
        epoch_train_loss_player_l = 0.
        epoch_train_batches = 0

        epoch_val_loss = 0.
        epoch_val_loss_ball_c = 0.
        epoch_val_loss_player_c = 0.
        epoch_val_loss_player_l = 0.
        epoch_val_batches = 0

        model.train()
        for ndx, (images, boxes, labels) in enumerate(dataloaders['train']):
            images = images.to(model.device)
            h, w = images.shape[-2], images.shape[-1]
            gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
            gt_maps = [e.to(model.device) for e in gt_maps]

            predictions = model(images)
            optimizer.zero_grad()
            loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)
            loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball
            loss.backward()
            optimizer.step()

            # statistics
            epoch_train_loss += loss.item()
            epoch_train_loss_ball_c += loss_c_ball.item()
            epoch_train_loss_player_c += loss_c_player.item()
            epoch_train_loss_player_l += loss_l_player.item()
            epoch_train_batches += 1

        model.eval()
        for ndx, (images, boxes, labels) in enumerate(dataloaders['val']):
            images = images.to(model.device)
            h, w = images.shape[-2], images.shape[-1]
            gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
            gt_maps = [e.to(model.device) for e in gt_maps]

            predictions = model(images)
            loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)
            loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball

            # statistics
            epoch_val_loss += loss.item()
            epoch_val_loss_ball_c += loss_c_ball.item()
            epoch_val_loss_player_c += loss_c_player.item()
            epoch_val_loss_player_l += loss_l_player.item()
            epoch_val_batches += 1

        epoch_train_loss /= epoch_train_batches
        epoch_train_loss_ball_c /= epoch_train_batches
        epoch_train_loss_player_c /= epoch_train_batches
        epoch_train_loss_player_l /= epoch_train_batches
        hist['train_loss'].append(epoch_train_loss)
        hist['train_loss_ball_c'].append(epoch_train_loss_ball_c)
        hist['train_loss_player_c'].append(epoch_train_loss_player_c)
        hist['train_loss_player_l'].append(epoch_train_loss_player_l)

        epoch_val_loss /= epoch_val_batches
        epoch_val_loss_ball_c /= epoch_val_batches
        epoch_val_loss_player_c /= epoch_val_batches
        epoch_val_loss_player_l /= epoch_val_batches
        hist['val_loss'].append(epoch_val_loss)
        hist['val_loss_ball_c'].append(epoch_val_loss_ball_c)
        hist['val_loss_player_c'].append(epoch_val_loss_player_c)
        hist['val_loss_player_l'].append(epoch_val_loss_player_l)

        epoch_duration = time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_of_epochs} | Duration: {epoch_duration:.2f}\n"
              f"Phase: train "
              f"| Loss total: {hist['train_loss'][-1]:.4f} "
              f"| Loss ball conf.: {hist['train_loss_ball_c'][-1]:.4f} "
              f"| Loss player conf.: {hist['train_loss_player_c'][-1]:.4f} "
              f"| Loss player loc.: {hist['train_loss_player_l'][-1]:.4f}")
        print(f"Phase: val "
              f"| Loss total: {hist['val_loss'][-1]:.4f} "
              f"| Loss ball conf.: {hist['val_loss_ball_c'][-1]:.4f} "
              f"| Loss player conf.: {hist['val_loss_player_c'][-1]:.4f} "
              f"| Loss player loc.: {hist['val_loss_player_l'][-1]:.4f}\n")

        if (epoch + 1) % num_of_epochs_until_save == 0:
            model_components = {'model': model,
                                'opt_func': optimizer}
            utils.save_model(model_components, hist,
                             f"{save_path}object_detector_{len(hist[next(iter(hist))])}.pth")
        if sched_func is not None:
            sched_func.step()
    return model, optimizer, hist


if __name__ == '__main__':
    import sys
    import os

    data_params = {
        'issia_path': utils.get_issia_dataset_path(),
        'spd_path': utils.get_soccer_player_detection_dataset_path(),
        'issia_train_cameras': [1, 2, 3, 4],
        'spd_train_cameras': [1, 2],
        'issia_val_cameras': [5, 6],
        'num_workers': 6,
        'batch_size': 10,
        'image_size': (720, 1280)
    }
    dataloaders = make_dataloaders(data_params)

    model_params = {
        'model': {'name': 'fab', 'method': FootAndBall.initialize(phase='train')},
        'loss': {'name': 'mse',
                 'method': SSDLoss(neg_pos_ratio=3, box_criterion=IouLoss(pred_mode='', losstype='Giou'))},
        'lr': 1e-3,
        'train_epochs': 22
    }
    optimizer = {
        'name': 'adam',
        'method': optim.Adam(model_params['model']['method'].parameters(), lr=model_params['lr'])
    }
    scheduler = {
        'name': 'multisteplr',
        'method': torch.optim.lr_scheduler.MultiStepLR(optimizer['method'], [8, 16, 20], gamma=0.1)
    }

    model_dir = f"{model_params['loss']['name']}_{data_params['batch_size']}_{model_params['lr']}/"
    save_path = f"{utils.get_footandball_model_path()}{model_dir}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # model, optimizer, history = FootAndBall.load_model(
    #     f'{utils.get_footandball_model_path()}object_detector_14.pth', model, optimizer, history=True)

    network, optimizer, history = fit_model(
        model_params=model_params,
        opt_func=optimizer,
        dataloaders=dataloaders,
        num_of_epochs_until_save=1,
        sched_func=scheduler,
        save_path=save_path)  # , history=history)

    model_components = {'model': network,
                        'opt_func': optimizer}
    utils.save_model(model_components, history,
                     f"{save_path}object_detector_{len(history[next(iter(history))])}.pth")

    sys.exit()
