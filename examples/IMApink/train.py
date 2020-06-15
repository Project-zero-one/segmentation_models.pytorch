import os
from dataclasses import dataclass, field, asdict
import typing
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import Dataset
import losses
from metrics import calculate_dice
from utils import get_preprocessing


def get_training_augmentation(mean, std):
    train_transform = [
        albu.Resize(height=360, width=640),
        albu.RandomCrop(height=256, width=512, always_apply=True),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.Resize(height=360, width=640),
        albu.CenterCrop(height=256, width=512, always_apply=True)
    ]
    return albu.Compose(test_transform)


def resize():
    transform = [
        albu.Resize(height=256, width=512)
    ]
    return albu.Compose(transform)


def main(config):
    ### main ###
    x_train_dir = os.path.join(config.DATA_DIR, 'train', config.SEASON, '*/movieframe')
    y_train_dir = os.path.join(config.DATA_DIR, 'train', config.SEASON, '*/label')

    x_valid_dir = os.path.join(config.DATA_DIR, 'validation', config.SEASON, '*/movieframe')
    y_valid_dir = os.path.join(config.DATA_DIR, 'validation', config.SEASON, '*/label')

    # create segmentation model with pretrained encoder
    model = getattr(smp, config.MODEL)(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=config.N_CLASSES,
        activation=config.ACTIVATION,
    )
    # model = torch.nn.DataParallel(model)

    # create loss function
    # dice_loss = losses.CategoricalDiceLoss()
    # focal_loss = losses.CategoricalFocalLoss(gamma=5.0)
    # focal_dice_loss = losses.CategoricalFocalDiceLoss(factor=config.LAMBDA, gamma=5.0)
    # focal_dice_loss = losses.BinaryFocalDiceLoss(factor=LAMBDA, gamma=5.0)
    loss = getattr(losses, config.LOSS)(
        **config.loss_params
    )

    # set metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0] if config.N_CLASSES != 1 else None),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0] if config.N_CLASSES != 1 else None),
    ]

    # set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(),
             lr=config.LR),
    ])

    # create Dataset and DataLoader
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
        binary_output=True if config.N_CLASSES == 1 else False,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
        binary_output=True if config.N_CLASSES == 1 else False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    # train model for N epochs
    max_score = 0

    for i in range(0, config.EPOCHS):

        print('\nEpoch: {}'.format(i + 1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        ### callbacks (save model, change lr, etc.) ###
        # calcurate dice on all validation data
        dice_of_all = calculate_dice(
            valid_loader,
            model, config.DEVICE,
            ignore_channels=[0] if config.N_CLASSES != 1 else None
        )

        # model checkpoint
        if max_score < dice_of_all:
            max_score = dice_of_all
            torch.save(model, f'{config.MODEL_SAVE_PATH}.pth')
            print('Model saved.')

        # learning rate schedule
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5.')


@dataclass
class Config:
    ### config ###
    DATA_DIR: str = '/data/input/IMA_root'
    SEASON: str = 'season5*'
    CONFIG_PATH: str = 'parameters.yml'
    MODEL_SAVE_PATH: str = 'deeplabv3p-resnest269_multiclass'

    # The type of CLASSES is `dict` or `list`
    CLASSES = {'IMApink': 1, 'IMAroot': 1}

    # background(`1`) + objects(`len(CLASSES)`)
    N_CLASSES: int = 1 + 1

    MODEL: str = 'DeepLabV3Plus'
    ENCODER: str = 'resnest269'
    ENCODER_WEIGHTS: str = 'imagenet'
    LOSS: str = 'CategoricalFocalDiceLoss'
    loss_params = {
        "factor": 0.5,  # dice * factor + focal * (1 - factor)
        "gamma": 5.0,  # focal loss
    }

    BATCH_SIZE: int = 8
    LR: float = 0.0001
    CLASS_WEIGHTS = None  # [1.0, 1.0]
    EPOCHS: int = 30

    # could be `sigmoid` for binary class or None for logits or 'softmax2d' for multicalss segmentation
    ACTIVATION: str = 'sigmoid' if N_CLASSES == 1 else 'softmax2d'
    DEVICE: str = 'cuda'


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    ### main ###
    config = Config()
    # save train params as yaml
    with open(config.CONFIG_PATH, 'w') as fw:
        print(asdict(config))
        fw.write(yaml.dump(asdict(config)))
    # fit
    main(config)
