import os
from dataclasses import dataclass, field, asdict
from typing import List, Union, Optional
import yaml

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import Dataset
import losses
from metrics import calculate_dice_per_class, plot_logs
from utils import get_preprocessing
from augmentations import CutOff


def get_training_augmentation():
    train_transform = [
        CutOff(width=1280, always_apply=True),
        albu.Resize(height=320, width=512),  # or 350, 560
        albu.RandomCrop(height=256, width=512, always_apply=True),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        CutOff(width=1280, always_apply=True),
        albu.Resize(height=320, width=512),  # or 350, 560
        albu.CenterCrop(height=256, width=512, always_apply=True)
    ]
    return albu.Compose(test_transform)


def resize():
    transform = [
        albu.Resize(height=256, width=512)
    ]
    return albu.Compose(transform)


def main(config):
    # set dataset path
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
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=config.CLASSES,
        binary_output=True if config.N_CLASSES == 1 else False,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
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
    # it is a simple loop of iterating over dataloader's samples
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

    # train model
    loggers = {
        'epoch': [],
        'loss': [],
        'val_loss': [],
        'metrics': [],
        'val_metrics': [],
    }

    max_score = 0
    dice_per_class = np.zeros((1, config.N_CLASSES))  # validation用

    for i in range(0, config.EPOCHS):

        print('\nEpoch: {}'.format(i + 1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        ## callbacks (save model, change lr, etc.) ##
        # calculate dice on all validation data respectively classes
        dice_per_class = np.append(
            dice_per_class,
            calculate_dice_per_class(
                valid_loader,
                config.N_CLASSES,
                model, config.DEVICE,
            ),
            axis=0
        )

        # model checkpoint
        if max_score < dice_per_class[i, 1:].mean():
            # 現epochに置いて背景を除いた全クラスの平均値を超えたらmodelを保存
            max_score = dice_per_class[i, 1:].mean()
            torch.save(model, os.path.join(config.RESULT_DIR, 'best_model.pth'))
            print('Model saved.')

        # learning rate schedule
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5.')

        ## logger ##
        # TODO: 冗長すぎる
        loggers['epoch'].append(i)
        loggers['loss'].append(train_logs['loss'])
        loggers['val_loss'].append(valid_logs['loss'])
        loggers['metrics'].append(train_logs['fscore'])
        loggers['val_metrics'].append(valid_logs['fscore'])

        # plot logs
        plot_logs(loggers, 'loss', config.RESULT_DIR)

        # plot validation dice coef. per class
        plt.figure()
        for c in range(dice_per_class.shape[1]):
            plt.plot(loggers['epoch'], dice_per_class[:, c], label=str(c))
        plt.legend()
        plt.savefig(os.path.join(config.RESULT_DIR, 'dice_per_class.png'))

    # save last epoch model
    torch.save(model, os.path.join(config.RESULT_DIR, 'last_model.pth'))


@dataclass
class Config:
    RESULT_DIR: str = 'results/cv'

    DATA_DIR: str = '/data/input/IMA_root'
    SEASON: str = 'season5*'

    CLASSES: Union[list, dict] = field(default_factory=dict)
    N_CLASSES: int = 1 + 1

    MODEL: str = 'DeepLabV3Plus'
    ENCODER: str = 'resnest269'
    ENCODER_WEIGHTS: str = 'imagenet'
    LOSS: str = 'CategoricalFocalDiceLoss'
    loss_params: dict = field(default_factory=dict)

    BATCH_SIZE: int = 8
    LR: float = 0.0001
    CLASS_WEIGHTS: Optional[List[float]] = None
    EPOCHS: int = 30

    ACTIVATION: str = 'sigmoid' if N_CLASSES == 1 else 'softmax2d'
    DEVICE: str = 'cuda'

    def __post_init__(self):
        self.CLASSES = {'IMApink': 1, 'IMAroot': 1}
        # class weights of loss func
        # self.CLASS_WEIGHTS = [1.0, 1.0]
        self.loss_params = {
            "factor": 0.5,  # dice * factor + focal * (1 - factor)
            "gamma": 5.0,  # focal loss
        }


if __name__ == "__main__":
    from pprint import pprint
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    config = Config()
    # make save directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    # save train params as yaml
    with open('parameters.yml', 'w') as fw:
        pprint(asdict(config))  # show config
        fw.write(os.path.join(config.RESULT_DIR, yaml.dump(asdict(config))))
    # fit
    main(config)
