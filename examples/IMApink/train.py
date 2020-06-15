import os
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


if __name__ == "__main__":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    ### config ###
    # The type of CLASSES is `dict` or `list`
    CLASSES = {'IMApink': 1, 'IMAroot': 1}

    # background(`1`) + objects(`len(CLASSES)`)
    N_CLASSES = 1 + 1

    DATA_DIR = '/data/input/IMA_root'
    SEASON = 'season5*'
    MODEL_SAVE_PATH = 'deeplabv3p-resnest269_multiclass'

    MODEL = 'DeepLabV3Plus'
    ENCODER = 'wide_resnet101_2'
    ENCODER_WEIGHTS = 'imagenet'
    BATCH_SIZE = 8
    LR = 0.0001
    CLASS_WEIGHTS = None  # [1.0, 1.0]
    LAMBDA = 0.5  # dice * LAMBDA + focal * (1 - LAMBDA)
    EPOCHS = 10

    # could be `sigmoid` for binary class or None for logits or 'softmax2d' for multicalss segmentation
    ACTIVATION = 'sigmoid' if N_CLASSES == 1 else 'softmax2d'
    DEVICE = 'cuda'

    ### main ###
    x_train_dir = os.path.join(DATA_DIR, 'train', SEASON, '*/movieframe')
    y_train_dir = os.path.join(DATA_DIR, 'train', SEASON, '*/label')

    x_valid_dir = os.path.join(DATA_DIR, 'validation', SEASON, '*/movieframe')
    y_valid_dir = os.path.join(DATA_DIR, 'validation', SEASON, '*/label')

    # create segmentation model with pretrained encoder
    model = getattr(smp, MODEL)(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=N_CLASSES,
        activation=ACTIVATION,
    )
    # model = torch.nn.DataParallel(model)

    # create loss function
    # dice_loss = losses.CategoricalDiceLoss()
    # focal_loss = losses.CategoricalFocalLoss(gamma=5.0)
    focal_dice_loss = losses.CategoricalFocalDiceLoss(factor=LAMBDA, gamma=5.0)
    # focal_dice_loss = losses.BinaryFocalDiceLoss(factor=LAMBDA, gamma=5.0)

    # set metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0] if N_CLASSES != 1 else None),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0] if N_CLASSES != 1 else None),
    ]

    # set optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LR),
    ])

    # create Dataset and DataLoader
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        binary_output=True if N_CLASSES == 1 else False,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        binary_output=True if N_CLASSES == 1 else False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
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
        loss=focal_dice_loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=focal_dice_loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for N epochs
    max_score = 0

    for i in range(0, EPOCHS):

        print('\nEpoch: {}'.format(i + 1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # callbacks (save model, change lr, etc.) #
        ## calcurate dice on all validation data ##
        dice_of_all = calculate_dice(valid_loader, model, DEVICE, ignore_channels=[0] if N_CLASSES != 1 else None)

        ## model checkpoint ##
        if max_score < dice_of_all:
            max_score = dice_of_all
            torch.save(model, f'{MODEL_SAVE_PATH}.pth')
            print('Model saved!')

        ## learning rate schedule ##
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
