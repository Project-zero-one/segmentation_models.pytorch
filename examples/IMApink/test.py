import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import Dataset
import losses
from metrics import calculate_dice
from utils import visualize, denormalize, get_preprocessing


def resize():
    transform = [
        albu.Resize(height=256, width=512)
    ]
    return albu.Compose(transform)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    ### config ###
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = {'IMApink': 1, 'IMAroot': 1}
    N_CLASSES = 1 + 1
    DATA_DIR = '/data/input/IMA_root'
    SEASON = 'season5*'
    NUM_VIS = 10

    DEVICE = 'cuda'

    ### main ###
    x_test_dir = os.path.join(DATA_DIR, 'validation', SEASON, '*/movieframe')
    y_test_dir = os.path.join(DATA_DIR, 'validation', SEASON, '*/label')

    # load best saved checkpoint
    best_model = torch.load('./best_model.pth')

    # create loss function and metrics
    loss = losses.CategoricalDiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0]),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0]),
    ]

    # create test dataset
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        binary_output=True if N_CLASSES == 1 else False,
    )
    test_loader = DataLoader(test_dataset)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    # logs = test_epoch.run(test_loader)
    dice_of_all = calculate_dice(test_loader, best_model, DEVICE, ignore_channels=[0])

    # visualize test data
    image_batch = []
    gt_mask_batch = []
    for i in range(NUM_VIS):
        n = np.random.choice(len(test_dataset))
        image, gt_mask = test_dataset[n]
        image_batch.append(image)
        gt_mask_batch.append(gt_mask)
    image_batch = np.array(image_batch)
    gt_mask_batch = np.array(gt_mask_batch)

    x_tensor = torch.from_numpy(image_batch).to(DEVICE)
    pr_mask_batch = best_model.predict(x_tensor)

    pr_mask_batch = pr_mask_batch.cpu().numpy().round()
    image_batch = image_batch.transpose(0, 2, 3, 1)

    visualize(
        image=denormalize(image_batch),
        ground_truth_mask=np.argmax(gt_mask_batch, axis=1),
        predicted_mask=np.argmax(pr_mask_batch, axis=1)
    )
