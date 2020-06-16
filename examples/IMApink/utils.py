import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as albu


def take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    col = len(images)
    row = [len(v) for v in images.values()][0]
    plt.figure(figsize=(15, 3 * row))
    for b in range(row):
        for i, (name, image) in enumerate(images.items(), 1):
            plt.subplot(row, col, b * 3 + i)
            # plt.xticks([])
            # plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image[b])
    plt.savefig('test.png')


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def plot_logs(logs: dict, objective: str, save_path: str):
    plt.figure()
    plt.plot(logs['epoch'], logs[objective], label='train_' + objective)
    plt.plot(logs['epoch'], logs['val_' + objective], label='val_' + objective)
    plt.grid()
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(objective)
    plt.savefig(os.path.join(save_path, objective + '.png'))
    plt.close()


palettes = [0, 0, 0,
            128, 0, 0,
            0, 128, 0,
            128, 128, 0,
            0, 0, 128,
            128, 0, 128,
            0, 128, 128,
            128, 128, 128,
            64, 0, 0,
            192, 0, 0,
            64, 128, 0,
            192, 128, 0,
            64, 0, 128,
            192, 0, 128,
            64, 128, 128,
            192, 128, 128,
            0, 64, 0,
            128, 64, 0,
            0, 192, 0,
            128, 192, 0,
            0, 64, 128,
            128, 64, 128,
            0, 192, 128,
            128, 192, 128,
            64, 64, 0,
            192, 64, 0,
            64, 192, 0,
            192, 192, 0,
            64, 64, 128,
            192, 64, 128,
            64, 192, 128,
            192, 192, 128,
            0, 0, 64,
            128, 0, 64,
            0, 128, 64,
            128, 128, 64,
            0, 0, 192,
            128, 0, 192,
            0, 128, 192,
            128, 128, 192,
            64, 0, 64,
            192, 0, 64,
            64, 128, 64,
            192, 128, 64,
            64, 0, 192,
            192, 0, 192,
            64, 128, 192,
            192, 128, 192,
            0, 64, 64,
            128, 64, 64,
            0, 192, 64,
            128, 192, 64,
            0, 64, 192,
            128, 64, 192,
            0, 192, 192,
            128, 192, 192,
            64, 64, 64,
            192, 64, 64,
            64, 192, 64,
            192, 192, 64,
            64, 64, 192,
            192, 64, 192,
            64, 192, 192,
            192, 192, 192,
            32, 0, 0,
            160, 0, 0,
            32, 128, 0,
            160, 128, 0,
            32, 0, 128,
            160, 0, 128,
            32, 128, 128,
            160, 128, 128,
            96, 0, 0,
            224, 0, 0,
            96, 128, 0,
            224, 128, 0,
            96, 0, 128,
            224, 0, 128,
            96, 128, 128,
            224, 128, 128,
            32, 64, 0,
            160, 64, 0,
            32, 192, 0,
            160, 192, 0,
            32, 64, 128,
            160, 64, 128,
            32, 192, 128,
            160, 192, 128,
            96, 64, 0,
            224, 64, 0,
            96, 192, 0,
            224, 192, 0,
            96, 64, 128,
            224, 64, 128,
            96, 192, 128,
            224, 192, 128,
            32, 0, 64,
            160, 0, 64,
            32, 128, 64,
            160, 128, 64,
            32, 0, 192,
            160, 0, 192,
            32, 128, 192,
            160, 128, 192,
            96, 0, 64,
            224, 0, 64,
            96, 128, 64,
            224, 128, 64,
            96, 0, 192,
            224, 0, 192,
            96, 128, 192,
            224, 128, 192,
            32, 64, 64,
            160, 64, 64,
            32, 192, 64,
            160, 192, 64,
            32, 64, 192,
            160, 64, 192,
            32, 192, 192,
            160, 192, 192,
            96, 64, 64,
            224, 64, 64,
            96, 192, 64,
            224, 192, 64,
            96, 64, 192,
            224, 64, 192,
            96, 192, 192,
            224, 192, 192,
            0, 32, 0,
            128, 32, 0,
            0, 160, 0,
            128, 160, 0,
            0, 32, 128,
            128, 32, 128,
            0, 160, 128,
            128, 160, 128,
            64, 32, 0,
            192, 32, 0,
            64, 160, 0,
            192, 160, 0,
            64, 32, 128,
            192, 32, 128,
            64, 160, 128,
            192, 160, 128,
            0, 96, 0,
            128, 96, 0,
            0, 224, 0,
            128, 224, 0,
            0, 96, 128,
            128, 96, 128,
            0, 224, 128,
            128, 224, 128,
            64, 96, 0,
            192, 96, 0,
            64, 224, 0,
            192, 224, 0,
            64, 96, 128,
            192, 96, 128,
            64, 224, 128,
            192, 224, 128,
            0, 32, 64,
            128, 32, 64,
            0, 160, 64,
            128, 160, 64,
            0, 32, 192,
            128, 32, 192,
            0, 160, 192,
            128, 160, 192,
            64, 32, 64,
            192, 32, 64,
            64, 160, 64,
            192, 160, 64,
            64, 32, 192,
            192, 32, 192,
            64, 160, 192,
            192, 160, 192,
            0, 96, 64,
            128, 96, 64,
            0, 224, 64,
            128, 224, 64,
            0, 96, 192,
            128, 96, 192,
            0, 224, 192,
            128, 224, 192,
            64, 96, 64,
            192, 96, 64,
            64, 224, 64,
            192, 224, 64,
            64, 96, 192,
            192, 96, 192,
            64, 224, 192,
            192, 224, 192,
            32, 32, 0,
            160, 32, 0,
            32, 160, 0,
            160, 160, 0,
            32, 32, 128,
            160, 32, 128,
            32, 160, 128,
            160, 160, 128,
            96, 32, 0,
            224, 32, 0,
            96, 160, 0,
            224, 160, 0,
            96, 32, 128,
            224, 32, 128,
            96, 160, 128,
            224, 160, 128,
            32, 96, 0,
            160, 96, 0,
            32, 224, 0,
            160, 224, 0,
            32, 96, 128,
            160, 96, 128,
            32, 224, 128,
            160, 224, 128,
            96, 96, 0,
            224, 96, 0,
            96, 224, 0,
            224, 224, 0,
            96, 96, 128,
            224, 96, 128,
            96, 224, 128,
            224, 224, 128,
            32, 32, 64,
            160, 32, 64,
            32, 160, 64,
            160, 160, 64,
            32, 32, 192,
            160, 32, 192,
            32, 160, 192,
            160, 160, 192,
            96, 32, 64,
            224, 32, 64,
            96, 160, 64,
            224, 160, 64,
            96, 32, 192,
            224, 32, 192,
            96, 160, 192,
            224, 160, 192,
            32, 96, 64,
            160, 96, 64,
            32, 224, 64,
            160, 224, 64,
            32, 96, 192,
            160, 96, 192,
            32, 224, 192,
            160, 224, 192,
            96, 96, 64,
            224, 96, 64,
            96, 224, 64,
            224, 224, 64,
            96, 96, 192,
            224, 96, 192,
            96, 224, 192,
            224, 224, 192]
