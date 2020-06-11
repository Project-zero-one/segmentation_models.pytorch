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
