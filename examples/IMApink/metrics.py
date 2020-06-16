import os
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt


def confusion_matrix(y_pred, y_true):
    """calculate confusion matrix(tp, fp, fn)
    args: shape=(N, H, W)
    return: dict(tp, fp, fn)
    """
    assert len(y_pred.shape) == 3, f'got {y_pred.shape}'
    assert len(y_true.shape) == 3, f'got {y_true.shape}'
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    return {'tp': tp, 'fp': fp, 'fn': fn}


def calculate_dice_per_class(dataloader, num_classes, model, device):
    tp, fp, fn = [np.zeros(num_classes) for _ in range(3)]
    for batch_x, batch_y in tqdm(dataloader):
        x_tensor = batch_x.to(device)
        y_pred = model.predict(x_tensor)  # N,C,H,W
        y_pred = y_pred.cpu().round()  # GPU->CPU

        # 各classごとに、batchでまとめてtp,fp,fnを計算
        for c in range(batch_y.shape[1]):
            # N,H,W
            conf = confusion_matrix(y_pred[:, c], batch_y[:, c])
            tp[c] += conf['tp']
            fp[c] += conf['fp']
            fn[c] += conf['fn']

    # classの次元のままbroadcast
    dice_per_class = 2 * tp / (2 * tp + fp + fn)
    np.set_printoptions(precision=4)
    print(f"validation dice: {dice_per_class}")

    return dice_per_class

