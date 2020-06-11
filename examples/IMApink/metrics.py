import torch
from tqdm import tqdm
from utils import take_channels


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


def calculate_dice(dataloader, model, device, ignore_channels=None):
    tp, fp, fn = 0, 0, 0
    for batch_x, batch_y in tqdm(dataloader):
        x_tensor = batch_x.to(device)
        y_pred = model.predict(x_tensor)
        # 背景を計算に入れない
        y_pred, y_true = take_channels(y_pred, batch_y, ignore_channels=ignore_channels)
        # channel次元を消す
        y_pred = y_pred.squeeze(1).cpu().round()
        y_true = y_true.squeeze(1)

        conf = confusion_matrix(y_pred, y_true)
        tp += conf['tp']
        fp += conf['fp']
        fn += conf['fn']

    dice_of_all = 2 * tp / (2 * tp + fp + fn)
    print(f"validation dice: {dice_of_all:.4f}")

    return dice_of_all
