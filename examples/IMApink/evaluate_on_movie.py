import os
import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import albumentations as albu
import segmentation_models_pytorch as smp

from dataset import VideoDataset
from utils import get_preprocessing, palettes


def resize():
    transform = [
        albu.Resize(height=256, width=512)
    ]
    return albu.Compose(transform)


class FPS(object):
    """ Calculate FPS.
        example) fps = FPS()
                 while(cap.isOpended()):
                     # Your processing
                     fps.calculate(draw)
                     cv2.imshow('test', draw)
    """

    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def calculate(self, draw, show=True):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time += exec_time
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        if show:
            cv2.rectangle(draw, (0, 0), (60, 20), (255, 255, 255), -1)
            cv2.putText(draw, self.fps, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        else:
            print(self.fps)


def make_predict_movie(
    dataloader, save_path,
    model, device,
    area_threshold=0,
):
    # for video writing(output)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (1280, 720))
    # for calculate frame per second
    fps = FPS()
    for raw_frame, input_image in tqdm(dataloader):
        # frame読み切ったら抜ける
        if raw_frame is None:
            break

        x_tensor = input_image.to(device)
        y_pred = model.predict(x_tensor)
        result = post_processing(y_pred, area_threshold)  # N,C,H,W -> N,H,W

        # Display the predicted frame
        # N,H,W,C -> H,W,C
        output = overlay(raw_frame.cpu().numpy().squeeze(0), result.squeeze(0))  # blend
        # output = surrounding(raw_frame.cpu().numpy().squeeze(0), result.squeeze(0))  # 枠線
        fps.calculate(output)  # video writeする前に必要
        out.write(output[..., ::-1])  # RGB -> BGR
    out.release()


def post_processing(output, area_threshold=0):
    # channel次元を消す
    output = output.cpu().numpy().round()  # GPU -> CPU
    output = output.transpose(0, 2, 3, 1)  # N,C,H,W -> N,H,W,C
    # N,H,W,C -> N,H,W
    mask = output.squeeze(3) if output.shape[3] == 1 else np.argmax(output, axis=-1)  # binaryではargmaxしない
    mask = mask.astype('uint8')
    # 小さい面積消去
    if not area_threshold:
        mask = delete_small_mask(mask, area_threshold)

    return mask


def overlay(frame, mask, area_threshold=0):
    """
    args: frame: shape=(720, 1280, 3)
         result: shape=(H, W, C)
    """
    # paletteを埋め込む
    height, width = frame.shape[:2]
    mask_pil = Image.fromarray(mask, mode="P")  # pallete形式で開く
    # mask_pil.putpalette(palettes)  # インデックスカラーで置き換え
    mask_pil.putpalette([0, 0, 0, 255, 255, 255])  # 背景:黒, 対象物:白
    mask_pil = mask_pil.resize((width, height))
    mask_rgb = np.array(mask_pil.convert("RGB"))  # cv2に戻す

    blend = cv2.addWeighted(src1=frame, alpha=0.9, src2=mask_rgb, beta=0.3, gamma=2.2)
    return blend


def surrounding(frame, mask):
    height, width = frame.shape[:2]
    # 元の画像が加工されてしまうのを防ぐ
    gray = mask.copy()
    gray = cv2.resize(gray, (width, height))
    output = frame.copy()
    # 2値化&島を認識
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return output

    for i in range(0, len(contours)):
        cv2.drawContours(output, contours, i, (0, 255, 0), 2)

    return output


def delete_small_mask(mask: np.array, threshold: int) -> np.array:
    gray = mask.copy()
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return mask

    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(gray, contours, i, 0, -1)
    return gray


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    ### config ###
    ENCODER = 'resnest269'
    ENCODER_WEIGHTS = 'imagenet'
    MODEL_PATH = 'pan-resnest269_categoricalloss.pth'
    VIDEO_PATH = '名称未設定.mp4'
    SAVE_PATH = '名称未設定(1).avi'

    DEVICE = 'cuda'
    AREA_THRESHOLD = 512 * 256 * 0.1
    ### main ###
    # load best saved checkpoint
    best_model = torch.load(MODEL_PATH)

    # create test dataset
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = VideoDataset(
        VIDEO_PATH,
        augmentation=resize(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    test_loader = DataLoader(test_dataset)

    make_predict_movie(test_loader, SAVE_PATH, best_model, DEVICE, AREA_THRESHOLD)
