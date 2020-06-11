import os
import numpy as np
import cv2
from glob import glob
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = {'background': 0, 'IMAroot': 110, 'IMApink': 112}

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            binary_output=False
    ):
        self.images_fps = glob(os.path.join(images_dir, '*.png'))
        self.masks_fps = glob(os.path.join(masks_dir, '*.png'))

        # convert str names to class values on masks
        if type(classes) == list:
            # classname mapが名前だけの場合
            self.class_values = [self.CLASSES[cls] for cls in classes]
        elif type(classes) == dict:
            # classname mapが名前とclass idの場合
            self.class_values = {self.CLASSES[cls]: v for cls, v in classes.items()}
        else:
            assert "classes's type dict or list only"

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.binary_output = binary_output

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask
        if type(self.class_values) == list:
            # classname mapが名前だけの場合
            masks = [(mask == k) for k in self.class_values]

        elif type(self.class_values) == dict:
            # classname mapが名前とclass idの場合
            for k, v in self.class_values.items():
                # kをvに置き換える
                mask[mask == k] = v
            # TODO: np.unique(list(self.class_values.values())) をスッキリさせたい
            masks = [(mask == v) for v in np.unique(list(self.class_values.values()))]

        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1 or not self.binary_output:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            # argmax時にbackgroundを下敷きにする
            mask = np.concatenate((background, mask), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        assert len(self.images_fps) == len(self.masks_fps)
        return len(self.masks_fps)
