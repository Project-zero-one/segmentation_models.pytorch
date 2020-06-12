import torch
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.utils import base

from utils import take_channels


class CategoricalDiceLoss(base.Loss):
    def __init__(self, class_weights=None, eps=1e-7, beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, pr, gt):
        """
        :param pr: shape=(N, C, H, W)
        :param gt: shape=(N, C, H, W)
        :return : shape=(1,)
        """
        pr = self.activation(pr)

        return categorical_dice_loss(
            pr, gt,
            self.class_weights,
            self.eps, self.beta,
            self.ignore_channels
        )


class CategoricalFocalLoss(base.Loss):
    r"""Creates a criterion that measures the Categorical Focal Loss between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \cdot \alpha \cdot (1 - pr)^\gamma \cdot \log(pr)
    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    Returns:
        A callable ``categorical_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    """

    def __init__(self, alpha=0.25, gamma=2., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt):
        pr = self.activation(pr)
        return categorical_focal_loss(
            pr, gt,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )


class CategoricalFocalDiceLoss(base.Loss):
    def __init__(
        self,
        factor=0.5,
        alpha=0.25, gamma=2.,
        class_weights=None, beta=1.,
        eps=1e-7, activation=None, ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # focalとdiceの足し合わせる比率
        self.factor = factor
        # focal loss
        self.alpha = alpha
        self.gamma = gamma
        # dice loss
        self.class_weights = class_weights
        self.beta = beta
        # 共通
        self.eps = eps
        self.activation = base.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt):
        pr = self.activation(pr)

        focal_loss = categorical_focal_loss(pr, gt, self.gamma, self.alpha, self.ignore_channels, self.eps)
        dice_loss = categorical_dice_loss(pr, gt, self.class_weights, self.eps, self.beta, self.ignore_channels)

        return self.factor * focal_loss + (1 - self.factor) * dice_loss


class BinaryFocalLoss(base.Loss):
    r"""Creates a criterion that measures the Binary Focal Loss between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \alpha (1 - pr)^\gamma \log(pr) - (1 - gt) \alpha pr^\gamma \log(1 - pr)
    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
    Returns:
        A callable ``binary_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    Example:
    .. code:: python
        loss = BinaryFocalLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, alpha=0.25, gamma=2., activation=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.activation = base.Activation(activation)

    def forward(self, pr, gt):
        pr = self.activation(pr)
        return binary_focal_loss(
            pr, gt,
            alpha=self.alpha,
            gamma=self.gamma
        )


class BinaryFocalDiceLoss(base.Loss):
    def __init__(
        self,
        factor=0.5,
        alpha=0.25, gamma=2.,
        beta=1.,
        eps=1e-7, activation=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # focalとdiceの足し合わせる比率
        self.factor = factor
        # focal loss
        self.alpha = alpha
        self.gamma = gamma
        # dice loss
        self.beta = beta
        # 共通
        self.eps = eps
        self.activation = base.Activation(activation)

    def forward(self, pr, gt):
        pr = self.activation(pr)

        focal_loss = binary_focal_loss(pr, gt, self.gamma, self.alpha, self.eps)
        dice_loss = binary_dice_loss(pr, gt, self.eps, self.beta)

        return self.factor * focal_loss + (1 - self.factor) * dice_loss


def categorical_dice_loss(pr, gt, class_weights=None, eps=1e-7, beta=1., ignore_channels=None):
    # class_weightsがない場合は1にする
    if class_weights is None:
        class_weights = torch.ones(pr.shape[1]).to(pr.device)

    dice_loss = 0
    for c in range(pr.shape[1]):
        if ignore_channels is None or not c in ignore_channels:
            dice_loss += class_weights[c] * (
                1 - F.f_score(
                    pr[:, c], gt[:, c],
                    beta=beta,
                    eps=eps,
                    threshold=None,
                )
            )
    return dice_loss


def categorical_focal_loss(pr, gt, gamma=2.0, alpha=0.25, ignore_channels=None, eps=1e-7, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        gt: ground truth 4D tensor (B, C, H, W)
        pr: prediction 4D tensor (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        ignore_channels: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """

    gt, pr = take_channels(gt, pr, ignore_channels=ignore_channels, **kwargs)
    # clip to prevent NaN's and Inf's
    pr = torch.clamp(pr, eps, 1.0 - eps)
    # Calculate focal loss
    loss = - gt * (alpha * torch.pow((1 - pr), gamma) * torch.log(pr + eps))

    return torch.mean(loss)


def binary_focal_loss(pr, gt, gamma=2.0, alpha=0.25, eps=1e-7, **kwargs):
    r"""Implementation of Focal Loss from the paper in binary classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)
    Args:
        gt: ground truth 4D tensor (B, C, H, W)
        pr: prediction 4D tensor (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
    """
    # clip to prevent NaN's and Inf's
    pr = torch.clamp(pr, eps, 1.0 - eps)
    # Calculate focal loss
    loss_1 = - gt * (alpha * torch.pow((1 - pr), gamma) * torch.log(pr + eps))
    loss_0 = - (1 - gt) * ((1 - alpha) * torch.pow(pr, gamma) * torch.log(1 - pr + eps))

    return torch.mean(loss_0 + loss_1)


def binary_dice_loss(pr, gt, beta=1., eps=1e-7, **kwargs):
    return 1 - F.f_score(
        pr, gt,
        beta,
        eps,
        threshold=None,
    )
