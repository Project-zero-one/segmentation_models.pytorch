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

    def forward(self, y_pred, y_true):
        """
        :param y_pred: shape=(N, C, H, W)
        :param y_true: shape=(N, C, H, W)
        :return : shape=(1,)
        """
        # class_weightsがない場合は1にする
        if self.class_weights is None:
            self.class_weights = torch.ones(y_pred.shape[1]).to(y_pred.device)

        y_pred = self.activation(y_pred)

        dice_loss = 0
        for c in range(y_pred.shape[1]):
            if self.ignore_channels is None or not c in self.ignore_channels:
                dice_loss += self.class_weights[c] * (
                    1 - F.f_score(
                        y_pred[:, c], y_true[:, c],
                        beta=self.beta,
                        eps=self.eps,
                        threshold=None,
                    )
                )
        return dice_loss


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
    Example:
        .. code:: python
            loss = CategoricalFocalLoss()
            model.compile('SGD', loss=loss)
    """

    def __init__(self, alpha=0.25, gamma=2., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_channels = ignore_channels

    def forward(self, gt, pr):
        return categorical_focal_loss(
            gt, pr,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )


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

    def __init__(self, alpha=0.25, gamma=2., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, gt, pr):
        return binary_focal_loss(
            gt, pr,
            alpha=self.alpha,
            gamma=self.gamma
        )


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, ignore_channels=None, eps=1e-7, **kwargs):
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


def binary_focal_loss(gt, pr, gamma=2.0, alpha=0.25, eps=1e-7, **kwargs):
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
