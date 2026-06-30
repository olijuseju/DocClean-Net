"""
model/losses.py
===============
Combined loss for DocClean-Net U-Net training.

    CombinedLoss = MSE * 0.7 + (1 - SSIM) * 0.3

Rationale (do not change without reopening):
    - MSE penalises pixel-level intensity errors uniformly — good for
      recovering ink coverage and suppressing grid residuals.
    - SSIM penalises structural/perceptual differences — preserves stroke
      continuity and fine texture that MSE alone tends to blur.
    - No VGG/perceptual loss: keeps the project self-contained (no pretrained
      backbone required).

References:
    pytorch-msssim: https://github.com/VainF/pytorch-msssim
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CombinedLoss(nn.Module):
    """Weighted combination of MSE and (1 - SSIM).

    Args:
        alpha (float): Weight for MSE term. Weight for SSIM term is
                       ``1 - alpha``. Default: 0.7.
        data_range (float): Value range of input tensors. Default: 1.0
                            (tensors normalised to [0, 1]).

    Example::

        criterion = CombinedLoss()
        loss = criterion(pred, target)   # both shape (B, 1, H, W), float32

    Properties:
        - loss == 0.0 when pred == target (identical tensors)
        - loss  > 0.0 when pred != target
        - loss is differentiable everywhere w.r.t. pred
    """

    def __init__(self, alpha: float = 0.7, data_range: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.data_range = data_range
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            pred   (torch.Tensor): Model output, shape (B, 1, H, W),
                                   float32, values in [0, 1].
            target (torch.Tensor): Ground truth, shape (B, 1, H, W),
                                   float32, values in [0, 1].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        mse_loss = self.mse(pred, target)

        # ssim() returns a scalar in [0, 1]; higher = more similar.
        # win_size=7 default; size_average=True averages over the batch.
        ssim_val = ssim(
            pred,
            target,
            data_range=self.data_range,
            size_average=True,
        )
        ssim_loss = 1.0 - ssim_val

        return self.alpha * mse_loss + (1.0 - self.alpha) * ssim_loss
