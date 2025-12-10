import torch
import torch.nn as nn
from torch import Tensor


class TverskyLoss(nn.Module):
    """
    Implementation of the Tversky loss function for binary segmentation.

    This loss is a generalization of the Dice loss that allows asymmetric
    weighting of false positives and false negatives.

    Args:
        alpha (float): Weight for false negatives.
        beta (float): Weight for false positives.
        smooth (float): Small constant to avoid division by zero.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.smooth: float = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the Tversky loss.

        Args:
            logits (Tensor): Raw model outputs of shape (N, 1, H, W) or similar.
            targets (Tensor): Ground truth binary masks with the same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        probs: Tensor = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp: Tensor = (probs * targets).sum(dim=1)
        fp: Tensor = (probs * (1 - targets)).sum(dim=1)
        fn: Tensor = ((1 - probs) * targets).sum(dim=1)

        tversky: Tensor = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """
    Implementation of the Focal Tversky loss for highly imbalanced
    segmentation problems.

    This loss applies a focal modulation to the standard Tversky loss
    to focus training on hard examples.

    Attributes:
        initialized (bool): Class-level flag used to print the activation
                            message only once per training session.

    Args:
        alpha (float): Weight for false negatives.
        beta (float): Weight for false positives.
        gamma (float): Focusing parameter for focal modulation.
        smooth (float): Small constant to avoid division by zero.
    """

    initialized: bool = False

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.smooth: float = smooth

        # Print ONCE per training session
        if not FocalTverskyLoss.initialized:
            print(
                "[INFO] Focal Tversky Loss ACTIVATED: "
                f"alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f}"
            )
            FocalTverskyLoss.initialized = True

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the Focal Tversky loss.

        Args:
            logits (Tensor): Raw model outputs of shape (N, 1, H, W) or similar.
            targets (Tensor): Ground truth binary masks with the same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        probs: Tensor = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp: Tensor = (probs * targets).sum(dim=1)
        fp: Tensor = (probs * (1 - targets)).sum(dim=1)
        fn: Tensor = ((1 - probs) * targets).sum(dim=1)

        tversky: Tensor = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        loss: Tensor = (1 - tversky) ** self.gamma
        return loss.mean()

