# Copyright 2026 solo-learn development team.

import torch
import torch.nn.functional as F


def ijepa_loss_func(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the Smooth-L1 latent prediction objective used by I-JEPA."""

    return F.smooth_l1_loss(predictions, targets)
