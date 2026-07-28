# Copyright 2026 solo-learn development team.
#
# Orthogonality Regularization is based on:
# "Preventing Dimensional Collapse in Self-Supervised Learning via Orthogonality
# Regularization", NeurIPS 2024.

from typing import Dict, Optional

import torch
import torch.nn as nn


_OR_GAMMA_TABLE: Dict[tuple, float] = {
    ("resnet18", "so"): 1e-6,
    ("resnet18", "srip"): 1e-3,
    ("resnet50", "so"): 1e-6,
    ("resnet50", "srip"): 1e-3,
    ("wideresnet28w2", "so"): 1e-6,
    ("wideresnet28w2", "srip"): 1e-4,
    ("vit-tiny", "so"): 1e-5,
    ("vit-small", "so"): 1e-5,
    ("vit-base", "so"): 1e-6,
}


def _normalize_backbone_name(backbone_name: str) -> str:
    name = str(backbone_name).lower().replace("_", "-")
    if name.startswith("wide-"):
        name = "wide" + name[5:]
    return name


def get_or_gamma(backbone_name: str, reg_type: str) -> float:
    """Returns the recommended OR coefficient for a backbone and regularizer."""

    reg_type = str(reg_type).lower()
    if reg_type not in {"so", "srip"}:
        raise ValueError(f"Unknown OR type '{reg_type}'. Choose from ('so', 'srip').")

    name = _normalize_backbone_name(backbone_name)
    if (name, reg_type) in _OR_GAMMA_TABLE:
        return _OR_GAMMA_TABLE[(name, reg_type)]
    return 1e-6 if reg_type == "so" else 1e-3


def _gram_deviation(w_2d: torch.Tensor) -> torch.Tensor:
    in_dim, out_dim = w_2d.shape
    if in_dim > out_dim:
        gram = w_2d.transpose(0, 1) @ w_2d
    else:
        gram = w_2d @ w_2d.transpose(0, 1)
    identity = torch.eye(gram.size(0), device=w_2d.device, dtype=w_2d.dtype)
    return gram - identity


def compute_soft_ortho(w_2d: torch.Tensor) -> torch.Tensor:
    """Computes squared Frobenius soft orthogonality error for ``[in_dim, out_dim]`` W."""

    if w_2d.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(w_2d.shape)}")
    return _gram_deviation(w_2d).pow(2).sum()


def spectral_norm_power_iter(matrix: torch.Tensor, power_iter: int = 5) -> torch.Tensor:
    """Approximates the spectral norm with differentiable power iterations."""

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}")
    if power_iter <= 0:
        raise ValueError("power_iter must be positive")

    vector = torch.ones(matrix.size(1), device=matrix.device, dtype=matrix.dtype)
    vector = vector / vector.norm().clamp_min(torch.finfo(matrix.dtype).eps)
    for _ in range(power_iter):
        vector = matrix.transpose(0, 1) @ (matrix @ vector)
        vector = vector / vector.norm().clamp_min(torch.finfo(matrix.dtype).eps)
    return (matrix @ vector).norm()


def compute_srip(w_2d: torch.Tensor, power_iter: int = 5) -> torch.Tensor:
    """Computes the SRIP spectral orthogonality error for ``[in_dim, out_dim]`` W."""

    if w_2d.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(w_2d.shape)}")
    return spectral_norm_power_iter(_gram_deviation(w_2d), power_iter=power_iter)


def _weight_to_2d(module: nn.Module) -> Optional[torch.Tensor]:
    if isinstance(module, nn.Conv2d):
        # [out_c, in_c, kh, kw] -> [in_c * kh * kw, out_c]
        return module.weight.flatten(1).transpose(0, 1)
    if isinstance(module, nn.Linear):
        # [out_dim, in_dim] -> [in_dim, out_dim]
        return module.weight.transpose(0, 1)
    return None


def calculate_ortho_loss(
    encoder_model: nn.Module,
    reg_type: str = "so",
    power_iter: int = 5,
    debug_print: bool = False,
) -> torch.Tensor:
    """Sums OR losses over Conv2d/Linear layers in an encoder only.

    The caller must pass the pure backbone module. Projectors, predictors, classifiers and
    output representations are intentionally outside this function's scope.
    """

    reg_type = str(reg_type).lower()
    if reg_type not in {"so", "srip"}:
        raise ValueError(f"Unknown OR type '{reg_type}'. Choose from ('so', 'srip').")

    total_loss = None
    for name, module in encoder_model.named_modules():
        weight = _weight_to_2d(module)
        if weight is None:
            continue
        layer_loss = (
            compute_soft_ortho(weight)
            if reg_type == "so"
            else compute_srip(weight, power_iter=power_iter)
        )
        if debug_print:
            print(f"OR[{reg_type}] {name or '<encoder>'}: {layer_loss.detach().item():.6e}")
        total_loss = layer_loss if total_loss is None else total_loss + layer_loss

    if total_loss is not None:
        return total_loss
    # Keep the returned scalar connected to the model device/dtype when an encoder has no
    # supported weighted layers, while still allowing callers to use it in a total loss.
    try:
        parameter = next(encoder_model.parameters())
    except StopIteration:
        return torch.zeros(())
    return parameter.sum() * 0.0
