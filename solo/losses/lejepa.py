# Copyright 2026 solo-learn development team.
#
# LeJEPA algorithm reference: https://github.com/galilai-group/lejepa
# This is an independent implementation of the published SIGReg objective.

import torch
import torch.nn as nn
from solo.utils.misc import gather


class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularization using random one-dimensional slices."""

    def __init__(
        self,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
    ):
        super().__init__()
        assert num_slices > 0
        assert num_knots > 1 and num_knots % 2 == 1
        assert t_max > 0
        self.num_slices = num_slices

        t = torch.linspace(0, t_max, num_knots, dtype=torch.float32)
        step = t_max / (num_knots - 1)
        quadrature = torch.full((num_knots,), 2 * step, dtype=torch.float32)
        quadrature[[0, -1]] = step
        gaussian_cf = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", quadrature * gaussian_cf)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Evaluates SIGReg on tensors shaped ``[..., samples, dimensions]``."""

        if embeddings.ndim < 2:
            raise ValueError("SIGReg expects [..., samples, dimensions] input.")

        # Use global samples under DDP and float32 characteristic functions for numerical
        # stability under mixed precision. gather retains gradients.
        sample_dim = embeddings.ndim - 2
        embeddings = gather(embeddings, dim=sample_dim).float()
        num_samples, embedding_dim = embeddings.shape[-2:]

        with torch.no_grad():
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed(int(self.global_step.item()))
            directions = torch.randn(
                embedding_dim,
                self.num_slices,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
            directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
            self.global_step.add_(1)

        projected = embeddings @ directions
        phases = projected.unsqueeze(-1) * self.t
        empirical_real = phases.cos().mean(dim=-3)
        empirical_imag = phases.sin().mean(dim=-3)
        error = (empirical_real - self.gaussian_cf).square() + empirical_imag.square()
        statistic = (error @ self.weights) * num_samples
        return statistic.mean()


def lejepa_invariance_loss(projected_views: torch.Tensor) -> torch.Tensor:
    """Pulls every view towards the per-image mean representation."""

    view_mean = projected_views.mean(dim=0, keepdim=True)
    return (projected_views - view_mean).square().mean()
