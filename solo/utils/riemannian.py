# Copyright 2026 solo-learn development team.

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer

from solo.utils.muon import _adam_update


def _matrix_view(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.reshape(parameter.shape[0], -1)


def project_stiefel_tangent(
    weight: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    """Projects a matrix gradient onto the tangent space of a (row/column) Stiefel point."""

    weight_matrix = _matrix_view(weight)
    gradient_matrix = _matrix_view(gradient)
    transposed = weight_matrix.size(0) < weight_matrix.size(1)
    if transposed:
        weight_matrix = weight_matrix.transpose(0, 1)
        gradient_matrix = gradient_matrix.transpose(0, 1)
    symmetric = 0.5 * (
        weight_matrix.transpose(0, 1) @ gradient_matrix
        + gradient_matrix.transpose(0, 1) @ weight_matrix
    )
    projected = gradient_matrix - weight_matrix @ symmetric
    if transposed:
        projected = projected.transpose(0, 1)
    return projected.reshape_as(gradient)


def stiefel_retraction(weight: torch.Tensor) -> torch.Tensor:
    """Retracts a matrix to the closest reduced-QR Stiefel point."""

    original_shape = weight.shape
    matrix = _matrix_view(weight)
    transposed = matrix.size(0) < matrix.size(1)
    work = matrix.transpose(0, 1) if transposed else matrix
    q, r = torch.linalg.qr(work, mode="reduced")
    # Fix the arbitrary QR column signs for a stable, continuous-enough retraction.
    diagonal = torch.diagonal(r, 0)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    matrix = q.transpose(0, 1) if transposed else q
    return matrix.reshape(original_shape)


class RiemannianAdam(Optimizer):
    """Adam with Stiefel tangent projection and QR retraction for matrix parameters.

    Groups with ``use_riemannian=True`` and ``manifold='stiefel'`` use the Riemannian update
    when their parameter has at least two dimensions. Vectors, scalars, and auxiliary groups
    use the same AdamW-style fallback as the Muon integration.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        manifold: str = "stiefel",
        retraction: bool = True,
        **kwargs,
    ):
        if manifold not in {"stiefel", "none"}:
            raise ValueError("manifold must be 'stiefel' or 'none'")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            manifold=manifold,
            retraction=retraction,
            use_riemannian=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_riemannian = group.get("use_riemannian", False)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1

                if (
                    use_riemannian
                    and group["manifold"] == "stiefel"
                    and parameter.ndim >= 2
                ):
                    gradient = project_stiefel_tangent(parameter, gradient)
                update = _adam_update(
                    gradient,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["betas"],
                    group["eps"],
                )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
                if (
                    use_riemannian
                    and group["manifold"] == "stiefel"
                    and group["retraction"]
                    and parameter.ndim >= 2
                ):
                    parameter.copy_(stiefel_retraction(parameter))
        return loss
