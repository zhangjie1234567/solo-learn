# Copyright 2026 solo-learn development team.
#
# Muon is based on Keller Jordan's MIT-licensed implementation:
# https://github.com/KellerJordan/Muon

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


def zeropower_via_newton_schulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximates the orthogonal factor of a matrix with Newton-Schulz iterations."""

    if matrix.ndim < 2:
        raise ValueError("Muon orthogonalization expects a matrix or batched matrices.")
    if steps <= 0:
        raise ValueError("steps must be positive")

    a, b, c = 3.4445, -4.7750, 2.0315
    original_shape = matrix.shape
    x = matrix.reshape(matrix.shape[0], -1)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.transpose(0, 1)
    compute_dtype = torch.bfloat16 if x.is_cuda else torch.float32
    x = x.to(compute_dtype)
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.transpose(0, 1)
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.transpose(0, 1)
    return x.to(matrix.dtype).reshape(original_shape)


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp_(momentum_buffer, momentum) if nesterov else momentum_buffer
    update = zeropower_via_newton_schulz5(update, steps=ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas,
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias_correction1 = 1 - betas[0] ** step
    bias_correction2 = 1 - betas[1] ** step
    return (exp_avg / bias_correction1) / (
        (exp_avg_sq / bias_correction2).sqrt() + eps
    )


class Muon(Optimizer):
    """Muon for matrix hidden weights with AdamW fallback for auxiliary parameters.

    Parameter groups may set ``use_muon=True``. Matrix parameters in such groups use Muon;
    all other parameters use the internal AdamW-style update. This lets the existing
    solo-learn single-optimizer scheduler continue to work unchanged.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        **kwargs,
    ):
        if lr < 0:
            raise ValueError("Invalid learning rate")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            betas=betas,
            eps=eps,
            use_muon=False,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if use_muon and parameter.ndim >= 2:
                    state = self.state[parameter]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        momentum=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                else:
                    state = self.state[parameter]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss
