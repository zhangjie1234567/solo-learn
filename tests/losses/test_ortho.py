import torch
import torch.nn as nn

from solo.losses.ortho import (
    calculate_ortho_loss,
    compute_soft_ortho,
    compute_srip,
    get_or_gamma,
    spectral_norm_power_iter,
)


def test_soft_ortho_and_srip_identity():
    identity = torch.eye(4, requires_grad=True)
    assert torch.allclose(compute_soft_ortho(identity), torch.zeros(()))
    assert torch.allclose(compute_srip(identity), torch.zeros(()))

    loss = compute_soft_ortho(identity) + compute_srip(identity)
    loss.backward()
    assert identity.grad is not None


def test_spectral_norm_power_iteration():
    matrix = torch.diag(torch.tensor([3.0, 1.0, 0.5]))
    value = spectral_norm_power_iter(matrix, power_iter=5)
    assert torch.allclose(value, torch.tensor(3.0), atol=1e-4)


def test_or_backbone_scope_and_gamma_lookup():
    encoder = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    loss = calculate_ortho_loss(encoder, reg_type="so")
    expected = compute_soft_ortho(encoder[0].weight.flatten(1).transpose(0, 1))
    expected = expected + compute_soft_ortho(encoder[2].weight.transpose(0, 1))
    assert torch.allclose(loss, expected)
    assert get_or_gamma("wide_resnet28w2", "srip") == 1e-4
    assert get_or_gamma("unknown_backbone", "so") == 1e-6
