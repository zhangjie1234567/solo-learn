import torch

from solo.losses.lejepa import SIGRegLoss, lejepa_invariance_loss


def test_sigreg_and_invariance_are_finite():
    projected = torch.randn(4, 32, 16, requires_grad=True)
    sigreg = SIGRegLoss(num_slices=8, num_knots=9)
    loss = lejepa_invariance_loss(projected) + sigreg(projected)
    assert torch.isfinite(loss)
    loss.backward()
    assert projected.grad is not None


def test_sigreg_prefers_standard_normal_samples():
    sigreg = SIGRegLoss(num_slices=64, num_knots=17)
    samples = torch.randn(2, 512, 8)
    standard = sigreg(samples)
    sigreg.global_step.zero_()
    shifted = sigreg(samples + 3)
    assert shifted > standard
