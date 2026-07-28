import torch

from solo.losses.ijepa import ijepa_loss_func
from solo.utils.jepa import MultiBlockMaskGenerator, apply_masks


def test_ijepa_loss_backward():
    pred = torch.randn(8, 4, 16, requires_grad=True)
    target = torch.randn(8, 4, 16)
    loss = ijepa_loss_func(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_jepa_masks_are_stackable_and_non_empty():
    generator = MultiBlockMaskGenerator(
        grid_size=(14, 14),
        context_scale=(0.5, 0.6),
        target_scale=(0.1, 0.15),
        num_context_masks=1,
        num_target_masks=2,
        min_keep=4,
    )
    context, target = generator(batch_size=3, device=torch.device("cpu"))
    tokens = torch.randn(3, 196, 8)
    assert len(context) == 1 and len(target) == 2
    assert apply_masks(tokens, context).shape[0] == 3
    assert apply_masks(tokens, target).shape[0] == 6
    assert all(mask.size(1) >= 4 for mask in context + target)
