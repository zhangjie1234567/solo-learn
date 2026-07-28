import torch

from solo.methods import IJEPA

from .utils import gen_base_cfg, gen_trainer, prepare_dummy_dataloaders


def test_ijepa_fast_dev_run():
    cfg = gen_base_cfg("ijepa", batch_size=2, num_classes=10, momentum=True)
    cfg.data.dataset = "cifar10"
    cfg.backbone = {"name": "vit_tiny", "kwargs": {"img_size": 32, "patch_size": 8}}
    cfg.method_kwargs = {
        "predictor_embed_dim": 96,
        "predictor_depth": 1,
        "predictor_num_heads": 3,
        "context_mask_scale": [0.5, 0.6],
        "target_mask_scale": [0.1, 0.15],
        "target_aspect_ratio": [0.75, 1.5],
        "num_context_masks": 1,
        "num_target_masks": 2,
        "min_keep": 2,
        "allow_overlap": False,
    }
    model = IJEPA(cfg)
    batch = torch.randn(2, 3, 32, 32)
    output = model(batch)
    assert output["feats"].shape == (2, model.features_dim)

    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10", num_large_crops=2, num_classes=10, batch_size=2
    )
    gen_trainer(cfg).fit(model, train_dl, val_dl)
