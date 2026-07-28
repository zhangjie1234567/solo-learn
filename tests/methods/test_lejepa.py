import torch

from solo.methods import LeJEPA

from .utils import gen_base_cfg, gen_trainer, prepare_dummy_dataloaders


def test_lejepa_fast_dev_run():
    cfg = gen_base_cfg("lejepa", batch_size=2, num_classes=10)
    cfg.data.dataset = "cifar10"
    cfg.backbone = {"name": "resnet18"}
    cfg.method_kwargs = {
        "proj_hidden_dim": 32,
        "proj_output_dim": 16,
        "proj_num_layers": 2,
        "lamb": 0.05,
        "num_slices": 8,
        "num_knots": 9,
        "t_max": 3.0,
    }
    model = LeJEPA(cfg)
    output = model(torch.randn(2, 3, 32, 32))
    assert output["z"].shape == (2, 16)

    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10", num_large_crops=2, num_classes=10, batch_size=2
    )
    gen_trainer(cfg).fit(model, train_dl, val_dl)
