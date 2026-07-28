# Copyright 2026 solo-learn development team.
#
# LeJEPA algorithm reference: https://github.com/galilai-group/lejepa
# This is an independent solo-learn integration built from the published algorithm.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.lejepa import SIGRegLoss, lejepa_invariance_loss
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class LeJEPA(BaseMethod):
    """LeJEPA with multi-view invariance and Sketched Isotropic Gaussian Regularization."""

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        proj_hidden_dim = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim = cfg.method_kwargs.proj_output_dim
        proj_num_layers = cfg.method_kwargs.proj_num_layers

        layers = []
        input_dim = self.features_dim
        for layer_idx in range(proj_num_layers):
            output_dim = proj_output_dim if layer_idx == proj_num_layers - 1 else proj_hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if layer_idx < proj_num_layers - 1:
                layers.extend([nn.BatchNorm1d(output_dim), nn.GELU()])
            input_dim = output_dim
        self.projector = nn.Sequential(*layers)

        self.lamb: float = cfg.method_kwargs.lamb
        self.sigreg = SIGRegLoss(
            num_slices=cfg.method_kwargs.num_slices,
            num_knots=cfg.method_kwargs.num_knots,
            t_max=cfg.method_kwargs.t_max,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(LeJEPA, LeJEPA).add_and_assert_specific_cfg(cfg)
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        cfg.method_kwargs.proj_hidden_dim = omegaconf_select(
            cfg, "method_kwargs.proj_hidden_dim", 2048
        )
        cfg.method_kwargs.proj_num_layers = omegaconf_select(
            cfg, "method_kwargs.proj_num_layers", 3
        )
        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.05)
        cfg.method_kwargs.num_slices = omegaconf_select(
            cfg, "method_kwargs.num_slices", 256
        )
        cfg.method_kwargs.num_knots = omegaconf_select(
            cfg, "method_kwargs.num_knots", 17
        )
        cfg.method_kwargs.t_max = omegaconf_select(cfg, "method_kwargs.t_max", 3.0)
        assert cfg.method_kwargs.proj_num_layers > 0
        assert 0 <= cfg.method_kwargs.lamb <= 1
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        return super().learnable_params + [
            {"name": "projector", "params": self.projector.parameters()}
        ]

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().forward(X)
        out.update({"z": self.projector(out["feats"])})
        return out

    def multicrop_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().multicrop_forward(X)
        out.update({"z": self.projector(out["feats"])})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        projected_views = torch.stack(out["z"])

        invariance_loss = lejepa_invariance_loss(projected_views)
        sigreg_loss = self.sigreg(projected_views)
        lejepa_loss = (1 - self.lamb) * invariance_loss + self.lamb * sigreg_loss

        with torch.no_grad():
            normalized = F.normalize(projected_views, dim=-1)
            feature_std = normalized.std(dim=1).mean()
        self.log_dict(
            {
                "train_lejepa_loss": lejepa_loss,
                "train_invariance_loss": invariance_loss,
                "train_sigreg_loss": sigreg_loss,
                "train_z_std": feature_std,
            },
            on_epoch=True,
            sync_dist=True,
        )
        return lejepa_loss + class_loss
