# Copyright 2026 solo-learn development team.
#
# I-JEPA algorithm reference: https://github.com/facebookresearch/ijepa
# This is an independent solo-learn integration built from the published algorithm.

from functools import partial
from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.ijepa import ijepa_loss_func
from solo.methods.base import BaseMethod, BaseMomentumMethod
from solo.utils.jepa import MultiBlockMaskGenerator, apply_masks
from solo.utils.misc import generate_2d_sincos_pos_embed, omegaconf_select, trunc_normal_
from timm.models.vision_transformer import Block


class IJEPAPredictor(nn.Module):
    """Transformer predictor mapping visible context tokens to target-region embeddings."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        pos_embed = generate_2d_sincos_pos_embed(
            predictor_embed_dim, int(num_patches**0.5), cls_token=False
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.Sequential(
            *[
                Block(
                    predictor_embed_dim,
                    num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(predictor_embed_dim)
        self.proj = nn.Linear(predictor_embed_dim, embed_dim)
        trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_masks: Sequence[torch.Tensor],
        target_masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = context_masks[0].size(0)
        context_tokens = self.predictor_embed(context_tokens)
        positional = self.predictor_pos_embed.expand(batch_size, -1, -1)
        context_positions = apply_masks(positional, context_masks)
        context_tokens = context_tokens + context_positions

        predictions = []
        for context_idx in range(len(context_masks)):
            start = context_idx * batch_size
            context = context_tokens[start : start + batch_size]
            for target_mask in target_masks:
                target_positions = apply_masks(positional, [target_mask])
                target = self.mask_token.expand(
                    batch_size, target_positions.size(1), -1
                ) + target_positions
                x = torch.cat([context, target], dim=1)
                x = self.blocks(x)
                x = self.norm(x[:, context.size(1) :])
                predictions.append(self.proj(x))
        return torch.cat(predictions, dim=0)


class IJEPA(BaseMomentumMethod):
    """Image-based Joint-Embedding Predictive Architecture."""

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        assert "vit" in self.backbone_name, "I-JEPA only supports ViT backbones."
        assert hasattr(self.backbone, "forward_tokens"), "I-JEPA requires the JEPA ViT backbone."

        num_patches = self.backbone.patch_embed.num_patches
        grid_size = self.backbone.patch_embed.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        predictor_num_heads = cfg.method_kwargs.predictor_num_heads
        if predictor_num_heads is None:
            predictor_num_heads = max(1, cfg.method_kwargs.predictor_embed_dim // 64)
        assert cfg.method_kwargs.predictor_embed_dim % predictor_num_heads == 0

        self.predictor = IJEPAPredictor(
            num_patches=num_patches,
            embed_dim=self.features_dim,
            predictor_embed_dim=cfg.method_kwargs.predictor_embed_dim,
            depth=cfg.method_kwargs.predictor_depth,
            num_heads=predictor_num_heads,
        )
        self.mask_generator = MultiBlockMaskGenerator(
            grid_size=tuple(grid_size),
            context_scale=tuple(cfg.method_kwargs.context_mask_scale),
            target_scale=tuple(cfg.method_kwargs.target_mask_scale),
            target_aspect_ratio=tuple(cfg.method_kwargs.target_aspect_ratio),
            num_context_masks=cfg.method_kwargs.num_context_masks,
            num_target_masks=cfg.method_kwargs.num_target_masks,
            min_keep=cfg.method_kwargs.min_keep,
            allow_overlap=cfg.method_kwargs.allow_overlap,
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(IJEPA, IJEPA).add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.predictor_embed_dim = omegaconf_select(
            cfg, "method_kwargs.predictor_embed_dim", 384
        )
        cfg.method_kwargs.predictor_depth = omegaconf_select(
            cfg, "method_kwargs.predictor_depth", 6
        )
        cfg.method_kwargs.predictor_num_heads = omegaconf_select(
            cfg, "method_kwargs.predictor_num_heads", None
        )
        cfg.method_kwargs.context_mask_scale = omegaconf_select(
            cfg, "method_kwargs.context_mask_scale", [0.85, 1.0]
        )
        cfg.method_kwargs.target_mask_scale = omegaconf_select(
            cfg, "method_kwargs.target_mask_scale", [0.15, 0.2]
        )
        cfg.method_kwargs.target_aspect_ratio = omegaconf_select(
            cfg, "method_kwargs.target_aspect_ratio", [0.75, 1.5]
        )
        cfg.method_kwargs.num_context_masks = omegaconf_select(
            cfg, "method_kwargs.num_context_masks", 1
        )
        cfg.method_kwargs.num_target_masks = omegaconf_select(
            cfg, "method_kwargs.num_target_masks", 4
        )
        cfg.method_kwargs.min_keep = omegaconf_select(cfg, "method_kwargs.min_keep", 10)
        cfg.method_kwargs.allow_overlap = omegaconf_select(
            cfg, "method_kwargs.allow_overlap", False
        )
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        return super().learnable_params + [
            {"name": "predictor", "params": self.predictor.parameters()}
        ]

    def _targets(
        self,
        target_tokens: torch.Tensor,
        context_masks: Sequence[torch.Tensor],
        target_masks: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))
        targets = []
        for _ in context_masks:
            for target_mask in target_masks:
                targets.append(apply_masks(target_tokens, [target_mask]))
        return torch.cat(targets, dim=0)

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        # Reuse BaseMethod's crop forwarding and online probe, while JEPA handles its target
        # encoder explicitly so context and target branches share exactly the same masks.
        out = BaseMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]
        images = batch[1]
        images = [images] if isinstance(images, torch.Tensor) else images

        prediction_loss = 0
        for image in images[: self.num_large_crops]:
            if not self.no_channel_last:
                image = image.to(memory_format=torch.channels_last)
            context_masks, target_masks = self.mask_generator(image.size(0), image.device)
            context_tokens = self.backbone.forward_tokens(image, context_masks)
            predictions = self.predictor(context_tokens, context_masks, target_masks)
            with torch.no_grad():
                target_tokens = self.momentum_backbone.forward_tokens(image)
                targets = self._targets(target_tokens, context_masks, target_masks)
            prediction_loss += ijepa_loss_func(predictions, targets)
        prediction_loss /= self.num_large_crops

        self.log("train_ijepa_loss", prediction_loss, on_epoch=True, sync_dist=True)
        return prediction_loss + class_loss
