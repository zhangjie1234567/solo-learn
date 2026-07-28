# Copyright 2026 solo-learn development team.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
from solo.utils.jepa import apply_masks
from solo.utils.misc import generate_2d_sincos_pos_embed
from timm.models.vision_transformer import VisionTransformer


class JEPAVisionTransformer(VisionTransformer):
    """ViT exposing patch-token masking while retaining a standard pooled forward.

    The regular ``forward`` returns a single feature vector and therefore remains compatible
    with solo-learn's linear, k-NN and UMAP evaluation entry points. I-JEPA training uses
    ``forward_tokens`` to mask patch tokens before the transformer blocks.
    """

    def __init__(self, **kwargs):
        super().__init__(num_classes=0, **kwargs)
        num_patches = self.patch_embed.num_patches
        pos_embed = generate_2d_sincos_pos_embed(
            self.num_features, int(num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.requires_grad = False

    def forward_tokens(
        self, x: torch.Tensor, masks: Optional[Sequence[torch.Tensor]] = None
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if masks is not None:
            x = apply_masks(x, masks)

        if hasattr(self, "pos_drop"):
            x = self.pos_drop(x)
        if hasattr(self, "patch_drop"):
            x = self.patch_drop(x)
        if hasattr(self, "norm_pre"):
            x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x).mean(dim=1)


def vit_tiny(**kwargs):
    return JEPAVisionTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_small(**kwargs):
    return JEPAVisionTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(**kwargs):
    return JEPAVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(**kwargs):
    return JEPAVisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
