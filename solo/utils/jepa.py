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

import math
from typing import List, Optional, Sequence, Tuple

import torch


def apply_masks(x: torch.Tensor, masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Selects token indices from ``x`` and concatenates mask outputs on the batch axis."""

    outputs = []
    for mask in masks:
        index = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        outputs.append(torch.gather(x, dim=1, index=index))
    return torch.cat(outputs, dim=0)


class MultiBlockMaskGenerator:
    """Generates rectangular context and target masks for I-JEPA.

    Mask sizes are sampled once per batch so tensors remain stackable, while locations are
    sampled independently for every image. Context masks can be constrained not to overlap
    target blocks, matching the multi-block masking protocol used by I-JEPA.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        context_scale: Tuple[float, float] = (0.85, 1.0),
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        num_context_masks: int = 1,
        num_target_masks: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ):
        self.height, self.width = grid_size
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.num_context_masks = num_context_masks
        self.num_target_masks = num_target_masks
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

        assert self.height > 1 and self.width > 1
        assert 0 < context_scale[0] <= context_scale[1] <= 1
        assert 0 < target_scale[0] <= target_scale[1] <= 1
        assert 0 < target_aspect_ratio[0] <= target_aspect_ratio[1]
        assert num_context_masks > 0 and num_target_masks > 0

    def _sample_block_size(
        self,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
        device: torch.device,
    ) -> Tuple[int, int]:
        random_value = torch.rand((), device=device).item()
        block_scale = scale[0] + random_value * (scale[1] - scale[0])
        block_aspect = aspect_ratio[0] + random_value * (
            aspect_ratio[1] - aspect_ratio[0]
        )
        max_keep = max(1, int(self.height * self.width * block_scale))
        block_h = int(round(math.sqrt(max_keep * block_aspect)))
        block_w = int(round(math.sqrt(max_keep / block_aspect)))
        block_h = min(max(block_h, 1), self.height - 1)
        block_w = min(max(block_w, 1), self.width - 1)
        return block_h, block_w

    def _sample_block(
        self,
        block_size: Tuple[int, int],
        batch_size: int,
        device: torch.device,
        acceptable_regions: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        block_h, block_w = block_size
        masks, complements = [], []

        for batch_idx in range(batch_size):
            best_mask = None
            best_complement = None
            for attempt in range(80):
                top = torch.randint(0, self.height - block_h + 1, (), device=device).item()
                left = torch.randint(0, self.width - block_w + 1, (), device=device).item()
                block = torch.zeros((self.height, self.width), dtype=torch.bool, device=device)
                block[top : top + block_h, left : left + block_w] = True
                complement = ~block

                if acceptable_regions is not None:
                    # Gradually relax overlap constraints if an unusually dense target layout
                    # makes a valid context block impossible.
                    regions = acceptable_regions[batch_idx]
                    keep_regions = max(len(regions) - attempt // 20, 0)
                    for region in regions[:keep_regions]:
                        block &= region

                indices = torch.nonzero(block.flatten(), as_tuple=False).flatten()
                if best_mask is None or indices.numel() > best_mask.numel():
                    best_mask = indices
                    best_complement = complement
                if indices.numel() >= self.min_keep:
                    break

            masks.append(best_mask)
            complements.append(best_complement)

        min_tokens = min(mask.numel() for mask in masks)
        if min_tokens == 0:
            raise RuntimeError("Unable to sample a non-empty JEPA block mask.")
        return torch.stack([mask[:min_tokens] for mask in masks]), complements

    @torch.no_grad()
    def __call__(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        target_size = self._sample_block_size(
            self.target_scale, self.target_aspect_ratio, device
        )
        context_size = self._sample_block_size(self.context_scale, (1.0, 1.0), device)

        target_masks = []
        target_complements: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        for _ in range(self.num_target_masks):
            mask, complements = self._sample_block(target_size, batch_size, device)
            target_masks.append(mask)
            for batch_idx, complement in enumerate(complements):
                target_complements[batch_idx].append(complement)

        acceptable_regions = None if self.allow_overlap else target_complements
        context_masks = []
        for _ in range(self.num_context_masks):
            mask, _ = self._sample_block(
                context_size,
                batch_size,
                device,
                acceptable_regions=acceptable_regions,
            )
            context_masks.append(mask)

        return context_masks, target_masks
