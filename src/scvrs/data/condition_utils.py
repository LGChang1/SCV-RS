from __future__ import annotations

import math
import random

import torch


GEOMETRY_FEATURE_NAMES = (
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "center_x",
    "center_y",
    "width",
    "height",
    "area",
    "log_area",
    "aspect_ratio",
    "area_rank",
    "x_rank",
    "y_rank",
    "area_to_mean_ratio",
    "center_x_minus_mean",
    "center_y_minus_mean",
)


def sample_candidate_supervision(
    labels,
    ious,
    weights,
    *,
    sample_seed: int,
    hard_negatives_per_positive: int = 6,
    random_negatives_per_positive: int = 2,
):
    """Mask excess negatives without changing the candidate mask pool.

    Positives are always supervised, ambiguous candidates (label -100) remain
    ignored, and only a deterministic hard/random subset of true negatives is
    supervised. Returned tensors have exactly the same length as the inputs.
    """
    labels = torch.as_tensor(labels, dtype=torch.long).clone()
    ious = torch.as_tensor(ious, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32).clone()
    if labels.ndim != 1 or ious.ndim != 1 or weights.ndim != 1:
        raise ValueError("candidate labels, ious and weights must be 1-D")
    if not (labels.numel() == ious.numel() == weights.numel()):
        raise ValueError(
            "candidate labels, ious and weights must have equal lengths"
        )
    if hard_negatives_per_positive < 0 or random_negatives_per_positive < 0:
        raise ValueError("negative sampling budgets must be non-negative")
    if not torch.isfinite(ious).all() or not torch.isfinite(weights).all():
        raise ValueError("candidate ious/weights contain NaN or Inf")

    positive_indices = torch.where(labels == 1)[0].tolist()
    negative_indices = torch.where(labels == 0)[0].tolist()
    # Train data is expected to contain a positive (GT fallback guarantees
    # this). Keep a defensive single-unit budget so diagnostics remain useful
    # if a malformed sample reaches this function.
    positive_count = max(len(positive_indices), 1)
    hard_budget = hard_negatives_per_positive * positive_count
    random_budget = random_negatives_per_positive * positive_count

    hard_indices = sorted(
        negative_indices,
        key=lambda index: (-float(ious[index]), index),
    )[:hard_budget]
    hard_set = set(hard_indices)
    random_pool = [
        index for index in negative_indices if index not in hard_set
    ]
    rng = random.Random(int(sample_seed))
    rng.shuffle(random_pool)
    random_indices = random_pool[:random_budget]
    selected_negatives = hard_set.union(random_indices)

    for index in negative_indices:
        if index not in selected_negatives:
            labels[index] = -100
            weights[index] = 0.0

    return labels, weights, {
        "positive_count": len(positive_indices),
        "negative_count": len(negative_indices),
        "hard_negative_count": len(hard_indices),
        "random_negative_count": len(random_indices),
        "supervised_negative_count": len(selected_negatives),
        "ignore_count": int((labels == -100).sum().item()),
    }


def _normalized_rank(values: torch.Tensor) -> torch.Tensor:
    count = values.numel()
    if count <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        count, device=values.device, dtype=values.dtype
    )
    return ranks / float(count - 1)


def compute_mask_geometry(mask_pool: torch.Tensor) -> torch.Tensor:
    """Return stable normalized geometry features for an N×H×W mask pool."""
    if mask_pool.ndim == 2:
        mask_pool = mask_pool.unsqueeze(0)
    if mask_pool.ndim != 3:
        raise ValueError(f"Expected N×H×W masks, got {tuple(mask_pool.shape)}")
    count, height, width = mask_pool.shape
    if count == 0:
        return torch.empty(
            (0, len(GEOMETRY_FEATURE_NAMES)),
            dtype=torch.float32,
            device=mask_pool.device,
        )
    masks = mask_pool > 0
    dtype = torch.float32
    device = masks.device
    x_axis = torch.arange(width, device=device).view(1, 1, width)
    y_axis = torch.arange(height, device=device).view(1, height, 1)
    any_x = masks.any(dim=1)
    any_y = masks.any(dim=2)
    valid = masks.flatten(1).any(dim=1)

    x_min = torch.where(
        valid,
        torch.where(any_x, x_axis[:, 0], width).amin(dim=1),
        torch.zeros(count, device=device, dtype=torch.long),
    ).to(dtype)
    x_max = torch.where(
        valid,
        torch.where(any_x, x_axis[:, 0], -1).amax(dim=1),
        torch.zeros(count, device=device, dtype=torch.long),
    ).to(dtype)
    y_min = torch.where(
        valid,
        torch.where(any_y, y_axis[0, :, 0], height).amin(dim=1),
        torch.zeros(count, device=device, dtype=torch.long),
    ).to(dtype)
    y_max = torch.where(
        valid,
        torch.where(any_y, y_axis[0, :, 0], -1).amax(dim=1),
        torch.zeros(count, device=device, dtype=torch.long),
    ).to(dtype)

    box_width = torch.where(valid, x_max - x_min + 1, 0.0)
    box_height = torch.where(valid, y_max - y_min + 1, 0.0)
    center_x = torch.where(valid, (x_min + x_max + 1) / 2, 0.0)
    center_y = torch.where(valid, (y_min + y_max + 1) / 2, 0.0)
    area_pixels = masks.flatten(1).sum(dim=1).to(dtype)
    image_area = float(max(height * width, 1))

    x_min = x_min / max(float(width), 1.0)
    x_max = (x_max + valid.to(dtype)) / max(float(width), 1.0)
    y_min = y_min / max(float(height), 1.0)
    y_max = (y_max + valid.to(dtype)) / max(float(height), 1.0)
    center_x = center_x / max(float(width), 1.0)
    center_y = center_y / max(float(height), 1.0)
    box_width = box_width / max(float(width), 1.0)
    box_height = box_height / max(float(height), 1.0)
    area = area_pixels / image_area
    log_area = torch.log(area.clamp_min(1.0 / image_area))
    aspect = box_width / box_height.clamp_min(1.0 / max(float(height), 1.0))
    aspect = torch.where(valid, aspect, torch.zeros_like(aspect))

    area_rank = _normalized_rank(area)
    x_rank = _normalized_rank(center_x)
    y_rank = _normalized_rank(center_y)
    mean_area = area.mean().clamp_min(1.0 / image_area)
    area_ratio = area / mean_area
    center_x_delta = center_x - center_x.mean()
    center_y_delta = center_y - center_y.mean()

    features = torch.stack(
        (
            x_min,
            y_min,
            x_max,
            y_max,
            center_x,
            center_y,
            box_width,
            box_height,
            area,
            log_area,
            aspect,
            area_rank,
            x_rank,
            y_rank,
            area_ratio,
            center_x_delta,
            center_y_delta,
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite mask geometry generated")
    return features


def pairwise_geometry(
    subject: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Compute relation geometry from two feature matrices with shape N×G."""
    if subject.shape != reference.shape or subject.ndim != 2:
        raise ValueError(
            f"Expected equal N×G tensors, got {subject.shape}, {reference.shape}"
        )
    cx, cy, area = 4, 5, 8
    dx = subject[:, cx] - reference[:, cx]
    dy = subject[:, cy] - reference[:, cy]
    # Tiny soft-reference areas can have finite forward values but an
    # unstable 1/x derivative through log.  Clamp operands before the ratio
    # and bound the semantic range used by the relation verifier.
    log_ratio = torch.log(
        subject[:, area].float().clamp_min(1e-5)
        / reference[:, area].float().clamp_min(1e-5)
    ).clamp(min=-10.0, max=10.0)
    return torch.stack(
        (dx, dy, dx.abs(), dy.abs(), torch.hypot(dx, dy), log_ratio),
        dim=1,
    )
