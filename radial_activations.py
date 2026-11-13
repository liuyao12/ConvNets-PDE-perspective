"""Numerically stable radial activations used by the PDE-inspired ConvNet."""

import math

import torch
from torch import nn


def _scaled_channels(x):
    """Return a max-scaled channel vector, avoiding overflow in ||x|| computation.

    The channel axis is dimension 1, matching NCHW tensors. Accumulating half and
    bfloat16 inputs in float32 also avoids overflow from x.square() in low precision.
    """
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    max_abs = work.abs().amax(dim=1, keepdim=True)
    safe_max = max_abs.clamp_min(torch.finfo(work.dtype).tiny)
    scaled = work / safe_max
    scaled_norm = torch.linalg.vector_norm(scaled, ord=2, dim=1, keepdim=True)
    return work, max_abs, scaled, scaled_norm


class softball(nn.Module):
    """Smooth radial saturation: x / sqrt(1 + ||x||^2 / R^2)."""

    def __init__(self, radius2=None, inplace=True):
        super().__init__()
        if radius2 is not None and radius2 <= 0:
            raise ValueError("radius2 must be positive")
        self.radius2 = float(radius2) if radius2 is not None else None

    def forward(self, x):
        if self.radius2 is None:
            self.radius2 = float(x.size(1))
        radius = math.sqrt(self.radius2)

        work, max_abs, scaled, scaled_norm = _scaled_channels(x)
        # For nonzero x, max(scaled) == 1, hence ||scaled|| >= 1.
        # The clamp also makes the zero-vector path entirely regular.
        scaled_norm = scaled_norm.clamp_min(1.0)

        # When max_abs <= R, compute the norm directly in a bounded range.
        small_max = max_abs.clamp_max(radius)
        small_norm = small_max * scaled_norm
        small_denom = torch.hypot(small_norm, torch.full_like(small_norm, radius))
        small = work * (radius / small_denom)

        # When max_abs > R, divide by max_abs first:
        # x / sqrt(1 + ||x||^2/R^2)
        #   = (x/max_abs) * R / sqrt((R/max_abs)^2 + ||x/max_abs||^2).
        # This avoids squaring large activations.
        large_max = max_abs.clamp_min(radius)
        radius_over_max = radius / large_max
        large_denom = torch.sqrt(radius_over_max.square() + scaled_norm.square())
        large = scaled * (radius / large_denom)

        out = torch.where(max_abs <= radius, small, large)
        return out.to(dtype=x.dtype)


class hardball(nn.Module):
    """Projection onto the channelwise Euclidean ball of radius sqrt(radius2)."""

    def __init__(self, radius2=None):
        super().__init__()
        if radius2 is not None and radius2 <= 0:
            raise ValueError("radius2 must be positive")
        self.radius = math.sqrt(float(radius2)) if radius2 is not None else None

    def forward(self, x):
        if self.radius is None:
            self.radius = math.sqrt(float(x.size(1)))

        work, max_abs, scaled, scaled_norm = _scaled_channels(x)
        # For nonzero x, ||x|| = max_abs * scaled_norm. The comparison below
        # is equivalent to ||x|| > R but never forms that potentially overflowing
        # product. Clamping handles x == 0 without a 0/0 branch.
        scaled_norm = scaled_norm.clamp_min(1.0)
        scaled_radius = self.radius / scaled_norm
        clipped = scaled * scaled_radius
        out = torch.where(max_abs > scaled_radius, clipped, work)
        return out.to(dtype=x.dtype)
