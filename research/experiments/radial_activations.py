"""Numerically stable radial activations used by the PDE-inspired ConvNet."""
import math
import torch
from torch import nn

def _scaled_channels(x):
    """Compute per-pixel channel scale and normalized vector in safe precision."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    max_abs = work.abs().amax(dim=1, keepdim=True)
    safe_max = max_abs.clamp_min(torch.finfo(work.dtype).tiny)
    scaled = work / safe_max
    scaled_norm = torch.linalg.vector_norm(scaled, ord=2, dim=1, keepdim=True)
    return work, max_abs, scaled, scaled_norm

class softball(nn.Module):
    def __init__(self, radius2=None, inplace=True):
        super().__init__()
        if radius2 is not None and radius2 <= 0:
            raise ValueError('radius2 must be positive')
        self.radius2 = float(radius2) if radius2 is not None else None

    def forward(self, x):
        if self.radius2 is None:
            self.radius2 = float(x.size(1))
        radius = math.sqrt(self.radius2)
        work, max_abs, scaled, scaled_norm = _scaled_channels(x)
        scaled_norm = scaled_norm.clamp_min(1.0)
        small_max = max_abs.clamp_max(radius)
        small_norm = small_max * scaled_norm
        small_denom = torch.hypot(small_norm, torch.full_like(small_norm, radius))
        small = work * (radius / small_denom)
        large_max = max_abs.clamp_min(radius)
        radius_over_max = radius / large_max
        large_denom = torch.sqrt(radius_over_max.square() + scaled_norm.square())
        large = scaled * (radius / large_denom)
        out = torch.where(max_abs <= radius, small, large)
        return out.to(dtype=x.dtype)

class hardball(nn.Module):
    def __init__(self, radius2=None):
        super().__init__()
        if radius2 is not None and radius2 <= 0:
            raise ValueError('radius2 must be positive')
        self.radius = math.sqrt(float(radius2)) if radius2 is not None else None
    def forward(self, x):
        if self.radius is None:
            self.radius = math.sqrt(float(x.size(1)))
        work, max_abs, scaled, scaled_norm = _scaled_channels(x)
        scaled_norm = scaled_norm.clamp_min(1.0)
        scaled_radius = self.radius / scaled_norm
        clipped = scaled * scaled_radius
        out = torch.where(max_abs > scaled_radius, clipped, work)
        return out.to(dtype=x.dtype)
