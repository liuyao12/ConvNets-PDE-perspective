"""Self-contained, fast QLNet NaN mechanism probes; no dataset or checkpoint required."""
import math
import torch

def legacy_hardball(x):
    radius = math.sqrt(x.size(1))
    norm = torch.sqrt((x * x).sum(1, keepdim=True))
    return torch.where(norm > radius, radius * x / norm, x)

def stable_hardball(x):
    """Max-scaled radial projection; zero-safe backward; float32 half-norms."""
    radius = math.sqrt(x.size(1))
    a = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    max_abs = a.abs().amax(1, keepdim=True)
    scaled = a / max_abs.clamp_min(torch.finfo(a.dtype).tiny)
    unit_norm = torch.linalg.vector_norm(scaled, dim=1, keepdim=True).clamp_min(1.0)
    cap = radius / unit_norm
    clipped = scaled * cap
    return torch.where(max_abs > cap, clipped, a).to(x.dtype)

def check_zero_gradient():
    a = torch.zeros(1, 8, 2, 2, requires_grad=True)
    y = legacy_hardball(a)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert (~torch.isfinite(a.grad)).any(), 'Expected original hardball zero-norm bug'
    b = torch.zeros_like(a, requires_grad=True)
    stable_hardball(b).sum().backward()
    assert torch.isfinite(b.grad).all(), 'Stable radial derivative should be finite'
    print('PASS: original finite-forward/NaN-backward reproduced and fixed')

def check_fp16_norm():
    a = torch.full((1, 512, 1, 1), 12, dtype=torch.float16)
    old, new = legacy_hardball(a), stable_hardball(a)
    assert torch.all(old == 0)
    assert torch.allclose(new, torch.ones_like(new), atol=1e-3)
    print('PASS: FP16 norm overflow silently erases a moderate activation')

def check_bilinear_overflow():
    assert torch.isinf(torch.tensor(300, dtype=torch.float16) ** 2)
    assert torch.isfinite(torch.tensor(300, dtype=torch.bfloat16) ** 2)
    print('PASS: FP16 paired product overflows before post-block clipping')

def check_preproduct_bound():
    torch.manual_seed(19)
    w = 16
    z = 1000 * torch.randn(2, 2*w, 4, 4, dtype=torch.float16)
    p = stable_hardball(z)
    product = p[:, :w] * p[:, w:]
    assert torch.isfinite(product).all()
    assert float(product.abs().max()) <= w+0.1
    # Mathematical guarantee: |p_i*p_j| <= ||p||_2^2/2 <= w.
    print('PASS: clipping 2w inputs to radius sqrt(2w) bounds each product by w')

if __name__ == '__main__':
    check_zero_gradient()
    check_fp16_norm()
    check_bilinear_overflow()
    check_preproduct_bound()
