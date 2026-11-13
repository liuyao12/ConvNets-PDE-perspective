import math
import unittest

import torch

from radial_activations import hardball, softball


class RadialActivationTests(unittest.TestCase):
    def test_zero_and_tiny_vectors_have_finite_gradients(self):
        for activation in (hardball(), softball()):
            x = torch.zeros(2, 8, 3, 3, requires_grad=True)
            y = activation(x)
            self.assertTrue(torch.isfinite(y).all())
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

            x = (torch.randn(2, 8, 3, 3) * 1e-20).requires_grad_()
            y = activation(x)
            y.square().sum().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_large_and_low_precision_inputs(self):
        cases = (
            (torch.float32, 1e30),
            (torch.float16, 1e4),
            (torch.bfloat16, 1e4),
        )
        for dtype, magnitude in cases:
            for activation in (hardball(), softball()):
                x = torch.full((2, 8, 2, 2), magnitude, dtype=dtype, requires_grad=True)
                y = activation(x)
                y.float().square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_radial_bounds(self):
        channels = 16
        radius = math.sqrt(channels)
        x = torch.randn(3, channels, 5, 5) * 100

        hard = hardball(radius2=channels)(x)
        soft = softball(radius2=channels)(x)
        hard_norm = torch.linalg.vector_norm(hard, dim=1)
        soft_norm = torch.linalg.vector_norm(soft, dim=1)

        self.assertLessEqual(float(hard_norm.max()), radius + 1e-5)
        self.assertTrue(torch.all(soft_norm < radius + 1e-5))

    def test_orthogonal_equivariance(self):
        torch.manual_seed(7)
        channels = 8
        q, _ = torch.linalg.qr(torch.randn(channels, channels))
        x = torch.randn(2, channels, 4, 4) * 5

        for factory in (hardball, softball):
            activation = factory(radius2=channels)
            lhs = activation(torch.einsum("ij,bjhw->bihw", q, x))
            rhs = torch.einsum("ij,bjhw->bihw", q, activation(x))
            self.assertTrue(torch.allclose(lhs, rhs, atol=2e-5, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
