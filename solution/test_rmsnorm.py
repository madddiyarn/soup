"""GPU correctness checks; run python -m pytest solution -q on Kaggle T4."""
import pytest
import torch

from .rmsnorm import rmsnorm, patch_qwen2_rmsnorm


def qwen_reference(x, w, eps):
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(variance + eps)
    return w * normed.to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7, 64, 256])
def test_forward_backward(dtype, rows):
    torch.manual_seed(123)
    x_data = torch.randn((rows, 1536), device="cuda", dtype=dtype) * 0.6
    w = (torch.ones(1536, device="cuda", dtype=torch.float32)
         + 0.01 * torch.randn(1536, device="cuda"))
    w.requires_grad_(False)
    x_ref = x_data.detach().clone().requires_grad_(True)
    x_ours = x_data.detach().clone().requires_grad_(True)
    dy = torch.randn((rows, 1536), device="cuda", dtype=torch.float32)

    y_ref = qwen_reference(x_ref, w, 1e-6)
    y_ours = rmsnorm(x_ours, w, 1e-6)
    assert y_ref.dtype == y_ours.dtype
    torch.testing.assert_close(y_ours, y_ref, atol=4e-3, rtol=4e-3)

    y_ref.backward(dy)
    y_ours.backward(dy)
    torch.testing.assert_close(
        x_ours.grad, x_ref.grad,
        atol=2e-2 if dtype == torch.float16 else 4e-4,
        rtol=2e-2 if dtype == torch.float16 else 4e-4,
    )
    assert w.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen_module_patch():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    patch_qwen2_rmsnorm()
    module = Qwen2RMSNorm(1536).to(device="cuda", dtype=torch.float32)
    module.weight.requires_grad_(False)
    x = torch.randn(3, 9, 1536, device="cuda", dtype=torch.float16,
                    requires_grad=True)
    got = module(x)
    expected = qwen_reference(x, module.weight, module.variance_epsilon)
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got, expected, atol=4e-3, rtol=4e-3)
