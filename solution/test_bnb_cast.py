"""GPU tests of v4 cast and limited bitsandbytes integration."""
import pytest
import torch

from .bnb_cast import qlora_cast, patch_bnb_linear4bit_casts


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA GPU')
@pytest.mark.parametrize('shape', [(4, 1536), (4, 8, 1536), (4, 8, 8960)])
@pytest.mark.parametrize('src,dst', [(torch.float32, torch.float16),
                                     (torch.float16, torch.float32)])
def test_bnb_cast_matches_torch_forward_and_backward(shape, src, dst):
    torch.manual_seed(2026)
    original = torch.randn(shape, device='cuda', dtype=src) * 0.1
    x = original.detach().clone().requires_grad_(True)
    y = original.detach().clone().requires_grad_(True)
    ref = x.to(dst)
    got = qlora_cast(y, dst)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    got.backward(grad)
    torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA GPU')
@pytest.mark.parametrize("in_features", [1536, 8960])
def test_linear4bit_patch_preserves_computation(in_features):
    import bitsandbytes as bnb

    torch.manual_seed(2026)
    module = bnb.nn.Linear4bit(
        in_features, 256, bias=False, compute_dtype=torch.float16,
        compress_statistics=True, quant_type='nf4',
    ).cuda().train()
    x_data = torch.randn((4, 8, in_features), device='cuda', dtype=torch.float32) * 0.05
    x_ref = x_data.detach().clone().requires_grad_(True)
    x_new = x_data.detach().clone().requires_grad_(True)
    ref = module(x_ref)
    ref.sum().backward()
    patch_bnb_linear4bit_casts()
    got = module(x_new)
    got.sum().backward()
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=1e-3, rtol=1e-3)
