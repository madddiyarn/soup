"""GPU-only correctness tests for experimental 151936-class kernels."""
import pytest
import torch
import torch.nn.functional as F

from .vocab_ce import cross_entropy_rows, patch_qwen_cross_entropy
from .entropy import entropy_from_qwen_logits, patch_trl_entropy


def _ref_entropy(logits):
    l = torch.log_softmax(logits.float(), dim=-1)
    return -(l.exp() * l).sum(dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("vocab", [32, 2049, 151936])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_tiled_ce_forward_backward(vocab, reduction):
    torch.manual_seed(42)
    n = 3
    x = torch.randn((n, vocab), dtype=torch.float32, device='cuda') * 2
    labels = torch.tensor([1, -100, vocab-1], device='cuda', dtype=torch.int64)

    a = x.detach().clone().requires_grad_(True)
    b = x.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(a, labels, reduction=reduction)
    got = cross_entropy_rows(b, labels, reduction=reduction)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    ref.backward()
    got.backward()
    torch.testing.assert_close(b.grad, a.grad, rtol=3e-4, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_patched_qwen_ce_is_used():
    patch_qwen_cross_entropy()
    x = torch.randn((2, 151936), device='cuda', requires_grad=True)
    labels = torch.tensor([7, -100], device='cuda')
    got = F.cross_entropy(x, labels)
    expected = torch.logsumexp(x[0], dim=0) - x[0, 7]
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
    got.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_entropy(dtype):
    torch.manual_seed(12)
    logits = torch.randn((2, 3, 151936), dtype=dtype, device='cuda') * 4
    expected = _ref_entropy(logits).to(dtype)
    got = entropy_from_qwen_logits(logits)
    torch.testing.assert_close(got.float(), expected.float(), rtol=3e-3, atol=3e-3)
    assert got.shape == logits.shape[:-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_patch_entropy_call():
    from trl.trainer import sft_trainer
    patch_trl_entropy()
    x = torch.randn((1, 2, 151936), device='cuda')
    with torch.no_grad():
        y = sft_trainer.entropy_from_logits(x)
    assert y.shape == (1, 2)
    assert torch.isfinite(y).all()
