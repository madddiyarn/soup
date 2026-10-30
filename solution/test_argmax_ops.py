"""Unit tests for Triton vocabulary argmax; GPU checks require CUDA."""

import pytest
import torch
from .argmax_ops import patch_qwen_token_accuracy_argmax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vocab_argmax_matches_pytorch(dtype):
    # Exact official vocabulary size; sequence length intentionally small.
    x = torch.randn((2, 3, 151936), device="cuda", dtype=dtype)
    x[0, 0, 27] = 250
    x[0, 0, 200] = 250  # Tie -> index 27 must win.
    x[1, 2, 151935] = 250
    expected = torch.argmax(x, dim=-1)
    patch_qwen_token_accuracy_argmax()
    with torch.no_grad():
        actual = x.argmax(dim=-1)
        keepdim = x.argmax(dim=-1, keepdim=True)
    assert torch.equal(actual, expected)
    assert torch.equal(keepdim, expected.unsqueeze(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_argmax_fallback_preserves_torch():
    patch_qwen_token_accuracy_argmax()
    x = torch.tensor([1.0, 9.0, 5.0], device="cuda")
    assert int(x.argmax().item()) == 1
