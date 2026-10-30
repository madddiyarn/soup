"""Original two-stage Triton vocabulary argmax for TRL's training token accuracy.

The official challenge uses TRL 0.29.1, whose SFTTrainer.compute_loss calls
`shift_logits.argmax(dim=-1)` for mean_token_accuracy. We replace ONLY that
operation for a contiguous CUDA [B, S, 151936] tensor under torch.no_grad().
The output remains the exact first-index argmax, same dtype and shape.

This is a GPU EXPERIMENT, not verified on Kaggle T4. In particular, the
competition's 5%-kernel-share check must be run; it is NOT guaranteed.
"""

import torch
import triton
import triton.language as tl

_VOCAB = 151936
_TILE = 2048


@triton.jit
def soup_argmax_vocab_parts(
    X, PART_MAX, PART_INDEX,
    VOC: tl.constexpr, PARTS: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    idx = part * TILE + tl.arange(0, TILE)
    valid = idx < VOC
    x = tl.load(X + row * VOC + idx, valid, other=-float("inf")).to(tl.float32)
    vmax = tl.max(x, axis=0)
    # PyTorch argmax returns the FIRST index on ties.
    first = tl.min(tl.where(valid & (x == vmax), idx, 2147483647), axis=0)
    tl.store(PART_MAX + row * PARTS + part, vmax)
    tl.store(PART_INDEX + row * PARTS + part, first)


@triton.jit
def soup_argmax_vocab_finish(
    PART_MAX, PART_INDEX, OUT,
    PARTS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < PARTS
    m = tl.load(PART_MAX + row * PARTS + col, valid, other=-float("inf"))
    index = tl.load(PART_INDEX + row * PARTS + col, valid, other=2147483647)
    best = tl.max(m, axis=0)
    best_index = tl.min(tl.where(valid & (m == best), index, 2147483647), axis=0)
    tl.store(OUT + row, best_index)


def triton_vocab_argmax(x, *, keepdim=False):
    """Exactly preserve torch.Tensor.argmax(dim=-1, keepdim=...)."""
    if x.ndim != 3 or x.shape[-1] != _VOCAB or not x.is_contiguous():
        raise ValueError("expected contiguous [B, S, 151936] logits")
    rows = x.shape[0] * x.shape[1]
    parts = triton.cdiv(_VOCAB, _TILE)
    max_parts = torch.empty((rows, parts), device=x.device, dtype=torch.float32)
    idx_parts = torch.empty((rows, parts), device=x.device, dtype=torch.int32)
    out = torch.empty((rows,), device=x.device, dtype=torch.int64)
    soup_argmax_vocab_parts[(rows, parts)](
        x, max_parts, idx_parts, _VOCAB, parts, _TILE,
        num_warps=4,
    )
    soup_argmax_vocab_finish[(rows,)](
        max_parts, idx_parts, out, parts, triton.next_power_of_2(parts),
        num_warps=4,
    )
    result = out.reshape(x.shape[:-1])
    return result.unsqueeze(-1) if keepdim else result


def patch_qwen_token_accuracy_argmax():
    """Narrow, reversible hook for the exact reduction performed by TRL.

    All other argmax calls retain their original torch behavior. This is not
    a training task alteration: the metric tensor stays bit-for-bit exact.
    """
    original = torch.Tensor.argmax
    if getattr(original, "_soup_vocab_argmax_patch", False):
        return

    def argmax(self, dim=None, keepdim=False):
        use_triton = (
            isinstance(self, torch.Tensor)
            and self.is_cuda
            and self.ndim == 3
            and self.shape[-1] == _VOCAB
            and 0 < self.shape[0] <= 4
            and 0 < self.shape[1] <= 512
            and self.is_contiguous()
            and self.dtype in (torch.float16, torch.float32)
            and dim in (-1, 2)
            and not torch.is_grad_enabled()
        )
        if use_triton:
            return triton_vocab_argmax(self, keepdim=keepdim)
        return original(self, dim=dim, keepdim=keepdim)

    argmax._soup_vocab_argmax_patch = True
    argmax._soup_vocab_argmax_original = original
    torch.Tensor.argmax = argmax
