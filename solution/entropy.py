"""Original two-stage Triton Shannon entropy across a large vocabulary.

Preserves a per-token entropy result required by TRL/Soup logging.
No-grad metric operation, not a training-loss replacement.
UNVALIDATED on Kaggle T4; test the official rewind/entropy tolerances.
"""

import torch
import triton
import triton.language as tl

_TILE = 2048


@triton.jit
def soup_entropy_partial(X, M, S, SX,
                         VOC: tl.constexpr, PARTS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    idx = part * TILE + tl.arange(0, TILE)
    values = tl.load(X + row * VOC + idx, idx < VOC, other=-float('inf')).to(tl.float32)
    vmax = tl.max(values, 0)
    vexp = tl.exp(values - vmax)
    s = tl.sum(vexp, 0)
    # Avoid NaN on masked elements: -inf * 0
    sx = tl.sum(vexp * tl.where(idx < VOC, values, 0.0), 0)
    tl.store(M + row * PARTS + part, vmax)
    tl.store(S + row * PARTS + part, s)
    tl.store(SX + row * PARTS + part, sx)


@triton.jit
def soup_entropy_finish(M, S, SX, OUT,
                           PARTS: tl.constexpr, PB: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, PB)
    m = tl.load(M + row * PARTS + idx, idx < PARTS, other=-float('inf'))
    s = tl.load(S + row * PARTS + idx, idx < PARTS, other=0.0)
    sx = tl.load(SX + row * PARTS + idx, idx < PARTS, other=0.0)
    vmax = tl.max(m, 0)
    a = tl.exp(m - vmax)
    total_s = tl.sum(a * s, 0)
    total_sx = tl.sum(a * sx, 0)
    entropy = vmax + tl.log(total_s) - total_sx / total_s
    tl.store(OUT + row, entropy)


def entropy_from_qwen_logits(logits):
    shape = logits.shape
    n = logits.numel() // shape[-1]
    voc = shape[-1]
    parts = triton.cdiv(voc, _TILE)
    x = logits.reshape(n, voc)
    m = torch.empty((n, parts), dtype=torch.float32, device=x.device)
    s = torch.empty_like(m)
    sx = torch.empty_like(m)
    out = torch.empty((n,), dtype=logits.dtype, device=x.device)
    soup_entropy_partial[(n, parts)](
        x, m, s, sx, voc, parts, _TILE, num_warps=4,
        enable_fp_fusion=False,
    )
    soup_entropy_finish[(n,)](
        m, s, sx, out, parts, triton.next_power_of_2(parts),
        num_warps=4, enable_fp_fusion=False,
    )
    return out.reshape(shape[:-1])


def patch_trl_entropy():
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    if getattr(original, '_soup_entropy_patch', False):
        return

    def replacement(logits, chunk_size=128):
        eligible = (
            isinstance(logits, torch.Tensor)
            and logits.is_cuda and logits.ndim == 3
            and logits.shape[-1] == 151936
            and logits.numel() > 0
            and logits.is_contiguous()
            and logits.dtype in (torch.float16, torch.float32)
            and not torch.is_grad_enabled()
        )
        if eligible:
            return entropy_from_qwen_logits(logits)
        return original(logits, chunk_size=chunk_size)

    replacement._soup_entropy_patch = True
    sft_trainer.entropy_from_logits = replacement
