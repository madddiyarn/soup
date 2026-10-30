"""Original (not derived from starter SwiGLU) tiled-vocabulary cross entropy.

Scopes itself to the official Qwen vocabulary and training-only FP32 2D logits.
Does NOT change token order, loss reduction, settings, or logged logits.

UNVALIDATED on Kaggle T4: run correctness and the complete official checks.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_TILE = 2048


@triton.jit
def soup_ce_split_stats(X, M, S, VOC: tl.constexpr, SPLITS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    idx = part * TILE + tl.arange(0, TILE)
    values = tl.load(X + row * VOC + idx, idx < VOC, other=-float("inf")).to(tl.float32)
    vmax = tl.max(values, 0)
    exps = tl.exp(values - vmax)
    esum = tl.sum(exps, 0)
    tl.store(M + row * SPLITS + part, vmax)
    tl.store(S + row * SPLITS + part, esum)


@triton.jit
def soup_ce_row_loss(
    X, LABEL, M, S, LSE, LOSS,
    VOC: tl.constexpr, SPLITS: tl.constexpr, PART_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, PART_BLOCK)
    m = tl.load(M + row * SPLITS + col, col < SPLITS, other=-float("inf"))
    s = tl.load(S + row * SPLITS + col, col < SPLITS, other=0.0)
    max_m = tl.max(m, 0)
    sum_exp = tl.sum(s * tl.exp(m - max_m), 0)
    lse = max_m + tl.log(sum_exp)
    label = tl.load(LABEL + row)
    valid = (label >= 0) & (label < VOC)
    chosen = tl.load(X + row * VOC + tl.maximum(label, 0), mask=valid, other=0.0)
    loss = tl.where(valid, lse - chosen, 0.0)
    tl.store(LSE + row, lse)
    tl.store(LOSS + row, loss)


@triton.jit
def soup_ce_dense_backward(
    X, LABEL, LSE, DY, DX,
    VOC: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.program_id(1) * TILE + tl.arange(0, TILE)
    label = tl.load(LABEL + row)
    lse = tl.load(LSE + row)
    dy = tl.load(DY + row)
    values = tl.load(X + row * VOC + idx, idx < VOC, other=-float("inf"))
    prob = tl.exp(values - lse)
    dx = (prob - (idx == label).to(tl.float32)) * dy
    dx = tl.where(label >= 0, dx, 0.0)
    tl.store(DX + row * VOC + idx, dx, idx < VOC)


class _TiledCrossEntropyRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, labels):
        n, voc = x.shape
        splits = triton.cdiv(voc, _TILE)
        partial_m = torch.empty((n, splits), device=x.device, dtype=torch.float32)
        partial_s = torch.empty_like(partial_m)
        lse = torch.empty((n,), device=x.device, dtype=torch.float32)
        row_loss = torch.empty_like(lse)
        soup_ce_split_stats[(n, splits)](
            x, partial_m, partial_s, voc, splits, _TILE,
            num_warps=4, enable_fp_fusion=False,
        )
        soup_ce_row_loss[(n,)](
            x, labels, partial_m, partial_s, lse, row_loss,
            voc, splits, triton.next_power_of_2(splits),
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, labels, lse)
        return row_loss

    @staticmethod
    def backward(ctx, dy):
        x, labels, lse = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None
        n, voc = x.shape
        dx = torch.empty_like(x)
        soup_ce_dense_backward[(n, triton.cdiv(voc, _TILE))](
            x, labels, lse, dy.contiguous(), dx, voc, _TILE,
            num_warps=4, enable_fp_fusion=False,
        )
        return dx, None


def cross_entropy_rows(x, labels, reduction="mean", ignore_index=-100):
    """Match PyTorch CE sum/mean for float32 2D tensors; gradients flow to x."""
    if ignore_index != -100:
        raise ValueError("This prototype supports ignore_index=-100 only")
    per_row = _TiledCrossEntropyRows.apply(x, labels)
    total = per_row.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        # PyTorch mean ignores -100 labels. Its all-ignored mean is NaN.
        return total / (labels != -100).sum()
    raise ValueError("Supported reductions: mean, sum")


def patch_qwen_cross_entropy():
    original = F.cross_entropy
    if getattr(original, "_soup_tiled_ce", False):
        return

    def replacement(
        input, target, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0,
    ):
        eligible = (
            isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor)
            and input.is_cuda and input.ndim == 2
            and input.shape[1] == 151936
            and 0 < input.shape[0] <= 2048
            and input.dtype == torch.float32 and input.requires_grad
            and input.is_contiguous()
            and target.is_cuda and target.ndim == 1
            and target.shape[0] == input.shape[0]
            and target.is_contiguous()
            and target.dtype in (torch.int64, torch.int32)
            and weight is None and size_average is None and reduce is None
            and ignore_index == -100 and label_smoothing == 0.0
            and reduction in ("mean", "sum")
        )
        if eligible:
            return cross_entropy_rows(input, target, reduction, ignore_index)
        return original(
            input, target, weight=weight, size_average=size_average,
            ignore_index=ignore_index, reduce=reduce, reduction=reduction,
            label_smoothing=label_smoothing,
        )

    replacement._soup_tiled_ce = True
    F.cross_entropy = replacement
