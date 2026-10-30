"""Original Qwen2 RMSNorm forward/backward Triton experiment for Soup Challenge.

Based on the RMSNorm mathematical definition and the *output dtype behaviour* of
Transformers 5.16.1's Qwen2RMSNorm (no third-party kernel copied).

Important: experimental candidate, NOT a validated Kaggle T4 submission.
The competition requires >=5% own-kernel training time and ten full checks.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rms_rows_forward(
    X, W, Y,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token row, reduce hidden dimension in float32."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H

    xf = tl.load(X + row * H + col, valid, other=0).to(tl.float32)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    mean_square = tl.sum(xf * xf, 0) / H
    inv_rms = tl.rsqrt(mean_square + EPS)

    # IMPORTANT: Qwen2RMSNorm rounds the normalized value back to x.dtype
    # BEFORE multiplying by self.weight. A float32 norm weight then promotes
    # the output to float32. Fusing without this cast changes forward outputs.
    normalized = (xf * inv_rms).to(X.dtype.element_ty).to(tl.float32)
    out = normalized * w
    tl.store(Y + row * H + col, out, valid)


@triton.jit
def soup_rms_rows_backward(
    DY, X, W, DX,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Recompute RMS and calculate dx, for frozen norm weights (QLoRA)."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H

    x = tl.load(X + row * H + col, valid, other=0).to(tl.float32)
    w = tl.load(W + col, valid, other=0).to(tl.float32)
    dy = tl.load(DY + row * H + col, valid, other=0).to(tl.float32)

    inv = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)

    # For float16 inputs, derivative through Qwen2's normalized.to(fp16)
    # is rounded to float16 by PyTorch autograd before reaching the fp32
    # RMS calculation. This cast models that piece of the backward graph.
    dz = (dy * w).to(X.dtype.element_ty).to(tl.float32)
    dot = tl.sum(dz * x, 0)
    dx = dz * inv - x * (inv * inv * inv / H) * dot
    tl.store(DX + row * H + col, dx, valid)


class _FrozenRMSNorm(torch.autograd.Function):
    """Custom autograd, norm weights are frozen in competition QLoRA."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        # Calling code enforces CUDA, contiguity, frozen weight, and safe dtype.
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        out_dtype = torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty(x.shape, device=x.device, dtype=out_dtype)
        block = triton.next_power_of_2(hidden)
        soup_rms_rows_forward[(rows,)](
            x, weight, y, hidden, float(eps), block,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, weight)
        ctx.eps = float(eps)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None
        grad = grad_output.contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        dx = torch.empty_like(x)
        soup_rms_rows_backward[(rows,)](
            grad, x, weight, dx, hidden, ctx.eps,
            triton.next_power_of_2(hidden), num_warps=4,
            enable_fp_fusion=False,
        )
        return dx, None, None


def rmsnorm(x, weight, eps):
    """Use custom op only for the frozen-weight path; otherwise reference."""
    return _FrozenRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patch the existing model class before Soup instantiates it.

    The patch includes a strict fallback for contexts outside the fixed task,
    including Soup regression tests, alternative shapes, dtypes, and devices.
    """
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    original = cls.forward
    if getattr(original, "_soup_own_rmsnorm_patch", False):
        return

    def forward(self, hidden_states):
        if not isinstance(hidden_states, torch.Tensor):
            return original(self, hidden_states)
        w = self.weight
        if (
            not hidden_states.is_cuda
            or hidden_states.ndim < 2
            or hidden_states.shape[-1] != 1536
            or hidden_states.numel() == 0
            or not hidden_states.is_contiguous()
            or not w.is_cuda
            or not w.is_contiguous()
            or w.requires_grad
            or hidden_states.dtype not in (torch.float16, torch.float32)
            or w.dtype not in (torch.float16, torch.float32)
            or w.numel() != hidden_states.shape[-1]
        ):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, w, self.variance_epsilon)

    forward._soup_own_rmsnorm_patch = True
    cls.forward = forward
