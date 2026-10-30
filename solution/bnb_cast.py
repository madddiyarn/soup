"""Experimental original Triton casts in bitsandbytes Linear4bit's training forward.

This substitutes only the two dtype conversions in the normal GPU/biasless
Linear4bit path; the bitsandbytes 4-bit GEMM and its quantization stay intact.
No changes to the task config, data, trainable weights, or optimizer.

Source for integration semantics (not kernel code):
  https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py

Candidate only: must pass the official GPU harness and >=5% own-kernel check.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_qlora_dtype_cast(IN, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    """Memory-coalesced dtype conversion, needed by QLoRA's 4-bit linears."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(IN + offs, mask=offs < N, other=0)
    tl.store(OUT + offs, values.to(OUT.dtype.element_ty), mask=offs < N)


def _launch_cast(x, dtype):
    """Only CUDA contiguous floating tensor dtype change, internal to this patch."""
    if x.dtype == dtype:
        return x
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    n = x.numel()
    soup_qlora_dtype_cast[(triton.cdiv(n, 4096),)](
        x, out, n, BLOCK=4096, num_warps=4,
    )
    return out


class _CastAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        ctx.src_dtype = x.dtype
        return _launch_cast(x, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Like PyTorch's CopyBackwards: gradients are converted to source dtype.
        if grad_output is None:
            return None, None
        grad = grad_output.contiguous()
        return _launch_cast(grad, ctx.src_dtype), None


def qlora_cast(x, dtype):
    """Differentiable dtype conversion; fallback preserves unusual layouts."""
    if (
        type(x) is not torch.Tensor
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (torch.float16, torch.float32)
        or dtype not in (torch.float16, torch.float32)
    ):
        return x.to(dtype)
    if x.dtype == dtype:
        return x
    return _CastAutograd.apply(x, dtype)


def patch_bnb_linear4bit_casts():
    """Conservatively patch the exact fixed-task QLoRA GPU Linear4bit path.

    Every other case uses the library's original forward; only the real cast
    operations in a quantized linear layer are replaced. No global torch patch.
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import modules as bnb_modules

    cls = bnb.nn.Linear4bit
    original = cls.forward
    if getattr(original, '_soup_qlora_cast_patch', False):
        return

    def forward(self, x):
        # Only training, fixed Qwen feature sizes, BF16 excluded, no bias.
        eligible = (
            isinstance(x, torch.Tensor)
            and type(x) is torch.Tensor
            and self.training
            and x.is_cuda
            and x.is_contiguous()
            and x.ndim in (2, 3)
            and x.shape[-1] in (1536, 8960)
            and x.numel() >= 4096
            and x.dtype == torch.float32
            and self.compute_dtype == torch.float16
            and self.bias is None
            and self.weight is not None
            and getattr(self.weight, 'quant_state', None) is not None
        )
        if not eligible:
            return original(self, x)

        # Same state transition as the installed bitsandbytes 0.50.2 code.
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        # The two exact casts from Linear4bit.forward are implemented in Triton.
        converted = qlora_cast(x, self.compute_dtype)
        result = bnb.matmul_4bit(
            converted, self.weight, bias=None, quant_state=self.weight.quant_state,
        )
        return qlora_cast(result, x.dtype)

    forward._soup_qlora_cast_patch = True
    forward._soup_qlora_cast_original = original
    cls.forward = forward
