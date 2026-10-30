"""Competition solution: official example + original RMSNorm experiment."""
from .swiglu import patch_qwen2_mlp  # Official provided example: NOT own work.
from .rmsnorm import patch_qwen2_rmsnorm


def apply():
    patch_qwen2_mlp()       # Permitted reference sample (does not count as ours).
    patch_qwen2_rmsnorm()   # Original custom Triton operation; must be validated.
