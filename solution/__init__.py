"""Soup kernel challenge candidate: original RMSNorm, tiled CE and entropy.

Official SwiGLU example is kept for its observed memory savings, but is
NOT submitted as original work. Do not alter its source or claim its time.
"""

from .swiglu import patch_qwen2_mlp
from .rmsnorm import patch_qwen2_rmsnorm
from .vocab_ce import patch_qwen_cross_entropy
from .entropy import patch_trl_entropy
from .argmax_ops import patch_qwen_token_accuracy_argmax


def apply():
    patch_qwen2_mlp()             # Official example; not counted as our kernel.
    patch_qwen2_rmsnorm()         # Original; prior Kaggle check: 1.3% share.
    patch_qwen_cross_entropy()   # New original candidate; must benchmark.
    patch_trl_entropy()          # New original candidate; must benchmark.
    patch_qwen_token_accuracy_argmax()  # Original token accuracy reduction; experimental.
