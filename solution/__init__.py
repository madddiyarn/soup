"""Soup Kernel Challenge original kernel candidate v4.

Original: RMSNorm, tiled CE, entropy, argmax, QLoRA dtype cast.
The official SwiGLU example remains unmodified and is NOT our own kernel.
"""
from .swiglu import patch_qwen2_mlp
from .rmsnorm import patch_qwen2_rmsnorm
from .vocab_ce import patch_qwen_cross_entropy
from .entropy import patch_trl_entropy
from .argmax_ops import patch_qwen_token_accuracy_argmax
from .bnb_cast import patch_bnb_linear4bit_casts


def apply():
    patch_qwen2_mlp()                       # official example, not counted
    patch_qwen2_rmsnorm()
    patch_qwen_cross_entropy()
    patch_trl_entropy()
    patch_qwen_token_accuracy_argmax()
    patch_bnb_linear4bit_casts()             # new original v4 candidate
