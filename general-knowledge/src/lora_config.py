# general-knowledge/src/lora_config.py
"""
Shared LoRA hyperparameters for inner (TTT) and outer (ReST-EM) loops.

Inner and outer MUST match on:
  - r (rank): LoRA A matrix shape is (r, in_features); required for O-LoRA orthogonality
  - target_modules: same layers → same A matrix keys in U_hist

lora_alpha only scales B and does not change A dimensions; kept equal for consistency.
"""
from __future__ import annotations

from typing import List

# Aligned for O-LoRA: W_SE (outer) and W_task (inner) share the same A shape per layer.
LORA_RANK: int = 32
LORA_ALPHA: int = 64
LORA_DROPOUT: float = 0.0
LORA_TARGET_MODULES: List[str] = ["q_proj", "v_proj"]

# vLLM --max-lora-rank must be >= LORA_RANK
VLLM_MAX_LORA_RANK: int = LORA_RANK


def lora_target_modules_csv() -> str:
    return ",".join(LORA_TARGET_MODULES)
