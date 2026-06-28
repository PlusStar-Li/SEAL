# general-knowledge/src/EM/lora_checkpoint_utils.py
"""Save / load outer (ReST-EM) LoRA checkpoints for O-LoRA experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel

from lora_config import LORA_RANK, LORA_TARGET_MODULES


def default_lora_adapter_dir(output_dir: str) -> str:
    """Default sidecar path: models/iter2 -> models/iter2_lora_adapter."""
    p = Path(output_dir.rstrip("/"))
    return str(p.parent / f"{p.name}_lora_adapter")


def extract_lora_a_matrices(model: PeftModel) -> Dict[str, torch.Tensor]:
    """Return LoRA A matrices keyed by parameter name (CPU copies)."""
    a_mats: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            a_mats[name] = param.detach().cpu().clone()
    validate_lora_a_matrices(a_mats)
    return a_mats


def validate_lora_a_matrices(
    a_mats: Dict[str, torch.Tensor],
    *,
    expected_rank: int = LORA_RANK,
    expected_modules: Optional[list] = None,
) -> None:
    """Raise if A matrix shapes/modules disagree with shared LoRA config."""
    expected_modules = expected_modules or LORA_TARGET_MODULES
    if not a_mats:
        raise ValueError("No LoRA A matrices found")
    for name, tensor in a_mats.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name}: expected 2D A matrix, got shape {tensor.shape}")
        if tensor.shape[0] != expected_rank:
            raise ValueError(
                f"{name}: rank mismatch (A shape {tensor.shape}, expected r={expected_rank})"
            )
        if not any(mod in name for mod in expected_modules):
            raise ValueError(
                f"{name}: module not in expected target_modules {expected_modules}"
            )


def save_lora_a_matrices(model: PeftModel, path: Path) -> int:
    a_mats = extract_lora_a_matrices(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(a_mats, path)
    return len(a_mats)


def save_outer_lora_checkpoint(
    model: PeftModel,
    adapter_dir: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    save_a_matrices: bool = True,
) -> None:
    """
    Persist the trained PEFT adapter before merge_and_unload().

    Writes:
      - adapter_dir/              standard PEFT adapter (A/B + adapter_config.json)
      - adapter_dir/lora_A_matrices.pt   A matrices only (for O-LoRA U_hist)
      - adapter_dir/outer_lora_metadata.json
    """
    out = Path(adapter_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)

    meta = dict(metadata or {})
    meta["adapter_dir"] = str(out.resolve())
    if save_a_matrices:
        n_a = save_lora_a_matrices(model, out / "lora_A_matrices.pt")
        meta["n_lora_a_matrices"] = n_a
    (out / "outer_lora_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_lora_a_matrices(path: str) -> Dict[str, torch.Tensor]:
    try:
        a_mats = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        a_mats = torch.load(path, map_location="cpu")
    validate_lora_a_matrices(a_mats)
    return a_mats
