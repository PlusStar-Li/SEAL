# general-knowledge/src/continual/olora_utils.py
"""O-LoRA utilities: U_hist I/O, orthogonal loss, adapter A extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from safetensors.torch import load_file

from ..EM.lora_checkpoint_utils import load_lora_a_matrices, validate_lora_a_matrices


UHist = List[Dict[str, torch.Tensor]]
UHIST_STORE_VERSION = "u_hist_store_v1"
MANIFEST_NAME = "manifest.json"


def init_u_hist_from_se(u_se_path: Union[str, Path]) -> UHist:
    """Task 0: U_hist <- {U_SE}."""
    return [load_lora_a_matrices(str(u_se_path))]


def save_task_a(a_dict: Dict[str, torch.Tensor], path: Union[str, Path]) -> None:
    """Persist one historical task's final LoRA A matrices (no B, no full U_hist)."""
    validate_lora_a_matrices(a_dict)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_a = {k: v.cpu().clone() for k, v in a_dict.items()}
    torch.save(cpu_a, path)


def load_task_a(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    path = Path(path)
    try:
        a_mats = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        a_mats = torch.load(str(path), map_location="cpu")
    validate_lora_a_matrices(a_mats)
    return a_mats


class UHistStore:
    """
    On-disk U_hist: U_SE referenced once via manifest; each merge task saved
    exactly once as task_XXX.pt (final A only). No redundant full-history blobs.
    """

    def __init__(self, seq_dir: Union[str, Path], u_se_path: Union[str, Path]):
        self.seq_dir = Path(seq_dir)
        self.u_se_path = str(Path(u_se_path).resolve())
        self.manifest_path = self.seq_dir / MANIFEST_NAME

    def init(self) -> None:
        self.seq_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(merge_tasks=[])

    def append_merge_task(self, a_dict: Dict[str, torch.Tensor]) -> int:
        manifest = self._read_manifest()
        merge_tasks: List[str] = list(manifest["merge_tasks"])
        task_idx = len(merge_tasks) + 1
        fname = f"task_{task_idx:03d}.pt"
        save_task_a(a_dict, self.seq_dir / fname)
        merge_tasks.append(fname)
        self._write_manifest(merge_tasks=merge_tasks)
        return 1 + len(merge_tasks)

    def load_u_hist(self) -> UHist:
        manifest = self._read_manifest()
        u_hist: UHist = [load_lora_a_matrices(manifest["u_se_path"])]
        for fname in manifest["merge_tasks"]:
            u_hist.append(load_task_a(self.seq_dir / fname))
        return u_hist

    @property
    def n_tasks(self) -> int:
        manifest = self._read_manifest()
        return 1 + len(manifest["merge_tasks"])

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"U_hist manifest not found: {self.manifest_path}")
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if data.get("format") != UHIST_STORE_VERSION:
            raise ValueError(f"Unsupported U_hist store format in {self.manifest_path}")
        return data

    def _write_manifest(self, merge_tasks: List[str]) -> None:
        payload = {
            "format": UHIST_STORE_VERSION,
            "u_se_path": self.u_se_path,
            "merge_tasks": merge_tasks,
            "n_tasks": 1 + len(merge_tasks),
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def save_u_hist(u_hist: UHist, path: Union[str, Path]) -> None:
    """Legacy: monolithic U_hist snapshot (prefer UHistStore for disk efficiency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_hist = [{k: v.cpu() for k, v in entry.items()} for entry in u_hist]
    torch.save({"u_hist": cpu_hist, "n_tasks": len(cpu_hist)}, path)


def load_u_hist(path_or_dir: Union[str, Path]) -> UHist:
    """
    Load U_hist from:
      - UHistStore directory (manifest.json + task_XXX.pt), or
      - legacy monolithic .pt snapshot.
    """
    loc = Path(path_or_dir)
    if loc.is_dir():
        manifest = loc / MANIFEST_NAME
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            store = UHistStore(loc, data["u_se_path"])
            return store.load_u_hist()
        raise FileNotFoundError(f"No {MANIFEST_NAME} under U_hist dir {loc}")

    data = torch.load(str(loc), map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "u_hist" in data:
        u_hist = data["u_hist"]
    elif isinstance(data, list):
        u_hist = data
    else:
        raise ValueError(f"Unrecognized U_hist format in {loc}")
    for entry in u_hist:
        validate_lora_a_matrices(entry)
    return u_hist


def extract_a_from_adapter_dir(adapter_dir: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Read LoRA A matrices from a PEFT adapter directory."""
    adapter_dir = Path(adapter_dir)
    sidecar = adapter_dir / "lora_A_matrices.pt"
    if sidecar.exists():
        return load_lora_a_matrices(str(sidecar))

    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No adapter weights under {adapter_dir}")
    state = load_file(str(weights_path))
    a_mats = {
        k: v.cpu().clone()
        for k, v in state.items()
        if "lora_A" in k
    }
    validate_lora_a_matrices(a_mats)
    return a_mats


def extract_a_from_peft_model(model: PeftModel) -> Dict[str, torch.Tensor]:
    a_mats: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            a_mats[name] = param.detach().cpu().clone()
    validate_lora_a_matrices(a_mats)
    return a_mats


def compute_ortho_loss(
    model: torch.nn.Module,
    u_hist: UHist,
    *,
    lambda_weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sum over layers and historical tasks: sum_i lam_i * ||A_t U_i^T||_F^2

    ``lambda_weights`` must align with ``u_hist`` (index 0 = U_SE). When omitted,
    every historical matrix uses weight 1.0 (caller applies global lambda_t).
    """
    if not u_hist:
        dev = device or next(model.parameters()).device
        return torch.zeros((), device=dev)

    total = None
    for name, param in model.named_parameters():
        if "lora_A" not in name or not param.requires_grad:
            continue
        a_t = param
        for hist_idx, hist_a in enumerate(u_hist):
            if name not in hist_a:
                continue
            lam = 1.0
            if lambda_weights is not None:
                lam = (
                    float(lambda_weights[hist_idx])
                    if hist_idx < len(lambda_weights)
                    else 1.0
                )
            u = hist_a[name].to(device=a_t.device, dtype=a_t.dtype)
            prod = a_t @ u.transpose(0, 1)
            term = lam * (prod ** 2).sum()
            total = term if total is None else total + term

    if total is None:
        dev = device or next(model.parameters()).device
        return torch.zeros((), device=dev)
    return total


def append_task_a(u_hist: UHist, a_dict: Dict[str, torch.Tensor]) -> UHist:
    validate_lora_a_matrices(a_dict)
    return u_hist + [{k: v.cpu().clone() for k, v in a_dict.items()}]
