# general-knowledge/src/continual/baseline_olora_checkpoint.py
"""Checkpoint save/load and driver-log recovery for Baseline 2/3 O-LoRA runs."""
from __future__ import annotations

import json
import re
import shutil
import statistics as _stats
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils import TASK_SIMILARITY_PROMPT_VERSION
from .merge_train_retention import (
    aggregate_lower_tri_over_sequences,
    inner_summary_from_merge_train,
)

CHECKPOINT_FORMAT = "baseline_olora_checkpoint_v1"
LEGACY_CHECKPOINT_FORMAT = "baseline3_checkpoint_v1"
SUPPORTED_CHECKPOINT_FORMATS = {CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT}
CHECKPOINT_FILENAME = "checkpoint_summary.json"

_VAL_RE = re.compile(r"\[seq(\d+)_row(\d+)_val(\d+)\].*acc=([\d.]+)")
_MERGE_BASE_RE = re.compile(
    r"\[seq(\d+)_merge_train_row0 merge-train base\] d(\d+) .* acc=([\d.]+)"
)
_RETENTION_RE = re.compile(
    r"\[merge-train retention\] step (\d+) → ((?:d\d+:[\d.]+ ?)+)"
)
_SEQ_BANNER_RE = re.compile(r"\[Seq (\d+)\]")
_ROW_MEAN_RE = re.compile(r"row (\d+) mean val acc: ([\d.]+)")
_REUSED_RE = re.compile(r"reused .+ — skip U_hist append")


def _is_adaptive(args: Any) -> bool:
    return getattr(args, "olora_mode", "adaptive") == "adaptive"


def _strip_log_prefix(line: str) -> str:
    if "] [pytorch]" in line:
        idx = line.find("] [pytorch]")
        rest = line[idx + len("] [pytorch]") :]
        if "]" in rest:
            rest = rest[rest.index("]") + 1 :]
        return rest.strip()
    return line.strip()


def _parse_retention_tasks(tail: str) -> List[float]:
    accs: List[float] = []
    for part in tail.strip().split():
        if ":" in part:
            accs.append(float(part.split(":", 1)[1]))
    return accs


def parse_driver_log(
    log_path: Path,
    *,
    max_seq_exclusive: Optional[int] = None,
    n_merge: int = 8,
    n_val: int = 8,
) -> Dict[int, Dict[str, Any]]:
    """Parse B2/B3 driver stdout log into per-sequence result dicts."""
    R = n_merge + 1
    K = n_merge
    text = log_path.read_text(encoding="utf-8", errors="replace")

    val_cells: Dict[Tuple[int, int, int], float] = {}
    merge_base: Dict[Tuple[int, int], float] = {}
    retention: Dict[Tuple[int, int, int], float] = {}
    row_means: Dict[Tuple[int, int], float] = {}
    reuse_counts: Dict[int, int] = {}

    current_seq = 0
    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)

        m = _SEQ_BANNER_RE.search(line)
        if m:
            current_seq = int(m.group(1))

        m = _VAL_RE.search(line)
        if m:
            val_cells[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = float(
                m.group(4)
            )
            continue

        m = _MERGE_BASE_RE.search(line)
        if m:
            merge_base[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
            continue

        m = _RETENTION_RE.search(line)
        if m:
            step_k = int(m.group(1))
            accs = _parse_retention_tasks(m.group(2))
            for i, acc in enumerate(accs):
                retention[(current_seq, step_k + 1, i)] = acc
            continue

        m = _ROW_MEAN_RE.search(line)
        if m:
            row_means[(current_seq, int(m.group(1)))] = float(m.group(2))
            continue

        if _REUSED_RE.search(line):
            reuse_counts[current_seq] = reuse_counts.get(current_seq, 0) + 1

    seq_indices = sorted(
        {s for s, _, _ in val_cells}
        | {s for s, _ in merge_base}
        | {s for s, _, _ in retention}
    )
    if max_seq_exclusive is not None:
        seq_indices = [s for s in seq_indices if s < max_seq_exclusive]

    records: Dict[int, Dict[str, Any]] = {}
    for seq_idx in seq_indices:
        val_mean = [[0.0] * n_val for _ in range(R)]
        val_std = [[0.0] * n_val for _ in range(R)]
        complete = True
        for r in range(R):
            for v in range(n_val):
                key = (seq_idx, r, v)
                if key not in val_cells:
                    complete = False
                    continue
                val_mean[r][v] = val_cells[key]
                val_std[r][v] = 0.0

        inner_mean = [[0.0] * K for _ in range(R)]
        inner_std = [[0.0] * K for _ in range(R)]
        for i in range(K):
            key0 = (seq_idx, i)
            if key0 in merge_base:
                inner_mean[0][i] = merge_base[key0]
                inner_std[0][i] = 0.0
            else:
                complete = False
        for step_k in range(K):
            row = step_k + 1
            for i in range(step_k + 1):
                key = (seq_idx, row, i)
                if key in retention:
                    inner_mean[row][i] = retention[key]
                    inner_std[row][i] = 0.0
                else:
                    complete = False

        if len([r for r in range(R) if (seq_idx, r) in row_means]) != R:
            complete = False

        if not complete:
            continue

        records[seq_idx] = {
            "seq_idx": seq_idx,
            "val_forgetting_mean": val_mean,
            "val_forgetting_std": val_std,
            "merge_train_retention_mean": inner_mean,
            "merge_train_retention_std": inner_std,
            "row_mean_val_acc": [
                row_means.get((seq_idx, r), 0.0) for r in range(R)
            ],
            "lambda_per_merge_step": [],
            "reuse_count": reuse_counts.get(seq_idx, 0),
        }

    return records


def load_lambda_steps_from_steps_log(
    steps_log: Path,
    seq_idx: int,
    n_merge: int,
) -> List[float]:
    if not steps_log.exists():
        return []
    lambdas: Dict[int, float] = {}
    for line in steps_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("seq_idx") != seq_idx or rec.get("phase") != "merge_train":
            continue
        step = rec.get("merge_step")
        if step is not None:
            lambdas[int(step)] = float(rec.get("lambda_t", 1.0))
    return [lambdas[k] for k in range(n_merge) if k in lambdas]


def enrich_sequences_from_steps_log(
    sequences: List[Dict[str, Any]],
    steps_log: Path,
    n_merge: int,
    *,
    lambda_fixed: Optional[float] = None,
) -> None:
    for seq in sequences:
        if lambda_fixed is not None:
            seq["lambda_per_merge_step"] = [float(lambda_fixed)] * n_merge
            continue
        seq_idx = seq["seq_idx"]
        lambdas = load_lambda_steps_from_steps_log(steps_log, seq_idx, n_merge)
        if len(lambdas) == n_merge:
            seq["lambda_per_merge_step"] = lambdas


def build_checkpoint_summary(
    *,
    experiment_name: str,
    args: Any,
    sequences: List[Dict[str, Any]],
    n_sequences: int,
    n_merge: int,
    n_val: int,
    status: str = "in_progress",
    recovered_from: Optional[str] = None,
) -> Dict[str, Any]:
    n_completed = len(sequences)
    seq_means = [s["val_forgetting_mean"] for s in sequences]
    seq_stds = [s["val_forgetting_std"] for s in sequences]
    inner_means = [s["merge_train_retention_mean"] for s in sequences]
    inner_stds = [s["merge_train_retention_std"] for s in sequences]
    all_lambda_steps = [s.get("lambda_per_merge_step", []) for s in sequences]
    total_reuse = sum(s.get("reuse_count", 0) for s in sequences)

    R = n_merge + 1
    M = n_val
    agg_mean = [[0.0] * M for _ in range(R)]
    agg_std = [[0.0] * M for _ in range(R)]
    if n_completed:
        for r in range(R):
            for i in range(M):
                vals = [seq_means[s][r][i] for s in range(n_completed)]
                agg_mean[r][i] = _stats.mean(vals)
                agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    inner_agg_mean, inner_agg_std = (
        aggregate_lower_tri_over_sequences(
            inner_means, inner_stds, n_merge, n_completed
        )
        if n_completed
        else (
            [[0.0] * n_merge for _ in range(R)],
            [[0.0] * n_merge for _ in range(R)],
        )
    )

    lambda_mean_per_merge_step: List[float] = []
    for k in range(n_merge):
        vals = [
            all_lambda_steps[s][k]
            for s in range(n_completed)
            if k < len(all_lambda_steps[s])
        ]
        lambda_mean_per_merge_step.append(_stats.mean(vals) if vals else 1.0)

    reuse_rate = (
        total_reuse / (n_completed * n_merge) if n_completed and n_merge else 0.0
    )
    completed_indices = [s["seq_idx"] for s in sequences]
    next_seq = max(completed_indices) + 1 if completed_indices else 0

    summary: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "status": status,
        "experiment": experiment_name,
        "olora_mode": getattr(args, "olora_mode", "adaptive"),
        "gamma": getattr(args, "gamma", 1.0),
        "u_se_path": getattr(args, "u_se_path", ""),
        "splits_dir": getattr(args, "splits_dir", None),
        "dataset": getattr(args, "dataset", ""),
        "base_model": getattr(args, "model", ""),
        "n_sequences": n_sequences,
        "n_merge": n_merge,
        "n_val": n_val,
        "n_sequences_completed": n_completed,
        "completed_sequences": completed_indices,
        "next_sequence": next_seq,
        "reuse_rate": reuse_rate,
        "lambda_mean_per_merge_step": lambda_mean_per_merge_step,
        "mean_over_sequences": agg_mean,
        "std_over_sequences": agg_std,
        "sequences": sequences,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    if _is_adaptive(args):
        summary.update(
            {
                "lambda_source": getattr(args, "lambda_source", "gpt4"),
                "similarity_model": (
                    "gpt-4.1"
                    if getattr(args, "lambda_source", "gpt4") == "gpt4"
                    else getattr(args, "embed_model", "")
                ),
                "similarity_prompt_version": TASK_SIMILARITY_PROMPT_VERSION,
                "embed_model": getattr(args, "embed_model", ""),
                "tau": getattr(args, "tau", 0.5),
                "fixed_lambda_t": getattr(args, "fixed_lambda_t", None),
                "metric": (
                    "val_adapter_accuracy_after_fresh_self_edit_and_adaptive_olora_ttt"
                ),
            }
        )
    else:
        summary.update(
            {
                "lambda_fixed": getattr(args, "lambda_fixed", 1.0),
                "metric": "val_adapter_accuracy_after_fresh_self_edit_and_olora_ttt",
            }
        )

    summary.update(
        inner_summary_from_merge_train(
            inner_agg_mean,
            inner_agg_std,
            n_sequences=n_completed,
            n_merge=n_merge,
        )
    )
    if recovered_from:
        summary["recovered_from"] = recovered_from
    return summary


def save_checkpoint(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Checkpoint] saved → {path}")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Unsupported checkpoint format in {path}: {data.get('format')}"
        )
    return data


def validate_checkpoint_for_resume(
    checkpoint: Dict[str, Any],
    start_seq: int,
    args: Any,
) -> None:
    completed = checkpoint.get("completed_sequences", [])
    expected = list(range(start_seq))
    if completed != expected:
        raise ValueError(
            f"Checkpoint completed_sequences={completed} does not match "
            f"--start_seq {start_seq} (expected {expected})"
        )
    for key in ("n_merge", "n_val", "n_sequences"):
        ck = checkpoint.get(key)
        arg = getattr(args, key)
        if ck != arg:
            raise ValueError(
                f"Checkpoint {key}={ck} disagrees with run args {key}={arg}"
            )
    ck_mode = checkpoint.get("olora_mode", "adaptive")
    if ck_mode != args.olora_mode:
        raise ValueError(
            f"Checkpoint olora_mode={ck_mode} != args.olora_mode={args.olora_mode}"
        )
    if _is_adaptive(args):
        if checkpoint.get("tau") != args.tau:
            raise ValueError(
                f"Checkpoint tau={checkpoint.get('tau')} != args.tau={args.tau}"
            )
    else:
        if checkpoint.get("lambda_fixed") != args.lambda_fixed:
            raise ValueError(
                "Checkpoint lambda_fixed="
                f"{checkpoint.get('lambda_fixed')} != "
                f"args.lambda_fixed={args.lambda_fixed}"
            )
        if checkpoint.get("gamma") != args.gamma:
            raise ValueError(
                f"Checkpoint gamma={checkpoint.get('gamma')} != args.gamma={args.gamma}"
            )


def cleanup_sequence_artifacts(output_dir: Path, seq_idx: int) -> None:
    out = Path(output_dir)
    targets = [
        out / "U_hist" / f"seq{seq_idx}",
        out / "task_bank" / f"seq{seq_idx}",
        out / "splits" / f"seq{seq_idx}",
    ]
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
            print(f"[Resume] removed {path}")

    for merged in out.glob(f"merged_seq{seq_idx}_step*"):
        if merged.is_dir():
            shutil.rmtree(merged)
            print(f"[Resume] removed {merged}")


def filter_steps_log(steps_log: Path, max_seq_inclusive: int) -> None:
    if not steps_log.exists():
        return
    kept: List[str] = []
    for line in steps_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec.get("seq_idx", 0)) <= max_seq_inclusive:
            kept.append(line)
    steps_log.write_text(
        ("\n".join(kept) + "\n") if kept else "",
        encoding="utf-8",
    )
    print(
        f"[Resume] {steps_log.name}: kept seq 0..{max_seq_inclusive} "
        f"({len(kept)} lines)"
    )


# Backward-compatible alias
filter_adaptive_log = filter_steps_log


def sequence_record_from_run(
    seq_idx: int,
    mean_mat: List[List[float]],
    std_mat: List[List[float]],
    inner_mean: List[List[float]],
    inner_std: List[List[float]],
    lambda_steps: List[float],
    reuse_count: int,
    mat_vals: Optional[List[List[List[float]]]] = None,
) -> Dict[str, Any]:
    M = len(mean_mat[0]) if mean_mat else 0
    if mat_vals is not None:
        row_means = []
        for r in range(len(mat_vals)):
            vals = [mat_vals[r][j][-1] for j in range(M) if mat_vals[r][j]]
            row_means.append(_stats.mean(vals) if vals else 0.0)
    else:
        row_means = [_stats.mean(row) for row in mean_mat]

    return {
        "seq_idx": seq_idx,
        "val_forgetting_mean": mean_mat,
        "val_forgetting_std": std_mat,
        "merge_train_retention_mean": inner_mean,
        "merge_train_retention_std": inner_std,
        "row_mean_val_acc": row_means,
        "lambda_per_merge_step": lambda_steps,
        "reuse_count": reuse_count,
    }


def recover_checkpoint_from_log(
    log_path: Path,
    output_dir: Path,
    *,
    max_seq_exclusive: int,
    n_sequences: int = 8,
    n_merge: int = 8,
    n_val: int = 8,
    experiment_name: str = "baseline3_gpt4_adaptive_olora",
    args_overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    class _Args:
        pass

    args = _Args()
    defaults: Dict[str, Any] = {
        "olora_mode": "adaptive",
        "lambda_source": "gpt4",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "tau": 0.3,
        "gamma": 0.3,
        "lambda_fixed": 1.0,
        "fixed_lambda_t": None,
        "u_se_path": "models/iter2_lora_adapter/lora_A_matrices.pt",
        "splits_dir": (
            "general-knowledge/results/baselines/baseline1_vanilla_sft/run0/splits"
        ),
        "dataset": "general-knowledge/data/squad_val.json",
        "model": "models/iter2",
        "n_sequences": n_sequences,
        "n_merge": n_merge,
        "n_val": n_val,
    }
    defaults.update(args_overrides or {})
    for k, v in defaults.items():
        setattr(args, k, v)

    records = parse_driver_log(
        log_path,
        max_seq_exclusive=max_seq_exclusive,
        n_merge=n_merge,
        n_val=n_val,
    )
    sequences = [records[i] for i in sorted(records)]
    steps_log = output_dir / "adaptive_steps.jsonl"
    enrich_sequences_from_steps_log(
        sequences,
        steps_log,
        n_merge,
        lambda_fixed=(
            float(args.lambda_fixed)
            if getattr(args, "olora_mode", "adaptive") == "standard"
            else None
        ),
    )

    summary = build_checkpoint_summary(
        experiment_name=experiment_name,
        args=args,
        sequences=sequences,
        n_sequences=n_sequences,
        n_merge=n_merge,
        n_val=n_val,
        status="in_progress",
        recovered_from=str(log_path),
    )
    out_path = output_dir / CHECKPOINT_FILENAME
    save_checkpoint(out_path, summary)
    return out_path
