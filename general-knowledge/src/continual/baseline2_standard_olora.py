# general-knowledge/src/continual/baseline2_standard_olora.py
"""
Baseline 2: Standard O-LoRA (lambda fixed 1.0) on continual self-edit gen forgetting.

Same protocol as continual_self_edit_gen_forgetting (single passage), with:
  - TTT_server_olora inner loop
  - U_hist initialized from iter2 W_SE, appended after each merge train step
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _stats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ..lora_config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK
from .continual_self_edit_gen_forgetting import (
    _merge_after_ttt,
    _resolve_inner_split_newlines,
    _resolve_k_completions,
    save_split_manifest,
    split_train_val_disjoint,
)
from .continual_self_edits import (
    _banner,
    _cleanup_inner_tmp,
    _connect_zmq,
    _inner_tmp_dir,
    _spawn_vllm,
)
from .continual_self_edit_gen_forgetting import (
    _completions_for_article,
    _questions_for_item,
    _single_passage_train_sequences,
)
from .olora_utils import (
    UHistStore,
    append_task_a,
    extract_a_from_adapter_dir,
    init_u_hist_from_se,
)


def _spawn_inner_server_olora(
    vllm_api: str,
    model: str,
    zmq_port: int,
    gpu: str,
    log_dir: Path,
    tag: str,
    *,
    keep_adapter_dir: bool = False,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "general-knowledge.src.inner.TTT_server_olora",
        "--vllm_api_url",
        vllm_api,
        "--model",
        model,
        "--zmq_port",
        str(zmq_port),
    ]
    if keep_adapter_dir:
        cmd.append("--keep_adapter_dir")
    _banner(f"[Inner O-LoRA] GPU {gpu}, ZMQ :{zmq_port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_dir / f"inner_{tag}.log"
    return subprocess.Popen(
        cmd, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT
    )


def _send_round_olora(
    sock,
    train_seqs: List[str],
    questions: List[Dict[str, str]],
    args: argparse.Namespace,
    u_hist_dir: Path,
) -> Dict[str, Any]:
    sock.send_json(
        {
            "train_sequences": train_seqs,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
            "olora": {
                "enabled": True,
                "lambda_t": args.lambda_fixed,
                "gamma": args.gamma,
                "U_hist_dir": str(u_hist_dir),
            },
        }
    )
    rep = sock.recv_json()
    if "error" in rep:
        raise RuntimeError(f"TTT_server_olora error: {rep['error']}")
    return rep


def _adapter_accuracy_olora(
    sock,
    item: Dict[str, Any],
    train_sequences: List[str],
    args: argparse.Namespace,
    u_hist_dir: Path,
) -> float:
    rep = _send_round_olora(
        sock, train_sequences, _questions_for_item(item), args, u_hist_dir
    )
    correct = rep["adapter_correct"]
    return sum(correct) / len(correct) if correct else 0.0


def _u_hist_store_dir(args: argparse.Namespace, seq_idx: int) -> Path:
    return Path(args.output_dir) / "U_hist" / f"seq{seq_idx}"


class _OLoRAServerSession:
    """vLLM + O-LoRA inner server; reads U_hist from on-disk store per request."""

    def __init__(
        self,
        model_path: str,
        args: argparse.Namespace,
        tag: str,
        max_model_len: int,
        u_hist_dir: Path,
        *,
        keep_adapter_dir: bool = False,
    ):
        self.args = args
        self.model_path = model_path
        self.u_hist_dir = u_hist_dir
        self.logs_dir = Path(args.output_dir) / "logs"
        self.vllm = _spawn_vllm(
            model_path,
            "127.0.0.1",
            args.vllm_port,
            args.vllm_gpus,
            self.logs_dir,
            tag,
            args.lora_rank,
            max_model_len,
        )
        from ..utils import set_vllm_api_url

        self.vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(self.vllm_api)
        self.inner = _spawn_inner_server_olora(
            self.vllm_api,
            model_path,
            args.zmq_port,
            args.inner_gpu,
            self.logs_dir,
            tag,
            keep_adapter_dir=keep_adapter_dir,
        )
        self.ctx, self.sock = _connect_zmq(args.zmq_port)

    def _train_sequences_for_item(self, item: Dict[str, Any]) -> List[str]:
        raw = _completions_for_article(
            self.vllm_api,
            self.model_path,
            item,
            self.args,
            self.args.k_completions,
        )
        return _single_passage_train_sequences(
            item, raw, split_newlines=self.args.inner_split_newlines
        )

    def eval_val_task(self, item: Dict[str, Any], label: str) -> float:
        train_sequences = self._train_sequences_for_item(item)
        acc = _adapter_accuracy_olora(
            self.sock, item, train_sequences, self.args, self.u_hist_dir
        )
        title_short = item["title"][:50] + (
            "…" if len(item["title"]) > 50 else ""
        )
        print(f"      [{label}] (O-LoRA) {title_short} acc={acc:.3f}")
        return acc

    def merge_train_task(self, item: Dict[str, Any], label: str) -> None:
        train_sequences = self._train_sequences_for_item(item)
        _adapter_accuracy_olora(
            self.sock, item, train_sequences, self.args, self.u_hist_dir
        )
        title_short = item["title"][:50] + (
            "…" if len(item["title"]) > 50 else ""
        )
        print(f"      [{label}] (O-LoRA merge) {title_short}")

    def close(self) -> None:
        try:
            self.sock.send_json({"cmd": "shutdown"})
            self.sock.recv_json()
        except Exception:
            pass
        self.sock.close()
        self.ctx.term()
        self.inner.terminate()
        self.vllm.terminate()
        self.vllm.wait()
        torch.cuda.empty_cache()


def _eval_all_val_olora(
    model_path: str,
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    tag: str,
    mat_row: List[List[float]],
    u_hist_dir: Path,
) -> None:
    max_model_len = args.max_tokens + 2048
    sess = _OLoRAServerSession(
        model_path,
        args,
        tag,
        max_model_len,
        u_hist_dir,
    )
    for j, v_item in enumerate(val_items):
        acc = sess.eval_val_task(v_item, f"{tag}_val{j}")
        mat_row[j].append(acc)
    sess.close()


def run_one_sequence(
    seq_idx: int,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    K = len(train_items)
    M = len(val_items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(R)]
    u_hist_sizes: List[int] = []

    u_hist_store = UHistStore(_u_hist_store_dir(args, seq_idx), args.u_se_path)
    u_hist_store.init()
    u_hist_dir = u_hist_store.seq_dir
    u_hist = init_u_hist_from_se(args.u_se_path)

    current_model = args.model

    _banner(
        f"[Seq {seq_idx}] Row 0 — iter2 + O-LoRA eval "
        f"(|U_hist|={len(u_hist)}, λ={args.lambda_fixed}, γ={args.gamma})"
    )
    _eval_all_val_olora(
        current_model,
        val_items,
        args,
        f"seq{seq_idx}_row0",
        mat_vals[0],
        u_hist_dir,
    )
    u_hist_sizes.append(len(u_hist))
    print(
        "  row 0 mean val acc: "
        f"{_stats.mean([mat_vals[0][j][-1] for j in range(M)]):.3f}"
    )

    for k in range(K):
        train_item = train_items[k]
        _banner(
            f"[Seq {seq_idx}] Merge {k}/{K - 1} — "
            f"{train_item['title'][:60]}  |U_hist|={len(u_hist)}"
        )
        max_model_len = args.max_tokens + 2048
        sess = _OLoRAServerSession(
            current_model,
            args,
            f"seq{seq_idx}_merge{k}",
            max_model_len,
            u_hist_dir,
            keep_adapter_dir=True,
        )
        sess.merge_train_task(train_item, f"merge_train{k}")
        sess.close()

        adapter_path = _inner_tmp_dir(args.zmq_port) / "final_adapter"
        if not adapter_path.exists():
            raise RuntimeError(
                f"Missing final_adapter after merge {k} "
                f"(expected {adapter_path}); cannot append A_k to U_hist"
            )
        a_k = extract_a_from_adapter_dir(adapter_path)
        u_hist = append_task_a(u_hist, a_k)
        n_tasks = u_hist_store.append_merge_task(a_k)
        print(f"      saved task_{k + 1:03d}.pt → |U_hist|={n_tasks}")

        current_model = _merge_after_ttt(current_model, seq_idx, k, args)

        _banner(
            f"[Seq {seq_idx}] Row {k + 1} — after {k + 1} merge(s), "
            f"|U_hist|={len(u_hist)}"
        )
        _eval_all_val_olora(
            current_model,
            val_items,
            args,
            f"seq{seq_idx}_row{k + 1}",
            mat_vals[k + 1],
            u_hist_dir,
        )
        u_hist_sizes.append(len(u_hist))
        print(
            f"  row {k + 1} mean val acc: "
            f"{_stats.mean([mat_vals[k + 1][j][-1] for j in range(M)]):.3f}"
        )

    from .continual_self_edits import _cleanup_prev_merge_dir

    _cleanup_prev_merge_dir(current_model, str(args.output_dir), "end of sequence")
    _cleanup_inner_tmp(args.zmq_port, "end of sequence")

    mean_mat: List[List[float]] = [[0.0] * M for _ in range(R)]
    std_mat: List[List[float]] = [[0.0] * M for _ in range(R)]
    for r in range(R):
        for i in range(M):
            vals = mat_vals[r][i]
            if vals:
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    print(
        "mean matrix (O-LoRA):\n", json.dumps(mean_mat, indent=2)
    )
    return mean_mat, std_mat, u_hist_sizes


def load_split_from_dir(
    splits_dir: Path, seq_idx: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_dir = splits_dir / f"seq{seq_idx}"
    train_manifest = json.loads(
        (split_dir / "train.json").read_text(encoding="utf-8")
    )
    val_manifest = json.loads(
        (split_dir / "val.json").read_text(encoding="utf-8")
    )
    train_items = train_manifest["articles"]
    val_items = val_manifest["articles"]
    return train_items, val_items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 2: Standard O-LoRA")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="models/iter2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--u_se_path", required=True)
    p.add_argument(
        "--splits_dir",
        default=None,
        help="Reuse train/val splits from Baseline 1 (e.g. .../single/run0/splits)",
    )
    p.add_argument("--n_sequences", type=int, default=3)
    p.add_argument("--n_merge", type=int, default=8)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--vllm_port", type=int, default=8001)
    p.add_argument("--zmq_port", type=int, default=5555)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--lora_rank", type=int, default=LORA_RANK)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")
    p.add_argument("--inner_sft_articles", type=int, default=1)
    p.add_argument("--k_completions", type=int, default=1)
    p.add_argument("--split_newlines", action="store_true", default=None)
    p.add_argument("--no_split_newlines", dest="split_newlines", action="store_false")
    p.add_argument("--olora_mode", default="standard")
    p.add_argument("--lambda_fixed", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.inner_split_newlines = _resolve_inner_split_newlines(args)
    if args.k_completions is None:
        args.k_completions = _resolve_k_completions(args)
    n_merge, n_val = args.n_merge, args.n_val

    _banner(
        "[Baseline 2 Args] "
        + json.dumps(
            {
                **vars(args),
                "n_merge": n_merge,
                "n_val": n_val,
            },
            indent=2,
            default=str,
        )
    )

    gpus = args.gpus.split(",")
    if len(gpus) < 2:
        sys.exit("[!] --gpus must list at least two IDs (vLLM,inner)")
    args.vllm_gpus, args.inner_gpu = gpus[0], gpus[1]

    full_data: List[Dict[str, Any]] = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir) if args.splits_dir else None

    seq_means: List[List[List[float]]] = []
    seq_stds: List[List[List[float]]] = []

    for seq_idx in range(args.n_sequences):
        if splits_dir is not None:
            train_items, val_items = load_split_from_dir(splits_dir, seq_idx)
            print(f"[Seq {seq_idx}] loaded split from {splits_dir}/seq{seq_idx}")
        else:
            seq_rng = __import__("random").Random(args.seed + seq_idx * 10007)
            train_items, val_items = split_train_val_disjoint(
                full_data, n_merge, n_val, seq_rng
            )
        save_split_manifest(out_dir, seq_idx, train_items, val_items)
        mean_mat, std_mat, _ = run_one_sequence(
            seq_idx, train_items, val_items, args
        )
        seq_means.append(mean_mat)
        seq_stds.append(std_mat)

    R = n_merge + 1
    M = n_val
    agg_mean = [[0.0] * M for _ in range(R)]
    agg_std = [[0.0] * M for _ in range(R)]
    for r in range(R):
        for i in range(M):
            vals = [seq_means[s][r][i] for s in range(args.n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    summary = {
        "experiment": "baseline2_standard_olora",
        "olora_mode": args.olora_mode,
        "lambda_fixed": args.lambda_fixed,
        "gamma": args.gamma,
        "u_se_path": args.u_se_path,
        "splits_dir": str(splits_dir) if splits_dir else None,
        "mean_over_sequences": agg_mean,
        "std_over_sequences": agg_std,
        "n_sequences": args.n_sequences,
        "n_merge": n_merge,
        "n_val": n_val,
        "inner_sft_articles": args.inner_sft_articles,
        "k_completions": args.k_completions,
        "split_newlines": args.inner_split_newlines,
        "dataset": args.dataset,
        "base_model": args.model,
        "metric": "val_adapter_accuracy_after_fresh_self_edit_and_olora_ttt",
        "reuse_rate": 0.0,
        "description": (
            "Standard O-LoRA: lambda=1, no task LoRA reuse. "
            "Same protocol as Baseline 1 single-passage forgetting matrix."
        ),
    }
    summary_path = out_dir / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished Baseline 2. Summary → {summary_path}")


if __name__ == "__main__":
    main()
