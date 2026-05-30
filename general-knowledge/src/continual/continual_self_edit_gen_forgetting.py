# general-knowledge/src/continual/continual_self_edit_gen_forgetting.py
"""
Experiment: forgetting of *self-edit generation* under continual inner merge.

Unlike continual_self_edits.py (which reuses the latest step's SE when evaluating
d_0..d_k), this script **regenerates** a fresh self-edit for every task after each
merge, runs per-task inner TTT, and records accuracy on that task only.

Protocol (K datapoints, S sequences — same layout as continual_self_edits):

  * Row 0 (base policy, no merge yet): for each task j = 0..K-1,
      generate SE with the initial model → inner TTT → accuracy on task j.

  * After merge step k on task k (same as continual_self_edits):
      for each j = 0..k, regenerate SE with the **current** merged model,
      inner TTT on that completion, evaluate on task j only → row k+1, col j.

Output: (K+1)×K lower-triangular matrices + summary JSON (compatible with
average_results.py and plot_self_edit_gen_forgetting.py).

Usage:
  python -m general-knowledge.src.continual.continual_self_edit_gen_forgetting \\
    --dataset general-knowledge/data/squad_val.json \\
    --output_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics as _stats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import torch

from ..data_generation.make_squad_data import make_prompt
from ..utils import build_train_sequences, set_vllm_api_url
from .continual_self_edits import (
    _banner,
    _connect_zmq,
    _merge_lora,
    _send_round,
    _spawn_inner_server,
    _spawn_vllm,
)

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _questions_for_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "title": item["title"],
            "context": item["context"],
            "question": f"Topic: {item['title']}\n{q['question']}",
            "answer": q["answer"],
        }
        for q in item["questions"]
    ]


def _generate_completion(
    vllm_api: str,
    model_path: str,
    item: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    prompt = make_prompt(item["title"], item["context"], instruct_model=False)
    resp = requests.post(
        f"{vllm_api}/v1/completions",
        json={
            "model": model_path,
            "prompt": [prompt],
            "n": 1,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def _adapter_accuracy(
    sock,
    item: Dict[str, Any],
    completion: str,
    args: argparse.Namespace,
) -> float:
    train_sequences = build_train_sequences(
        completion or item["context"],
        item["context"],
        item["title"],
        split_newlines=True,
    )
    rep = _send_round(sock, train_sequences, _questions_for_item(item), args)
    correct = rep["adapter_correct"]
    return sum(correct) / len(correct) if correct else 0.0


class _ServerSession:
    """One vLLM + inner-server session for multiple generate→TTT calls."""

    def __init__(
        self,
        model_path: str,
        args: argparse.Namespace,
        tag: str,
        max_model_len: int,
    ):
        self.args = args
        self.model_path = model_path
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
        self.vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(self.vllm_api)
        self.inner = _spawn_inner_server(
            self.vllm_api,
            model_path,
            args.zmq_port,
            args.inner_gpu,
            self.logs_dir,
            tag,
        )
        self.ctx, self.sock = _connect_zmq(args.zmq_port)

    def eval_task(self, item: Dict[str, Any], label: str) -> Tuple[float, str]:
        completion = _generate_completion(
            self.vllm_api, self.model_path, item, self.args
        )
        acc = _adapter_accuracy(self.sock, item, completion, self.args)
        print(f"      [{label}] {item['title'][:50]}… acc={acc:.3f}")
        return acc, completion

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


def _merge_after_ttt(
    current_model: str,
    seq_idx: int,
    step_k: int,
    args: argparse.Namespace,
) -> str:
    """Merge the adapter produced by the most recent inner TTT round."""
    adapter_path = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0/final_adapter")
    if not adapter_path.exists():
        print(f"[!] merge step {step_k}: adapter missing — keeping model")
        return current_model

    merged_dir = Path(args.output_dir) / f"merged_seq{seq_idx}_step{step_k}"
    prev = current_model
    new_path = _merge_lora(current_model, adapter_path, merged_dir)
    if (
        step_k > 0
        and Path(prev).is_dir()
        and str(prev).startswith(str(Path(args.output_dir)))
    ):
        try:
            shutil.rmtree(prev)
        except OSError as exc:
            print(f"[Cleanup] {exc}")
    return new_path


def run_one_sequence(
    seq_idx: int,
    items: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[List[float]], List[List[float]]]:
    K = len(items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(K)] for _ in range(R)]
    out_dir = Path(args.output_dir)

    current_model = args.model
    max_model_len = args.max_tokens + 2048

    # ---- Row 0: initial model, fresh SE + TTT per task (no merge) ------------
    _banner(f"[Seq {seq_idx}] Row 0 — base model, regenerate+TTT each task")
    sess = _ServerSession(current_model, args, f"seq{seq_idx}_row0", max_model_len)
    for j, item in enumerate(items):
        acc, _ = sess.eval_task(item, f"row0_d{j}")
        mat_vals[0][j].append(acc)
    sess.close()

    # ---- Steps k = 0..K-1: merge on k, then regen-eval tasks 0..k ------------
    for k, item in enumerate(items):
        _banner(f"[Seq {seq_idx}] Merge step {k}/{K - 1} — {item['title']}")

        sess = _ServerSession(
            current_model, args, f"seq{seq_idx}_mergegen{k}", max_model_len
        )
        sess.eval_task(item, f"merge_train_d{k}")
        sess.close()

        current_model = _merge_after_ttt(current_model, seq_idx, k, args)

        _banner(f"[Seq {seq_idx}] Row {k + 1} — regen+TTT tasks 0..{k}")
        sess = _ServerSession(
            current_model, args, f"seq{seq_idx}_row{k + 1}", max_model_len
        )
        for j in range(k + 1):
            acc, _ = sess.eval_task(items[j], f"row{k + 1}_d{j}")
            mat_vals[k + 1][j].append(acc)
        sess.close()

        print(
            f"  row {k + 1} accs: "
            + ", ".join(f"d{j}={mat_vals[k + 1][j][-1]:.3f}" for j in range(k + 1))
        )

    # Cleanup final merge dir (same as continual_self_edits)
    last_merge = Path(current_model)
    if last_merge.is_dir() and str(last_merge).startswith(str(out_dir)):
        try:
            shutil.rmtree(last_merge)
        except OSError as exc:
            print(f"[Cleanup] final merge: {exc}")

    tmp_dir = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0")
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
        except OSError as exc:
            print(f"[Cleanup] tmp adapter: {exc}")

    mean_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    std_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = mat_vals[r][i]
            if vals:
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    print("mean matrix:\n", json.dumps(mean_mat, indent=2))
    return mean_mat, std_mat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual self-edit *generation* forgetting experiment"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_sequences", type=int, default=8)
    p.add_argument("--n_datapoints", type=int, default=8)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--vllm_port", type=int, default=8001)
    p.add_argument("--zmq_port", type=int, default=5555)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")
    p.add_argument(
        "--output_dir",
        default="general-knowledge/results/continual_self_edit_gen_forgetting",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _banner("[Args] " + json.dumps(vars(args), indent=2))
    random.seed(args.seed)

    gpus = args.gpus.split(",")
    if len(gpus) < 2:
        sys.exit("[!] --gpus must list at least two IDs (vLLM,inner)")
    args.vllm_gpus, args.inner_gpu = gpus[0], gpus[1]

    full_data: List[Dict[str, Any]] = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )
    if args.n_datapoints > len(full_data):
        sys.exit("[!] n_datapoints exceeds dataset size")

    seq_means: List[List[List[float]]] = []
    seq_stds: List[List[List[float]]] = []

    for seq_idx in range(args.n_sequences):
        items = random.sample(full_data, args.n_datapoints)
        mean_mat, std_mat = run_one_sequence(seq_idx, items, args)
        seq_means.append(mean_mat)
        seq_stds.append(std_mat)

    K = args.n_datapoints
    R = K + 1
    agg_mean = [[0.0] * K for _ in range(R)]
    agg_std = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = [seq_means[s][r][i] for s in range(args.n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "continual_self_edit_gen_forgetting",
        "mean_over_sequences": agg_mean,
        "std_over_sequences": agg_std,
        "n_sequences": args.n_sequences,
        "n_datapoints": args.n_datapoints,
        "dataset": args.dataset,
        "base_model": args.model,
        "metric": "per_task_adapter_accuracy_after_fresh_self_edit",
        "description": (
            "Row 0: base model regenerates SE + TTT per task. "
            "Row r>0: after merge steps 0..r-1, regenerate SE + TTT for tasks 0..r-1."
        ),
    }
    summary_path = out_dir / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished. Summary → {summary_path}")


if __name__ == "__main__":
    main()
