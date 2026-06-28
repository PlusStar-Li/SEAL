# general-knowledge/src/continual/continual_self_edit_gen_forgetting.py
"""
Self-edit *generation* forgetting under continual inner merge (held-out validation).

Single vs CPT differ only in **articles per task** and **split_newlines**; train/val
logic is parallel:

- **Train (merge)**: ``n_merge`` disjoint tasks. Single = 1 article/task; CPT = N
  articles/task. Step k: fresh SE → inner TTT on task k → merge adapter.
- **Val (eval)**: ``n_val`` disjoint held-out tasks (never merged). Same structure;
  each checkpoint re-runs fresh SE + inner TTT + eval on every val task.

Single: per-passage adapter acc. CPT: overall adapter acc on all questions in the
task corpus (``k_completions=5``, ``split_newlines=False``, CPT.sh hyperparams).

Matrix shape: (n_merge+1) × n_val.

Usage:
  python -m general-knowledge.src.continual.continual_self_edit_gen_forgetting \\
    --dataset general-knowledge/data/squad_val.json \\
    --n_merge 8 --n_val 8 --inner_sft_articles 1 \\
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
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import torch

from ..data_generation.make_squad_data import make_prompt
from ..lora_config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK
from ..utils import build_train_sequences, set_vllm_api_url
from .continual_self_edits import (
    _banner,
    _cleanup_inner_tmp,
    _cleanup_prev_merge_dir,
    _connect_zmq,
    _inner_tmp_dir,
    _merge_lora,
    _send_round,
    _spawn_inner_server,
    _spawn_vllm,
)

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _resolve_inner_split_newlines(args: argparse.Namespace) -> bool:
    """
    CPT.sh uses SPLIT_NEWLINES=0 (split_newlines=False).
    continual_self_edits single-passage inner loop uses split_newlines=True.
    """
    if args.split_newlines is not None:
        return args.split_newlines
    return args.inner_sft_articles <= 1


def _resolve_k_completions(args: argparse.Namespace) -> int:
    """CPT.sh uses k_completions=5; single-passage continual uses 1."""
    if args.k_completions is not None:
        return args.k_completions
    return 5 if _cpt_inner_mode(args) else 1


def _cpt_inner_mode(args: argparse.Namespace) -> bool:
    return args.inner_sft_articles > 1


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


def _stored_completions_from_item(
    item: Dict[str, Any], k_completions: int
) -> Optional[List[str]]:
    """Pick up to k completions from dataset fields (same logic as CPT.py)."""
    if k_completions == 0:
        return [""]
    comps: List[str] = []
    completion_key = (
        "completions"
        if "completions" in item
        else "pair_completions"
        if "pair_completions" in item
        else None
    )
    if completion_key:
        raw = item.get(completion_key)
        if isinstance(raw, list):
            comps.extend([c for c in raw if isinstance(c, str) and c.strip()])
        elif isinstance(raw, str) and raw.strip():
            comps.append(raw)
    if "triple_completions" in item:
        triple = item["triple_completions"]
        if isinstance(triple, list):
            comps.extend([c for c in triple if isinstance(c, str) and c.strip()])
    comps = [c for c in comps if c.strip()]
    if not comps:
        return None
    picked = comps[:k_completions]
    return picked if picked else [""]


def _generate_completions_live(
    vllm_api: str,
    model_path: str,
    item: Dict[str, Any],
    args: argparse.Namespace,
    k_completions: int,
) -> List[str]:
    """Sample k self-edit completions from vLLM (``n=k``), matching CPT multiplicity."""
    if k_completions == 0:
        return [""]
    prompt = make_prompt(item["title"], item["context"], instruct_model=False)
    resp = requests.post(
        f"{vllm_api}/v1/completions",
        json={
            "model": model_path,
            "prompt": [prompt],
            "n": k_completions,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        timeout=600,
    )
    resp.raise_for_status()
    choices = resp.json()["choices"]
    return [c["text"].strip() for c in choices[:k_completions]]


def _completions_for_article(
    vllm_api: str,
    model_path: str,
    item: Dict[str, Any],
    args: argparse.Namespace,
    k_completions: int,
) -> List[str]:
    stored = _stored_completions_from_item(item, k_completions)
    if stored is not None:
        return stored
    return _generate_completions_live(
        vllm_api, model_path, item, args, k_completions
    )


def _train_sequences_from_completion_list(
    item: Dict[str, Any],
    raw_completions: List[str],
    *,
    split_newlines: bool,
) -> List[str]:
    """Build train sequences like CPT.py (add_context only for first completion)."""
    train_sequences: List[str] = []
    for i, comp in enumerate(raw_completions):
        train_sequences.extend(
            build_train_sequences(
                comp,
                item["context"],
                item["title"],
                split_newlines=split_newlines,
                add_context=(i == 0),
            )
        )
    return train_sequences


def _aggregate_train_sequences(
    corpus: List[Dict[str, Any]],
    completions_by_article: Dict[Tuple[str, str], List[str]],
    *,
    split_newlines: bool,
) -> List[str]:
    """CPT-style corpus: concatenate train sequences from all passages."""
    train_sequences: List[str] = []
    for art in corpus:
        key = _article_key(art)
        raw = completions_by_article.get(key)
        if not raw:
            raw = [""]
        train_sequences.extend(
            _train_sequences_from_completion_list(
                art, raw, split_newlines=split_newlines
            )
        )
    return train_sequences


def _questions_for_corpus(corpus: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """All eval questions from each article (CPT.py aggregation order)."""
    questions: List[Dict[str, str]] = []
    for item in corpus:
        questions.extend(_questions_for_item(item))
    return questions


def _adapter_accuracy(
    sock,
    item: Dict[str, Any],
    train_sequences: List[str],
    args: argparse.Namespace,
) -> float:
    rep = _send_round(sock, train_sequences, _questions_for_item(item), args)
    correct = rep["adapter_correct"]
    return sum(correct) / len(correct) if correct else 0.0


def _adapter_accuracy_corpus(
    sock,
    corpus: List[Dict[str, Any]],
    train_sequences: List[str],
    args: argparse.Namespace,
) -> float:
    """CPT-style: train on aggregated SEs, eval on all corpus questions."""
    eval_questions = _questions_for_corpus(corpus)
    rep = _send_round(sock, train_sequences, eval_questions, args)
    correct = rep["adapter_correct"]
    return sum(correct) / len(correct) if correct else 0.0


def _single_passage_train_sequences(
    item: Dict[str, Any],
    raw_completions: List[str],
    *,
    split_newlines: bool,
) -> List[str]:
    if not raw_completions:
        raw_completions = [""]
    return _train_sequences_from_completion_list(
        item, raw_completions, split_newlines=split_newlines
    )


class _ServerSession:
    """One vLLM + inner-server session for multiple generate→TTT calls."""

    def __init__(
        self,
        model_path: str,
        args: argparse.Namespace,
        tag: str,
        max_model_len: int,
        *,
        cpt_corpus: Optional[List[Dict[str, Any]]] = None,
        keep_adapter_dir: bool = False,
    ):
        self.args = args
        self.model_path = model_path
        self.cpt_corpus: List[Dict[str, Any]] = list(cpt_corpus or [])
        self._cpt_completions: Optional[Dict[Tuple[str, str], List[str]]] = None
        self._cpt_train_sequences: Optional[List[str]] = None
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
            keep_adapter_dir=keep_adapter_dir,
        )
        self.ctx, self.sock = _connect_zmq(args.zmq_port)

    def _use_cpt_inner(self) -> bool:
        return _cpt_inner_mode(self.args)

    def _reset_cpt_cache(self) -> None:
        self._cpt_completions = None
        self._cpt_train_sequences = None

    def set_cpt_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        """Switch held-out CPT val task; force fresh SE generation on next eval."""
        self.cpt_corpus = list(corpus)
        self._reset_cpt_cache()

    def _ensure_cpt_train_sequences(self) -> List[str]:
        if self._cpt_train_sequences is not None:
            return self._cpt_train_sequences
        if not self.cpt_corpus:
            sys.exit("[!] inner_sft_articles>1 but cpt_corpus is empty")
        k = self.args.k_completions
        print(
            f"      [CPT inner] generating k={k} SE per article for "
            f"{len(self.cpt_corpus)} corpus articles…"
        )
        completions: Dict[Tuple[str, str], List[str]] = {}
        for idx, art in enumerate(self.cpt_corpus):
            completions[_article_key(art)] = _completions_for_article(
                self.vllm_api,
                self.model_path,
                art,
                self.args,
                k,
            )
            if (idx + 1) % 20 == 0 or idx + 1 == len(self.cpt_corpus):
                print(f"      [CPT inner] generated {idx + 1}/{len(self.cpt_corpus)}")
        self._cpt_completions = completions
        self._cpt_train_sequences = _aggregate_train_sequences(
            self.cpt_corpus,
            completions,
            split_newlines=self.args.inner_split_newlines,
        )
        print(
            f"      [CPT inner] {len(self._cpt_train_sequences):,} "
            f"aggregated train sequences "
            f"(k_completions={k}, split_newlines={self.args.inner_split_newlines})"
        )
        return self._cpt_train_sequences

    def _train_sequences_for_item(
        self, item: Dict[str, Any]
    ) -> List[str]:
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
        acc = _adapter_accuracy(self.sock, item, train_sequences, self.args)
        title_short = item["title"][:50] + ("…" if len(item["title"]) > 50 else "")
        print(f"      [{label}] (single) {title_short} acc={acc:.3f}")
        return acc

    def eval_cpt_corpus(self, label: str) -> float:
        """CPT validation: overall adapter acc on all corpus questions."""
        train_sequences = self._ensure_cpt_train_sequences()
        n_q = len(_questions_for_corpus(self.cpt_corpus))
        acc = _adapter_accuracy_corpus(
            self.sock, self.cpt_corpus, train_sequences, self.args
        )
        print(
            f"      [{label}] (CPT eval) {len(self.cpt_corpus)} articles, "
            f"{n_q} questions, overall acc={acc:.3f}"
        )
        return acc

    def merge_train_task(self, item: Dict[str, Any], label: str) -> None:
        """Single mode: SE + inner TTT on one merge-stream article."""
        train_sequences = self._train_sequences_for_item(item)
        _adapter_accuracy(self.sock, item, train_sequences, self.args)
        title_short = item["title"][:50] + ("…" if len(item["title"]) > 50 else "")
        print(f"      [{label}] (single) merge train {title_short}")

    def merge_train_corpus(self, corpus: List[Dict[str, Any]], label: str) -> None:
        """CPT mode: SE on all N articles → aggregated inner TTT (adapter for merge)."""
        self.set_cpt_corpus(corpus)
        train_sequences = self._ensure_cpt_train_sequences()
        _adapter_accuracy_corpus(self.sock, corpus, train_sequences, self.args)
        print(
            f"      [{label}] (CPT) merge train on {len(corpus)} articles, "
            f"{len(train_sequences):,} train sequences"
        )

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
    adapter_path = _inner_tmp_dir(args.zmq_port) / "final_adapter"
    if not adapter_path.exists():
        print(f"[!] merge step {step_k}: adapter missing — keeping model")
        _cleanup_inner_tmp(args.zmq_port, "stale after missing adapter")
        return current_model

    merged_dir = Path(args.output_dir) / f"merged_seq{seq_idx}_step{step_k}"
    prev = current_model
    new_path = _merge_lora(current_model, adapter_path, merged_dir)
    _cleanup_inner_tmp(args.zmq_port, f"merge step {step_k}")
    if step_k > 0:
        _cleanup_prev_merge_dir(prev, args.output_dir, f"merge step {step_k}")
    return new_path


def _eval_all_val(
    model_path: str,
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    tag: str,
    mat_row: List[List[float]],
    *,
    val_cpt_corpora: Optional[List[List[Dict[str, Any]]]] = None,
) -> None:
    max_model_len = args.max_tokens + 2048
    if _cpt_inner_mode(args) and val_cpt_corpora:
        for j, corpus in enumerate(val_cpt_corpora):
            sess = _ServerSession(
                model_path,
                args,
                f"{tag}_val{j}",
                max_model_len,
                cpt_corpus=corpus,
            )
            acc = sess.eval_cpt_corpus(f"{tag}_val{j}")
            mat_row[j].append(acc)
            sess.close()
        return

    sess = _ServerSession(model_path, args, tag, max_model_len)
    for j, v_item in enumerate(val_items):
        acc = sess.eval_val_task(v_item, f"{tag}_val{j}")
        mat_row[j].append(acc)
    sess.close()


def _article_key(item: Dict[str, Any]) -> Tuple[str, str]:
    """Unique passage id (title may repeat across squad_val)."""
    return (item["title"], item["context"])


def split_train_val_cpt_disjoint(
    full_data: List[Dict[str, Any]],
    n_merge: int,
    n_val: int,
    n_articles_per_task: int,
    rng: random.Random,
) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Sample n_merge train + n_val val CPT tasks (each task = n_articles_per_task
  articles), all disjoint — mirrors split_train_val_disjoint for single mode.
    """
    need = (n_merge + n_val) * n_articles_per_task
    if need > len(full_data):
        sys.exit(
            f"[!] (n_merge({n_merge}) + n_val({n_val})) × "
            f"inner_sft_articles({n_articles_per_task}) = {need} exceeds "
            f"dataset size {len(full_data)}"
        )
    picked = rng.sample(full_data, need)
    train_corpora: List[List[Dict[str, Any]]] = []
    val_corpora: List[List[Dict[str, Any]]] = []
    idx = 0
    for _ in range(n_merge):
        train_corpora.append(picked[idx : idx + n_articles_per_task])
        idx += n_articles_per_task
    for _ in range(n_val):
        val_corpora.append(picked[idx : idx + n_articles_per_task])
        idx += n_articles_per_task
    train_keys = {_article_key(a) for c in train_corpora for a in c}
    val_keys = {_article_key(a) for c in val_corpora for a in c}
    assert train_keys.isdisjoint(val_keys), "train/val article overlap"
    return train_corpora, val_corpora


def split_train_val_disjoint(
    full_data: List[Dict[str, Any]],
    n_merge: int,
    n_val: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample train/val articles without replacement (same title OK)."""
    need = n_merge + n_val
    if need > len(full_data):
        sys.exit(
            f"[!] n_merge({n_merge}) + n_val({n_val}) = {need} exceeds "
            f"dataset size {len(full_data)}"
        )
    pool = rng.sample(full_data, need)
    train_items = pool[:n_merge]
    val_items = pool[n_merge:]
    train_keys = {_article_key(x) for x in train_items}
    val_keys = {_article_key(x) for x in val_items}
    assert train_keys.isdisjoint(val_keys), "train/val article overlap"
    return train_items, val_items


def _write_cpt_task_manifests(
    split_dir: Path,
    prefix: str,
    corpora: List[List[Dict[str, Any]]],
) -> List[str]:
    paths: List[str] = []
    for j, corpus in enumerate(corpora):
        task_path = split_dir / f"{prefix}_cpt_task{j}.json"
        task_manifest = {
            "role": f"cpt_{prefix}_task",
            "task_idx": j,
            "n_articles": len(corpus),
            "titles": [t["title"] for t in corpus],
            "articles": corpus,
        }
        task_path.write_text(
            json.dumps(task_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.append(str(task_path))
    return paths


def save_split_manifest(
    out_dir: Path,
    seq_idx: int,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    *,
    train_cpt_corpora: Optional[List[List[Dict[str, Any]]]] = None,
    val_cpt_corpora: Optional[List[List[Dict[str, Any]]]] = None,
) -> None:
    split_dir = out_dir / "splits" / f"seq{seq_idx}"
    split_dir.mkdir(parents=True, exist_ok=True)
    n_per_task = (
        len(train_cpt_corpora[0])
        if train_cpt_corpora
        else 1
    )
    if train_cpt_corpora is not None:
        train_manifest = {
            "role": "cpt_merge_train_tasks",
            "n_merge": len(train_cpt_corpora),
            "inner_sft_articles_per_task": n_per_task,
            "note": "Each merge step: fresh SE on N articles → inner TTT → merge",
        }
    else:
        train_manifest = {
            "role": "merge_train",
            "n_merge": len(train_items),
            "titles": [t["title"] for t in train_items],
            "articles": train_items,
        }
    (split_dir / "train.json").write_text(
        json.dumps(train_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if val_cpt_corpora is not None:
        val_manifest = {
            "role": "cpt_val_tasks",
            "n_val": len(val_cpt_corpora),
            "inner_sft_articles_per_task": (
                len(val_cpt_corpora[0]) if val_cpt_corpora else 0
            ),
            "note": (
                "Each task: fresh SE on N articles → inner TTT → overall acc "
                "on that task's questions"
            ),
        }
    else:
        val_manifest = {
            "role": "eval_only",
            "n_val": len(val_items),
            "titles": [t["title"] for t in val_items],
            "articles": val_items,
        }
    (split_dir / "val.json").write_text(
        json.dumps(val_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index: Dict[str, Any] = {
        "seq_idx": seq_idx,
        "train_path": str(split_dir / "train.json"),
        "val_path": str(split_dir / "val.json"),
    }
    if train_cpt_corpora is not None:
        index["train_cpt_task_paths"] = _write_cpt_task_manifests(
            split_dir, "train", train_cpt_corpora
        )
    if val_cpt_corpora is not None:
        index["val_cpt_task_paths"] = _write_cpt_task_manifests(
            split_dir, "val", val_cpt_corpora
        )
    (split_dir / "index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )


def run_one_sequence(
    seq_idx: int,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    train_cpt_corpora: Optional[List[List[Dict[str, Any]]]] = None,
    val_cpt_corpora: Optional[List[List[Dict[str, Any]]]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    cpt_mode = (
        _cpt_inner_mode(args)
        and train_cpt_corpora is not None
        and val_cpt_corpora is not None
    )
    K = len(train_cpt_corpora) if cpt_mode else len(train_items)
    M = len(val_cpt_corpora) if cpt_mode else len(val_items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(R)]
    out_dir = Path(args.output_dir)

    current_model = args.model

    # ---- Row 0: base model, eval (no merge) ----------------------------------
    if cpt_mode:
        n_art = len(val_cpt_corpora[0]) if val_cpt_corpora else 0
        eval_desc = (
            f"CPT val: {M} tasks × {n_art} articles (fresh SE + inner TTT each)"
        )
    else:
        eval_desc = f"val eval ({M} passages)"
    _banner(f"[Seq {seq_idx}] Row 0 — base model, {eval_desc}")
    _eval_all_val(
        current_model,
        val_items,
        args,
        f"seq{seq_idx}_row0",
        mat_vals[0],
        val_cpt_corpora=val_cpt_corpora,
    )
    print(
        "  row 0 mean val acc: "
        f"{_stats.mean([mat_vals[0][j][-1] for j in range(M)]):.3f}"
    )

    # ---- Merge train stream; after each merge, eval all val --------------------
    for k in range(K):
        max_model_len = args.max_tokens + 2048
        if cpt_mode:
            train_corpus = train_cpt_corpora[k]
            _banner(
                f"[Seq {seq_idx}] Merge step {k}/{K - 1} — "
                f"CPT train task {k} ({len(train_corpus)} articles)"
            )
            sess = _ServerSession(
                current_model,
                args,
                f"seq{seq_idx}_merge{k}",
                max_model_len,
                cpt_corpus=train_corpus,
                keep_adapter_dir=True,
            )
            sess.merge_train_corpus(train_corpus, f"merge_train{k}")
        else:
            train_item = train_items[k]
            _banner(
                f"[Seq {seq_idx}] Merge step {k}/{K - 1} — "
                f"train only: {train_item['title']}"
            )
            sess = _ServerSession(
                current_model,
                args,
                f"seq{seq_idx}_merge{k}",
                max_model_len,
                keep_adapter_dir=True,
            )
            sess.merge_train_task(train_item, f"merge_train{k}")
        sess.close()

        current_model = _merge_after_ttt(current_model, seq_idx, k, args)

        if cpt_mode:
            n_art = len(val_cpt_corpora[0]) if val_cpt_corpora else 0
            row_desc = (
                f"CPT val: {M} tasks × {n_art} articles (fresh SE + inner TTT each)"
            )
        else:
            row_desc = f"val eval ({M} passages, disjoint from merge train)"
        _banner(f"[Seq {seq_idx}] Row {k + 1} — after {k + 1} merge(s), {row_desc}")
        _eval_all_val(
            current_model,
            val_items,
            args,
            f"seq{seq_idx}_row{k + 1}",
            mat_vals[k + 1],
            val_cpt_corpora=val_cpt_corpora,
        )
        print(
            "  row "
            f"{k + 1} mean val acc: "
            f"{_stats.mean([mat_vals[k + 1][j][-1] for j in range(M)]):.3f}"
        )

    _cleanup_prev_merge_dir(current_model, str(out_dir), "end of sequence")

    _cleanup_inner_tmp(args.zmq_port, "end of sequence")

    mean_mat: List[List[float]] = [[0.0] * M for _ in range(R)]
    std_mat: List[List[float]] = [[0.0] * M for _ in range(R)]
    for r in range(R):
        for i in range(M):
            vals = mat_vals[r][i]
            if vals:
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    print("mean matrix (rows=merge stage, cols=val tasks):\n", json.dumps(mean_mat, indent=2))
    return mean_mat, std_mat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Self-edit generation forgetting: merge on train stream, "
            "eval on disjoint val stream"
        )
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_sequences", type=int, default=8)
    p.add_argument("--n_merge", type=int, default=8, help="Passages in continual merge stream")
    p.add_argument(
        "--n_val",
        type=int,
        default=8,
        help=(
            "Held-out val tasks: single mode = n_val passages; CPT mode = n_val "
            "disjoint N-article corpora (N = inner_sft_articles)"
        ),
    )
    p.add_argument(
        "--n_datapoints",
        type=int,
        default=None,
        help="Deprecated: if set without n_merge/n_val, splits evenly into merge+val",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
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
    p.add_argument(
        "--inner_sft_articles",
        type=int,
        default=1,
        help=(
            "Inner SFT corpus size per val task: 1 = single passage; "
            "200 = CPT-style N-article val task (n_val such tasks per sequence)"
        ),
    )
    p.add_argument(
        "--k_completions",
        type=int,
        default=None,
        help=(
            "Self-edits per article for inner SFT (CPT.sh: 5; single-passage default: 1). "
            "Uses dataset completions when present, else vLLM n=k."
        ),
    )
    p.add_argument(
        "--split_newlines",
        action="store_true",
        default=None,
        help="Override inner split_newlines (default: True if inner_sft_articles=1, else False)",
    )
    p.add_argument(
        "--no_split_newlines",
        dest="split_newlines",
        action="store_false",
        help="Force split_newlines=False (CPT.sh SPLIT_NEWLINES=0)",
    )
    p.add_argument(
        "--output_dir",
        default="general-knowledge/results/continual_self_edit_gen_forgetting",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_split_sizes(args: argparse.Namespace) -> Tuple[int, int]:
    if args.n_datapoints is not None:
        if args.n_merge != 8 or args.n_val != 8:
            return args.n_merge, args.n_val
        half = args.n_datapoints // 2
        return half, args.n_datapoints - half
    return args.n_merge, args.n_val


def main() -> None:
    args = parse_args()
    args.inner_split_newlines = _resolve_inner_split_newlines(args)
    args.k_completions = _resolve_k_completions(args)
    n_merge, n_val = _resolve_split_sizes(args)
    _banner(
        "[Args] "
        + json.dumps(
            {
                **vars(args),
                "n_merge": n_merge,
                "n_val": n_val,
                "inner_split_newlines": args.inner_split_newlines,
                "k_completions": args.k_completions,
            },
            indent=2,
        )
    )

    rng = random.Random(args.seed)
    gpus = args.gpus.split(",")
    if len(gpus) < 2:
        sys.exit("[!] --gpus must list at least two IDs (vLLM,inner)")
    args.vllm_gpus, args.inner_gpu = gpus[0], gpus[1]

    full_data: List[Dict[str, Any]] = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_means: List[List[List[float]]] = []
    seq_stds: List[List[List[float]]] = []

    matrix_n_val = n_val

    for seq_idx in range(args.n_sequences):
        seq_seed = args.seed + seq_idx * 10007
        seq_rng = random.Random(seq_seed)
        if _cpt_inner_mode(args):
            train_cpt_corpora, val_cpt_corpora = split_train_val_cpt_disjoint(
                full_data,
                n_merge,
                n_val,
                args.inner_sft_articles,
                seq_rng,
            )
            train_items = []
            val_items = []
        else:
            train_items, val_items = split_train_val_disjoint(
                full_data, n_merge, n_val, seq_rng
            )
            train_cpt_corpora = None
            val_cpt_corpora = None
        save_split_manifest(
            out_dir,
            seq_idx,
            train_items,
            val_items,
            train_cpt_corpora=train_cpt_corpora,
            val_cpt_corpora=val_cpt_corpora,
        )
        mean_mat, std_mat = run_one_sequence(
            seq_idx,
            train_items,
            val_items,
            args,
            train_cpt_corpora=train_cpt_corpora,
            val_cpt_corpora=val_cpt_corpora,
        )
        seq_means.append(mean_mat)
        seq_stds.append(std_mat)

    R = n_merge + 1
    M = matrix_n_val
    agg_mean = [[0.0] * M for _ in range(R)]
    agg_std = [[0.0] * M for _ in range(R)]
    for r in range(R):
        for i in range(M):
            vals = [seq_means[s][r][i] for s in range(args.n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    cpt_mode = _cpt_inner_mode(args)
    inner_mode = "cpt_aggregate" if cpt_mode else "single_passage"
    if cpt_mode:
        eval_desc = (
            f"CPT: n_merge={n_merge} + n_val={n_val} tasks, each "
            f"inner_sft_articles={args.inner_sft_articles}; train/val disjoint; "
            f"each step/task fresh SE + inner TTT; val = overall acc per task."
        )
    else:
        eval_desc = (
            f"Per-val-passage adapter accuracy (n_val={n_val}), disjoint from merge train."
        )
    summary = {
        "experiment": "continual_self_edit_gen_forgetting_disjoint_val",
        "mean_over_sequences": agg_mean,
        "std_over_sequences": agg_std,
        "n_sequences": args.n_sequences,
        "n_merge": n_merge,
        "n_val": n_val,
        "matrix_n_val_cols": matrix_n_val,
        "inner_sft_articles": args.inner_sft_articles,
        "inner_sft_mode": inner_mode,
        "k_completions": args.k_completions,
        "eval_mode": "cpt_val_tasks" if cpt_mode else "per_passage",
        "split_newlines": args.inner_split_newlines,
        "dataset": args.dataset,
        "base_model": args.model,
        "metric": "val_adapter_accuracy_after_fresh_self_edit_and_ttt",
        "description": (
            "Rows: row 0 = base; row r = after r train merges. "
            + eval_desc
            + f" Inner SFT: {inner_mode}."
        ),
    }
    summary_path = out_dir / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished. Summary → {summary_path}")


if __name__ == "__main__":
    main()
