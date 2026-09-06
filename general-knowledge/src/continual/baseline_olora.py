# general-knowledge/src/continual/baseline_olora.py
"""
Unified O-LoRA continual-learning driver for Baseline 2 (standard) and Baseline 3 (adaptive).

Baseline 2 (--olora_mode standard): fixed lambda, no task reuse.
Baseline 3 (--olora_mode adaptive): per-U_hist adaptive lambda + optional LoRA reuse.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _stats
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..lora_config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK
from ..utils import (
    TASK_SIMILARITY_PROMPT_VERSION,
    score_task_similarity_with_gpt4,
)
from .baseline_olora_checkpoint import (
    CHECKPOINT_FILENAME,
    build_checkpoint_summary,
    cleanup_sequence_artifacts,
    filter_steps_log,
    load_checkpoint,
    save_checkpoint,
    sequence_record_from_run,
    validate_checkpoint_for_resume,
)
from .merge_train_retention import (
    aggregate_lower_tri_over_sequences,
    build_agg_questions_and_spans,
    eval_merge_train_row0,
    finalize_lower_tri_matrix,
    init_lower_tri_mat_vals,
    inner_summary_from_merge_train,
    record_merge_train_step,
)
from .continual_self_edit_gen_forgetting import (
    _completions_for_article,
    _merge_after_ttt,
    _questions_for_item,
    _resolve_inner_split_newlines,
    _resolve_k_completions,
    _resolve_add_context,
    _single_passage_train_sequences,
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
from .olora_utils import (
    UHistStore,
    append_task_a,
    extract_a_from_adapter_dir,
    init_u_hist_from_se,
)
from .task_bank import (
    TaskBank,
    build_adaptive_decision_from_anchor_lambdas,
    compute_per_uhist_similarities_embedding,
    lambda_from_similarity,
    u_hist_anchor_entries,
)


@dataclass
class LambdaDecision:
    lambda_t: float
    lambda_weights: List[float]
    per_task_lambdas: Dict[str, float]
    s_max: float = 0.0
    reuse: bool = False
    matched_task_id: Optional[str] = None
    init_adapter_path: Optional[str] = None
    anchor_task_ids: List[str] | None = None
    gpt4_prompt: str = ""
    gpt4_raw: str = ""
    similarity_scores: Dict[str, float] | None = None
    fallback: bool = False

    def __post_init__(self) -> None:
        if self.anchor_task_ids is None:
            self.anchor_task_ids = []


def is_adaptive_mode(args: argparse.Namespace) -> bool:
    return args.olora_mode == "adaptive"


def resolve_experiment_name(args: argparse.Namespace) -> str:
    if args.olora_mode == "standard":
        return "baseline2_standard_olora"
    if args.lambda_source == "gpt4":
        return "baseline3_gpt4_adaptive_olora"
    return "baseline3_embedding_adaptive_olora"


def default_lambda_t(args: argparse.Namespace) -> float:
    if is_adaptive_mode(args):
        return float(args.fixed_lambda_t) if args.fixed_lambda_t is not None else 1.0
    return float(args.lambda_fixed)


def fixed_lambda_decision(args: argparse.Namespace) -> LambdaDecision:
    lam = default_lambda_t(args)
    return LambdaDecision(
        lambda_t=lam,
        lambda_weights=[lam],
        per_task_lambdas={},
        s_max=max(0.0, 1.0 - lam),
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
    instruct_model: bool = False,
    thinking_mode: bool = False,
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
    if instruct_model:
        cmd.append("--instruct_model")
    if thinking_mode:
        cmd.append("--thinking_mode")
    _banner(f"[Inner O-LoRA] GPU {gpu}, ZMQ :{zmq_port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_dir / f"inner_{tag}.log"
    return subprocess.Popen(
        cmd, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT
    )


def _i_t_text_from_completions(raw_completions: List[str]) -> str:
    if not raw_completions:
        return ""
    return "\n".join(c.strip() for c in raw_completions if c.strip())


def _compute_adaptive_decision(
    *,
    merge_step: int,
    item: Dict[str, Any],
    i_t_text: str,
    task_bank: TaskBank,
    u_hist_dir: Path,
    args: argparse.Namespace,
) -> LambdaDecision:
    fixed = args.fixed_lambda_t
    default_lam = float(fixed) if fixed is not None else 1.0

    cold_start = merge_step == 0 or task_bank.is_empty()
    if cold_start:
        return LambdaDecision(
            lambda_t=default_lam,
            lambda_weights=[default_lam],
            per_task_lambdas={},
            s_max=max(0.0, 1.0 - default_lam),
        )

    anchors = u_hist_anchor_entries(task_bank, u_hist_dir)
    anchor_ids = [a.task_id for a in anchors]
    if not anchors:
        return LambdaDecision(
            lambda_t=default_lam,
            lambda_weights=[default_lam],
            per_task_lambdas={},
            s_max=max(0.0, 1.0 - default_lam),
        )

    u_se_lambda = float(fixed) if fixed is not None else 1.0
    gpt4_prompt = ""
    gpt4_raw = ""
    similarity_scores: Optional[Dict[str, float]] = None
    fallback = False

    if fixed is not None:
        per_task = {a.task_id: float(fixed) for a in anchors}
    elif args.lambda_source == "embedding":
        similarity_scores = compute_per_uhist_similarities_embedding(
            i_t_text, anchors, task_bank.encoder
        )
        per_task = {
            tid: lambda_from_similarity(sim)
            for tid, sim in similarity_scores.items()
        }
    else:
        candidate_payload = [
            {"task_id": a.task_id, "I_t_text": a.I_t_text} for a in anchors
        ]
        sim_result = score_task_similarity_with_gpt4(
            title=item["title"],
            context=item["context"],
            current_implications=i_t_text,
            candidates=candidate_payload,
        )
        gpt4_prompt = sim_result.get("prompt", "")
        gpt4_raw = sim_result.get("raw", "")
        fallback = bool(sim_result.get("fallback"))
        if fallback:
            return LambdaDecision(
                lambda_t=1.0,
                lambda_weights=[1.0] + [1.0] * len(anchors),
                per_task_lambdas={a.task_id: 1.0 for a in anchors},
                s_max=0.0,
                anchor_task_ids=anchor_ids,
                gpt4_prompt=gpt4_prompt,
                gpt4_raw=gpt4_raw,
                similarity_scores={
                    s["task_id"]: float(s["similarity"])
                    for s in (sim_result.get("scores") or [])
                    if s.get("task_id")
                }
                or None,
                fallback=True,
            )
        similarity_scores = {}
        per_task = {}
        for score_entry in sim_result.get("scores") or []:
            tid = score_entry.get("task_id")
            if tid:
                sim = float(score_entry.get("similarity", 0.0))
                similarity_scores[tid] = sim
                per_task[tid] = lambda_from_similarity(sim)
        for anchor in anchors:
            per_task.setdefault(anchor.task_id, 1.0)
            if similarity_scores is not None:
                similarity_scores.setdefault(anchor.task_id, 0.0)

    (
        lambda_weights,
        min_lambda,
        s_max,
        reuse,
        matched_id,
        init_path,
    ) = build_adaptive_decision_from_anchor_lambdas(
        anchor_lambdas=per_task,
        anchors=anchors,
        task_bank=task_bank,
        tau=args.tau,
        u_se_lambda=u_se_lambda,
    )

    return LambdaDecision(
        lambda_t=min_lambda,
        lambda_weights=lambda_weights,
        per_task_lambdas=per_task,
        s_max=s_max,
        reuse=reuse,
        matched_task_id=matched_id,
        init_adapter_path=init_path,
        anchor_task_ids=anchor_ids,
        gpt4_prompt=gpt4_prompt,
        gpt4_raw=gpt4_raw,
        similarity_scores=similarity_scores,
        fallback=fallback,
    )


def _log_step(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _similarity_log_fields(
    decision: LambdaDecision,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "similarity_source": args.lambda_source,
        "similarity_prompt_version": TASK_SIMILARITY_PROMPT_VERSION,
        "gpt4_prompt": decision.gpt4_prompt,
        "gpt4_raw": decision.gpt4_raw,
        "similarity_scores": decision.similarity_scores,
        "per_task_lambdas": decision.per_task_lambdas,
        "lambda_weights": decision.lambda_weights,
        "s_max": decision.s_max,
        "lambda_t": decision.lambda_t,
        "tau": args.tau,
        "reuse": decision.reuse,
        "matched_task_id": decision.matched_task_id,
        "anchor_task_ids": decision.anchor_task_ids,
        "fallback": decision.fallback,
        "fixed_lambda_t": args.fixed_lambda_t,
    }


def _u_hist_store_dir(args: argparse.Namespace, seq_idx: int) -> Path:
    return Path(args.output_dir) / "U_hist" / f"seq{seq_idx}"


def _task_bank_dir(args: argparse.Namespace, seq_idx: int) -> Path:
    return Path(args.output_dir) / "task_bank" / f"seq{seq_idx}"


class OLoRAServerSession:
    """vLLM + O-LoRA inner server; supports fixed or per-request adaptive lambdas."""

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
        self._olora: Dict[str, Any] = {
            "lambda_t": default_lambda_t(args),
            "lambda_weights": [default_lambda_t(args)],
            "init_adapter_path": None,
            "reuse_mode": False,
        }
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
            instruct_model=bool(getattr(args, "instruct_model", False)),
            thinking_mode=bool(getattr(args, "thinking_mode", False)),
        )
        self.ctx, self.sock = _connect_zmq(args.zmq_port)

    def set_olora_params(
        self,
        *,
        lambda_t: float,
        lambda_weights: List[float],
        init_adapter_path: Optional[str] = None,
        reuse_mode: bool = False,
    ) -> None:
        self._olora = {
            "lambda_t": lambda_t,
            "lambda_weights": lambda_weights,
            "init_adapter_path": init_adapter_path,
            "reuse_mode": reuse_mode,
        }

    def _send_round(
        self,
        train_seqs: List[str],
        questions: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        olora_payload: Dict[str, Any] = {
            "enabled": True,
            "lambda_t": self._olora["lambda_t"],
            "lambda_weights": self._olora["lambda_weights"],
            "gamma": self.args.gamma,
            "U_hist_dir": str(self.u_hist_dir),
        }
        if self._olora.get("init_adapter_path"):
            olora_payload["init_adapter_path"] = self._olora["init_adapter_path"]
            olora_payload["reuse_mode"] = self._olora.get("reuse_mode", False)
        self.sock.send_json(
            {
                "train_sequences": train_seqs,
                "eval_questions": questions,
                "lora_rank": self.args.lora_rank,
                "lora_alpha": self.args.lora_alpha,
                "lora_dropout": self.args.lora_dropout,
                "finetune_epochs": self.args.finetune_epochs,
                "finetune_lr": self.args.finetune_lr,
                "batch_size": self.args.batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "end_mask_substring": self.args.end_mask_substring,
                "olora": olora_payload,
            }
        )
        rep = self.sock.recv_json()
        if "error" in rep:
            raise RuntimeError(f"TTT_server_olora error: {rep['error']}")
        return rep

    def step1_completions(self, item: Dict[str, Any]) -> List[str]:
        return _completions_for_article(
            self.vllm_api,
            self.model_path,
            item,
            self.args,
            self.args.k_completions,
        )

    def train_sequences_from_completions(
        self,
        item: Dict[str, Any],
        raw_completions: List[str],
    ) -> List[str]:
        return _single_passage_train_sequences(
            item,
            raw_completions,
            split_newlines=self.args.inner_split_newlines,
            add_context=self.args.add_context,
        )

    def _lambda_label(self) -> str:
        if is_adaptive_mode(self.args):
            return f"adaptive O-LoRA λ={self._olora['lambda_t']:.3f}"
        return "O-LoRA"

    def run_inner(
        self,
        item: Dict[str, Any],
        train_sequences: List[str],
        label: str,
    ) -> float:
        rep = self._send_round(train_sequences, _questions_for_item(item))
        correct = rep["adapter_correct"]
        acc = sum(correct) / len(correct) if correct else 0.0
        title_short = item["title"][:50] + ("…" if len(item["title"]) > 50 else "")
        print(f"      [{label}] ({self._lambda_label()}) {title_short} acc={acc:.3f}")
        return acc

    def merge_train_with_retention(
        self,
        item: Dict[str, Any],
        train_sequences: List[str],
        step_k: int,
        agg_questions: List[Dict[str, str]],
        q_spans: List[Tuple[int, int]],
        inner_mat_vals: List[List[List[float]]],
        label: str,
    ) -> None:
        record_merge_train_step(
            self.sock,
            train_sequences,
            agg_questions,
            q_spans,
            step_k,
            lambda sock, train_seqs, questions: self._send_round(
                train_seqs, questions
            ),
            inner_mat_vals,
        )
        title_short = item["title"][:50] + ("…" if len(item["title"]) > 50 else "")
        print(f"      [{label}] ({self._lambda_label()} merge) {title_short}")

    def eval_merge_train_base_row(
        self,
        train_items: List[Dict[str, Any]],
        inner_mat_vals: List[List[List[float]]],
        tag: str,
    ) -> None:
        eval_merge_train_row0(
            self.sock,
            train_items,
            _questions_for_item,
            lambda sock, train_seqs, questions: self._send_round(
                train_seqs, questions
            ),
            inner_mat_vals,
            log_prefix=f"{tag} merge-train base",
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


def _resolve_lambda_for_val(
    args: argparse.Namespace,
    *,
    merge_step: int,
    item: Dict[str, Any],
    raw_completions: List[str],
    task_bank: Optional[TaskBank],
    u_hist_dir: Path,
) -> LambdaDecision:
    if not is_adaptive_mode(args):
        return fixed_lambda_decision(args)
    assert task_bank is not None
    i_t_text = _i_t_text_from_completions(raw_completions)
    return _compute_adaptive_decision(
        merge_step=merge_step,
        item=item,
        i_t_text=i_t_text,
        task_bank=task_bank,
        u_hist_dir=u_hist_dir,
        args=args,
    )


def _eval_all_val(
    model_path: str,
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
    tag: str,
    mat_row: List[List[float]],
    u_hist_dir: Path,
    task_bank: Optional[TaskBank],
    steps_log: Optional[Path],
    seq_idx: int,
    checkpoint_row: int,
) -> None:
    max_model_len = args.max_tokens + 2048
    sess = OLoRAServerSession(
        model_path,
        args,
        tag,
        max_model_len,
        u_hist_dir,
    )
    for j, v_item in enumerate(val_items):
        raw = sess.step1_completions(v_item)
        decision = _resolve_lambda_for_val(
            args,
            merge_step=checkpoint_row,
            item=v_item,
            raw_completions=raw,
            task_bank=task_bank,
            u_hist_dir=u_hist_dir,
        )
        sess.set_olora_params(
            lambda_t=decision.lambda_t,
            lambda_weights=decision.lambda_weights,
            init_adapter_path=decision.init_adapter_path,
            reuse_mode=decision.reuse,
        )
        train_sequences = sess.train_sequences_from_completions(v_item, raw)
        acc = sess.run_inner(v_item, train_sequences, f"{tag}_val{j}")
        mat_row[j].append(acc)
        if steps_log is not None and is_adaptive_mode(args):
            _log_step(
                steps_log,
                {
                    "seq_idx": seq_idx,
                    "phase": "val_eval",
                    "checkpoint_row": checkpoint_row,
                    "val_idx": j,
                    "task_id": f"val_s{seq_idx}_j{j}",
                    **_similarity_log_fields(decision, args),
                },
            )
    sess.close()


def run_one_sequence(
    seq_idx: int,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[int],
    List[List[float]],
    List[List[float]],
    List[float],
    int,
]:
    adaptive = is_adaptive_mode(args)
    K = len(train_items)
    M = len(val_items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(M)] for _ in range(R)]
    inner_mat_vals = init_lower_tri_mat_vals(K)
    agg_questions, q_spans = build_agg_questions_and_spans(
        train_items, _questions_for_item
    )
    u_hist_sizes: List[int] = []
    lambda_per_step: List[float] = []
    reuse_count = 0

    u_hist_store = UHistStore(_u_hist_store_dir(args, seq_idx), args.u_se_path)
    u_hist_store.init()
    u_hist_dir = u_hist_store.seq_dir
    u_hist = init_u_hist_from_se(args.u_se_path)

    task_bank: Optional[TaskBank] = None
    steps_log: Optional[Path] = None
    if adaptive:
        task_bank = TaskBank(
            _task_bank_dir(args, seq_idx),
            seq_idx,
            embed_model=args.embed_model,
        )
        steps_log = Path(args.output_dir) / "adaptive_steps.jsonl"

    current_model = args.model
    row0_lam = default_lambda_t(args)
    mode_label = "adaptive O-LoRA" if adaptive else "O-LoRA"
    extra = (
        f"τ={args.tau}, source={args.lambda_source}"
        if adaptive
        else f"λ={args.lambda_fixed}, γ={args.gamma}"
    )

    _banner(
        f"[Seq {seq_idx}] Row 0 — iter2 + {mode_label} eval "
        f"(|U_hist|={len(u_hist)}, {extra})"
    )
    max_model_len = args.max_tokens + 2048
    base_sess = OLoRAServerSession(
        current_model,
        args,
        f"seq{seq_idx}_merge_train_row0",
        max_model_len,
        u_hist_dir,
    )
    base_sess.set_olora_params(
        lambda_t=row0_lam,
        lambda_weights=[row0_lam],
    )
    base_sess.eval_merge_train_base_row(
        train_items, inner_mat_vals, f"seq{seq_idx}_merge_train_row0"
    )
    base_sess.close()

    _eval_all_val(
        current_model,
        val_items,
        args,
        f"seq{seq_idx}_row0",
        mat_vals[0],
        u_hist_dir,
        task_bank,
        steps_log,
        seq_idx,
        checkpoint_row=0,
    )
    u_hist_sizes.append(len(u_hist))
    print(
        "  row 0 mean val acc: "
        f"{_stats.mean([mat_vals[0][j][-1] for j in range(M)]):.3f}"
    )

    for k in range(K):
        train_item = train_items[k]
        bank_info = (
            f"  |bank|={len(task_bank.list_entries())}" if task_bank else ""
        )
        _banner(
            f"[Seq {seq_idx}] Merge {k}/{K - 1} — "
            f"{train_item['title'][:60]}  |U_hist|={len(u_hist)}{bank_info}"
        )
        max_model_len = args.max_tokens + 2048
        sess = OLoRAServerSession(
            current_model,
            args,
            f"seq{seq_idx}_merge{k}",
            max_model_len,
            u_hist_dir,
            keep_adapter_dir=True,
        )

        raw = sess.step1_completions(train_item)
        if adaptive:
            assert task_bank is not None
            i_t_text = _i_t_text_from_completions(raw)
            decision = _compute_adaptive_decision(
                merge_step=k,
                item=train_item,
                i_t_text=i_t_text,
                task_bank=task_bank,
                u_hist_dir=u_hist_dir,
                args=args,
            )
        else:
            decision = fixed_lambda_decision(args)

        sess.set_olora_params(
            lambda_t=decision.lambda_t,
            lambda_weights=decision.lambda_weights,
            init_adapter_path=decision.init_adapter_path,
            reuse_mode=decision.reuse,
        )
        train_sequences = sess.train_sequences_from_completions(train_item, raw)
        sess.merge_train_with_retention(
            train_item,
            train_sequences,
            k,
            agg_questions[: q_spans[k][1]],
            q_spans,
            inner_mat_vals,
            f"merge_train{k}",
        )
        sess.close()

        lambda_per_step.append(decision.lambda_t)
        if decision.reuse:
            reuse_count += 1

        if steps_log is not None and adaptive:
            _log_step(
                steps_log,
                {
                    "seq_idx": seq_idx,
                    "phase": "merge_train",
                    "merge_step": k,
                    "task_id": f"merge_s{seq_idx}_k{k}",
                    **_similarity_log_fields(decision, args),
                },
            )

        adapter_path = _inner_tmp_dir(args.zmq_port) / "final_adapter"
        if not adapter_path.exists():
            raise RuntimeError(
                f"Missing final_adapter after merge {k} (expected {adapter_path})"
            )

        a_matrices_path: Optional[Path] = None
        if adaptive and decision.reuse:
            print(f"      reused {decision.matched_task_id} — skip U_hist append")
        else:
            a_k = extract_a_from_adapter_dir(adapter_path)
            u_hist = append_task_a(u_hist, a_k)
            n_tasks = u_hist_store.append_merge_task(a_k)
            a_matrices_path = u_hist_dir / f"task_{n_tasks - 1:03d}.pt"
            print(f"      saved task_{k + 1:03d}.pt → |U_hist|={n_tasks}")

        if adaptive and task_bank is not None:
            task_bank.append_entry(
                merge_step=k,
                item=train_item,
                i_t_text=_i_t_text_from_completions(raw),
                adapter_src=adapter_path,
                a_matrices_path=a_matrices_path,
                lambda_t=decision.lambda_t,
                s_max=decision.s_max,
                similarity_source=args.lambda_source,
                reused_from=decision.matched_task_id if decision.reuse else None,
            )

        current_model = _merge_after_ttt(current_model, seq_idx, k, args)

        row_banner = (
            f"[Seq {seq_idx}] Row {k + 1} — after {k + 1} merge(s), "
            f"|U_hist|={len(u_hist)}"
        )
        if adaptive:
            row_banner += (
                f"  reuse={decision.reuse} λ={decision.lambda_t:.3f}"
            )
        _banner(row_banner)

        _eval_all_val(
            current_model,
            val_items,
            args,
            f"seq{seq_idx}_row{k + 1}",
            mat_vals[k + 1],
            u_hist_dir,
            task_bank,
            steps_log,
            seq_idx,
            checkpoint_row=k + 1,
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

    print(f"mean matrix ({mode_label}):\n", json.dumps(mean_mat, indent=2))
    inner_mean, inner_std = finalize_lower_tri_matrix(inner_mat_vals, K)
    print(
        f"merge-train retention matrix ({mode_label}, lower-tri):\n",
        json.dumps(inner_mean, indent=2),
    )
    return mean_mat, std_mat, u_hist_sizes, inner_mean, inner_std, lambda_per_step, reuse_count


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
    return train_manifest["articles"], val_manifest["articles"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline 2/3 unified O-LoRA continual learning driver"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="models/iter2")
    p.add_argument(
        "--instruct_model",
        action="store_true",
        help="Set this flag if you are using a Qwen instruct model",
    )
    p.add_argument(
        "--thinking_mode",
        action="store_true",
        help="Set this flag to enable thinking mode for Qwen3 models",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--u_se_path", required=True)
    p.add_argument("--splits_dir", default=None)
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
    p.add_argument(
        "--no_add_context",
        action="store_true",
        help="Inner TTT: self-edit implications only (no passage training row)",
    )
    p.add_argument(
        "--olora_mode",
        default="standard",
        choices=["standard", "adaptive"],
        help="standard = Baseline 2 fixed λ; adaptive = Baseline 3 adaptive λ + reuse",
    )
    p.add_argument(
        "--lambda_fixed",
        type=float,
        default=1.0,
        help="Baseline 2: fixed O-LoRA lambda for every step",
    )
    p.add_argument(
        "--lambda_source",
        default="gpt4",
        choices=["gpt4", "embedding"],
        help="Baseline 3 adaptive: similarity source for per-anchor lambdas",
    )
    p.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument(
        "--fixed_lambda_t",
        type=float,
        default=None,
        help="Adaptive mode: skip similarity API and use this lambda for all slots",
    )
    p.add_argument("--start_seq", type=int, default=0)
    p.add_argument("--checkpoint_path", default=None)
    return p.parse_args(argv)


def _build_final_summary(
    args: argparse.Namespace,
    experiment_name: str,
    sequences: List[Dict[str, Any]],
    n_merge: int,
    n_val: int,
    splits_dir: Optional[Path],
) -> Dict[str, Any]:
    seq_means = [s["val_forgetting_mean"] for s in sequences]
    seq_stds = [s["val_forgetting_std"] for s in sequences]
    inner_seq_means = [s["merge_train_retention_mean"] for s in sequences]
    inner_seq_stds = [s["merge_train_retention_std"] for s in sequences]
    all_lambda_steps = [s["lambda_per_merge_step"] for s in sequences]
    total_reuse = sum(s.get("reuse_count", 0) for s in sequences)

    R = n_merge + 1
    M = n_val
    agg_mean = [[0.0] * M for _ in range(R)]
    agg_std = [[0.0] * M for _ in range(R)]
    for r in range(R):
        for i in range(M):
            vals = [seq_means[s][r][i] for s in range(len(sequences))]
            agg_mean[r][i] = _stats.mean(vals) if vals else 0.0
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    inner_agg_mean, inner_agg_std = aggregate_lower_tri_over_sequences(
        inner_seq_means, inner_seq_stds, n_merge, len(sequences)
    )

    lambda_mean_per_merge_step: List[float] = []
    for k in range(n_merge):
        vals = [
            all_lambda_steps[s][k]
            for s in range(len(sequences))
            if k < len(all_lambda_steps[s])
        ]
        lambda_mean_per_merge_step.append(_stats.mean(vals) if vals else 1.0)

    reuse_rate = (
        total_reuse / (args.n_sequences * n_merge) if n_merge else 0.0
    )

    summary: Dict[str, Any] = {
        "experiment": experiment_name,
        "olora_mode": args.olora_mode,
        "gamma": args.gamma,
        "u_se_path": args.u_se_path,
        "splits_dir": str(splits_dir) if splits_dir else None,
        "reuse_rate": reuse_rate,
        "lambda_mean_per_merge_step": lambda_mean_per_merge_step,
        "mean_over_sequences": agg_mean,
        "std_over_sequences": agg_std,
        "n_sequences": args.n_sequences,
        "n_merge": n_merge,
        "n_val": n_val,
        "inner_sft_articles": args.inner_sft_articles,
        "k_completions": args.k_completions,
        "split_newlines": args.inner_split_newlines,
        "add_context": args.add_context,
        "dataset": args.dataset,
        "base_model": args.model,
    }

    if is_adaptive_mode(args):
        summary.update(
            {
                "lambda_source": args.lambda_source,
                "similarity_model": (
                    "gpt-4.1"
                    if args.lambda_source == "gpt4"
                    else args.embed_model
                ),
                "similarity_prompt_version": TASK_SIMILARITY_PROMPT_VERSION,
                "embed_model": args.embed_model,
                "tau": args.tau,
                "fixed_lambda_t": args.fixed_lambda_t,
                "metric": (
                    "val_adapter_accuracy_after_fresh_self_edit_and_adaptive_olora_ttt"
                ),
                "description": (
                    "Adaptive O-LoRA: per-U_hist lambda weights, min-lambda reuse when "
                    f"min(lambda)<tau; "
                    + (
                        f"fixed_lambda_t={args.fixed_lambda_t} (no similarity API)."
                        if args.fixed_lambda_t is not None
                        else f"lambda from {args.lambda_source}."
                    )
                ),
            }
        )
    else:
        summary.update(
            {
                "lambda_fixed": args.lambda_fixed,
                "metric": "val_adapter_accuracy_after_fresh_self_edit_and_olora_ttt",
                "description": (
                    "Standard O-LoRA: fixed lambda, no task LoRA reuse. "
                    "Same protocol as Baseline 1 single-passage forgetting matrix."
                ),
            }
        )

    summary.update(
        inner_summary_from_merge_train(
            inner_agg_mean,
            inner_agg_std,
            n_sequences=args.n_sequences,
            n_merge=n_merge,
        )
    )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.thinking_mode and not args.instruct_model:
        raise SystemExit("[!] --thinking_mode requires --instruct_model")
    args.inner_split_newlines = _resolve_inner_split_newlines(args)
    if args.k_completions is None:
        args.k_completions = _resolve_k_completions(args)
    args.add_context = _resolve_add_context(args)
    n_merge, n_val = args.n_merge, args.n_val
    experiment_name = resolve_experiment_name(args)

    _banner(
        f"[{experiment_name} Args] "
        + json.dumps(
            {**vars(args), "n_merge": n_merge, "n_val": n_val},
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
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else out_dir / CHECKPOINT_FILENAME
    )

    sequences: List[Dict[str, Any]] = []
    if args.start_seq > 0:
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_for_resume(checkpoint, args.start_seq, args)
        sequences = list(checkpoint.get("sequences", []))
        print(
            f"[Resume] loaded checkpoint with {len(sequences)} completed "
            f"sequence(s) → starting at seq {args.start_seq}"
        )
        if is_adaptive_mode(args):
            filter_steps_log(out_dir / "adaptive_steps.jsonl", args.start_seq - 1)
    elif checkpoint_path.exists():
        print(
            f"[Resume] --start_seq=0: ignoring existing checkpoint at {checkpoint_path}"
        )

    for seq_idx in range(args.start_seq, args.n_sequences):
        cleanup_sequence_artifacts(out_dir, seq_idx)
        if splits_dir is not None:
            train_items, val_items = load_split_from_dir(splits_dir, seq_idx)
            print(f"[Seq {seq_idx}] loaded split from {splits_dir}/seq{seq_idx}")
        else:
            seq_rng = __import__("random").Random(args.seed + seq_idx * 10007)
            train_items, val_items = split_train_val_disjoint(
                full_data, n_merge, n_val, seq_rng
            )
        save_split_manifest(out_dir, seq_idx, train_items, val_items)

        mean_mat, std_mat, _, inner_mean, inner_std, lambda_steps, reuse_count = (
            run_one_sequence(seq_idx, train_items, val_items, args)
        )
        if not lambda_steps and not is_adaptive_mode(args):
            lambda_steps = [float(args.lambda_fixed)] * n_merge

        sequences.append(
            sequence_record_from_run(
                seq_idx,
                mean_mat,
                std_mat,
                inner_mean,
                inner_std,
                lambda_steps,
                reuse_count,
            )
        )

        save_checkpoint(
            checkpoint_path,
            build_checkpoint_summary(
                experiment_name=experiment_name,
                args=args,
                sequences=sequences,
                n_sequences=args.n_sequences,
                n_merge=n_merge,
                n_val=n_val,
                status=(
                    "in_progress"
                    if len(sequences) < args.n_sequences
                    else "complete"
                ),
            ),
        )

    summary = _build_final_summary(
        args, experiment_name, sequences, n_merge, n_val, splits_dir
    )
    summary_path = out_dir / f"summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished {experiment_name}. Summary → {summary_path}")
    if is_adaptive_mode(args):
        print(f"Reuse rate: {summary['reuse_rate']:.3f}")


if __name__ == "__main__":
    main()
