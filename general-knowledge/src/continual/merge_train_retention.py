# general-knowledge/src/continual/merge_train_retention.py
"""Merge-train retention matrix (continual_self_edits protocol) for B1/B2 drivers."""
from __future__ import annotations

import statistics as _stats
from typing import Any, Callable, Dict, List, Tuple

SendRoundFn = Callable[[Any, List[str], List[Dict[str, str]]], Dict[str, Any]]


def build_agg_questions_and_spans(
    train_items: List[Dict[str, Any]],
    questions_for_item: Callable[[Dict[str, Any]], List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], List[Tuple[int, int]]]:
    """Cumulative eval questions and (start, end) spans per merge-train task."""
    agg_questions: List[Dict[str, str]] = []
    q_spans: List[Tuple[int, int]] = []
    cum = 0
    for item in train_items:
        n_q = len(item["questions"])
        q_spans.append((cum, cum + n_q))
        cum += n_q
        agg_questions.extend(questions_for_item(item))
    return agg_questions, q_spans


def per_task_accuracies(
    adapter_correct: List[bool],
    q_spans: List[Tuple[int, int]],
    up_to_task: int,
) -> List[float]:
    """Mean adapter accuracy for merge-train tasks 0..up_to_task (inclusive)."""
    accs: List[float] = []
    for i in range(up_to_task + 1):
        s, e = q_spans[i]
        chunk = adapter_correct[s:e]
        accs.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return accs


def init_lower_tri_mat_vals(K: int) -> List[List[List[float]]]:
    """(K+1) x K; row r>0 only cols i < r are used (row 0 uses all K)."""
    R = K + 1
    return [[[] for _ in range(K)] for _ in range(R)]


def eval_merge_train_row0(
    sock: Any,
    train_items: List[Dict[str, Any]],
    questions_for_item: Callable[[Dict[str, Any]], List[Dict[str, str]]],
    send_round: SendRoundFn,
    mat_vals: List[List[List[float]]],
    *,
    log_prefix: str = "merge-train base",
) -> None:
    """Row 0: no inner SFT; eval each merge-train task on the current generator."""
    for i, item in enumerate(train_items):
        rep = send_round(sock, [], questions_for_item(item))
        correct = rep["adapter_correct"]
        acc = sum(correct) / len(correct) if correct else 0.0
        mat_vals[0][i].append(acc)
        title = item["title"][:50] + ("…" if len(item["title"]) > 50 else "")
        print(f"      [{log_prefix}] d{i} {title} acc={acc:.3f}")


def record_merge_train_step(
    sock: Any,
    train_sequences: List[str],
    agg_questions: List[Dict[str, str]],
    q_spans: List[Tuple[int, int]],
    step_k: int,
    send_round: SendRoundFn,
    mat_vals: List[List[List[float]]],
) -> None:
    """
    After inner TTT on merge step k: eval adapter on train tasks d_0..d_k.
    Writes row k+1, cols 0..k (continual_self_edits lower-triangular fill).
    """
    rep = send_round(sock, train_sequences, agg_questions)
    if "error" in rep:
        raise RuntimeError(f"inner TTT merge-train retention failed: {rep['error']}")
    correct = rep["adapter_correct"]
    accs = per_task_accuracies(correct, q_spans, step_k)
    row = step_k + 1
    for i, acc in enumerate(accs):
        mat_vals[row][i].append(acc)
    print(
        f"      [merge-train retention] step {step_k} → "
        + " ".join(f"d{i}:{accs[i]:.3f}" for i in range(len(accs)))
    )


def finalize_lower_tri_matrix(
    mat_vals: List[List[List[float]]], K: int
) -> Tuple[List[List[float]], List[List[float]]]:
    R = K + 1
    mean_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    std_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = mat_vals[r][i]
            if vals:
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0
    return mean_mat, std_mat


def inner_summary_from_merge_train(
    mean_mat: List[List[float]],
    std_mat: List[List[float]],
    *,
    n_sequences: int,
    n_merge: int,
) -> Dict[str, Any]:
    return {
        "merge_train_retention_mean_over_sequences": mean_mat,
        "merge_train_retention_std_over_sequences": std_mat,
        "merge_train_retention_n_datapoints": n_merge,
        "merge_train_retention_metric": (
            "adapter_accuracy_on_merge_train_tasks_after_current_step_inner_ttt"
        ),
        "merge_train_retention_note": (
            "Lower-triangular (K+1)xK on merge-train stream. "
            "Row 0: no inner SFT. Row r>0: after merge step r-1 inner TTT on d_{r-1}, "
            "eval adapter on d_0..d_{r-1}. Same protocol as continual_self_edits."
        ),
    }


def aggregate_lower_tri_over_sequences(
    seq_means: List[List[List[float]]],
    seq_stds: List[List[List[float]]],
    n_merge: int,
    n_sequences: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    R = n_merge + 1
    K = n_merge
    agg_mean = [[0.0] * K for _ in range(R)]
    agg_std = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = [seq_means[s][r][i] for s in range(n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0
    return agg_mean, agg_std
