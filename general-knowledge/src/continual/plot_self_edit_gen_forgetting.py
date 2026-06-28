# general-knowledge/src/continual/plot_self_edit_gen_forgetting.py
"""
Plot continual-learning forgetting heatmaps:

1. Self-edit generation forgetting (continual_self_edit_gen_forgetting):
   full (n_merge+1) × n_val matrix on held-out val passages.

2. Inner-TTT knowledge retention (continual_self_edits):
   lower-triangular (K+1) × K matrix — row 0 = base on all tasks;
   row r>0 only cols d_0..d_{r-1} are defined.

Usage:
  # SE gen only
  python general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \\
    --results_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0

  # Both side-by-side
  python general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \\
    --results_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0 \\
    --inner_results_dir general-knowledge/results/continual_self_edits/run0 \\
    --combined_output general-knowledge/results/continual_self_edit_gen_forgetting/run0/forgetting_heatmaps.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_latest_summary(results_dir: Path) -> dict:
    candidates = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.startswith("summary_") and f.endswith(".json"):
                candidates.append(Path(root) / f)
    if not candidates:
        raise FileNotFoundError(f"No summary_*.json under {results_dir}")
    return json.loads(
        max(candidates, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8")
    )


def _annotate_cells(
    ax,
    data: np.ndarray,
    std_mat: Optional[np.ndarray],
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> None:
    R, M = data.shape
    for r in range(R):
        for i in range(M):
            if valid_mask is not None and not valid_mask[r, i]:
                continue
            val = data[r, i]
            if np.isnan(val):
                continue
            std_txt = ""
            if std_mat is not None and not np.isnan(std_mat[r, i]):
                std_txt = f"\n±{std_mat[r][i]:.2f}"
            ax.text(
                i,
                r,
                f"{val:.2f}{std_txt}",
                ha="center",
                va="center",
                color="white" if val < 0.55 else "black",
                fontsize=8,
            )


def lower_triangular_valid_mask(R: int, K: int) -> np.ndarray:
    """continual_self_edits: row 0 all K cols; row r>0 only cols i < r."""
    mask = np.zeros((R, K), dtype=bool)
    for r in range(R):
        if r == 0:
            mask[r, :] = True
        else:
            mask[r, :r] = True
    return mask


def mask_lower_triangular(
    mean_mat: List[List[float]],
    std_mat: Optional[List[List[float]]],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    data = np.array(mean_mat, dtype=float)
    R, K = data.shape
    valid = lower_triangular_valid_mask(R, K)
    masked = data.copy()
    masked[~valid] = np.nan
    std_masked = None
    if std_mat is not None:
        std_masked = np.array(std_mat, dtype=float)
        std_masked[~valid] = np.nan
    return masked, std_masked, valid


def plot_se_gen_matrix(
    mean_mat: List[List[float]],
    std_mat: Optional[List[List[float]]],
    out_path: Path,
    *,
    title: str,
    n_sequences: int,
    n_merge: int,
    n_val: int,
    eval_mode: str = "per_passage",
    inner_sft_articles: int = 1,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    data = np.array(mean_mat, dtype=float)
    R, M = data.shape

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(7, M * 0.95), max(5.5, R * 0.75)))
    else:
        fig = ax.figure

    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    if own_fig:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Val adapter accuracy (fresh SE + inner TTT)")

    _annotate_cells(ax, data, np.array(std_mat) if std_mat else None)

    if eval_mode in ("cpt_val_tasks", "cpt_overall"):
        ax.set_xlabel(
            f"CPT val task ({inner_sft_articles} articles, overall acc)"
        )
        xticks = (
            ["overall"]
            if eval_mode == "cpt_overall" and M == 1
            else [f"v{i}" for i in range(M)]
        )
    else:
        ax.set_xlabel("Held-out val passage (never merged)")
        xticks = [f"v{i}" for i in range(M)]
    ax.set_ylabel("Generator checkpoint")
    ax.set_title(title)
    ax.set_xticks(range(M))
    ax.set_yticks(range(R))
    ax.set_xticklabels(xticks)
    if R == 1:
        ylabels = ["base"]
    else:
        ylabels = ["base (0 merges)"] + [
            f"after {r} train merge(s)" for r in range(1, R)
        ]
    ax.set_yticklabels(ylabels)

    if own_fig:
        if eval_mode in ("cpt_val_tasks", "cpt_overall"):
            note = (
                f"Merge train: {n_merge}; CPT val: {n_val} tasks × "
                f"{inner_sft_articles} articles (fresh SE + inner TTT). "
                f"n_sequences={n_sequences}."
            )
        else:
            note = (
                f"Merge train: {n_merge}; val: {n_val} disjoint. "
                f"Full matrix. n_sequences={n_sequences}."
            )
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved SE-gen heatmap → {out_path}")
    return ax


def plot_inner_ttt_matrix(
    mean_mat: List[List[float]],
    std_mat: Optional[List[List[float]]],
    out_path: Path,
    *,
    title: str,
    n_sequences: int,
    n_datapoints: int,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    data, std_masked, valid = mask_lower_triangular(mean_mat, std_mat)
    R, K = data.shape

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(7, K * 0.95), max(5.5, R * 0.75)))
    else:
        fig = ax.figure

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    if own_fig:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Adapter accuracy (merged model + inner TTT)")

    _annotate_cells(ax, data, std_masked, valid_mask=valid)

    ax.set_xlabel("Train datapoint in merge stream")
    ax.set_ylabel("After continual merge step")
    ax.set_title(title)
    ax.set_xticks(range(K))
    ax.set_yticks(range(R))
    ax.set_xticklabels([f"d{i}" for i in range(K)])
    if R == 1:
        ylabels = ["base (no merge)"]
    else:
        ylabels = ["base (no merge)"] + [
            f"after merge step {r}" for r in range(1, R)
        ]
    ax.set_yticklabels(ylabels)

    if own_fig:
        note = (
            f"continual_self_edits lower-triangular matrix (K={n_datapoints}). "
            f"Row r>0: eval on d_0..d_{{r-1}} after r merges. "
            f"Gray = undefined. n_sequences={n_sequences}."
        )
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved inner-TTT heatmap → {out_path}")
    return ax


def plot_combined(
    se_data: dict,
    inner_data: dict,
    out_path: Path,
    *,
    se_title: str,
    inner_title: str,
) -> None:
    se_kw = _se_plot_kwargs(se_data)
    n_dp = inner_data.get("n_datapoints", len(inner_data["mean_over_sequences"][0]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(5.5, len(se_data["mean_over_sequences"]) * 0.75)),
        layout="constrained",
    )

    plot_se_gen_matrix(
        se_data["mean_over_sequences"],
        se_data.get("std_over_sequences"),
        out_path,
        title=se_title,
        n_sequences=se_data.get("n_sequences", 1),
        ax=axes[0],
        **se_kw,
    )
    plot_inner_ttt_matrix(
        inner_data["mean_over_sequences"],
        inner_data.get("std_over_sequences"),
        out_path,
        title=inner_title,
        n_sequences=inner_data.get("n_sequences", 1),
        n_datapoints=n_dp,
        ax=axes[1],
    )

    mappable = axes[0].images[0]
    fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02, label="Accuracy")
    fig.suptitle("Continual forgetting: SE generation (left) vs inner TTT (right)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined heatmaps → {out_path}")


def _se_plot_kwargs(se_data: dict) -> dict:
    n_merge = se_data.get(
        "n_merge",
        se_data.get("n_datapoints", len(se_data["mean_over_sequences"]) - 1),
    )
    cols = len(se_data["mean_over_sequences"][0])
    return {
        "n_merge": n_merge,
        "n_val": se_data.get("n_val", cols),
        "eval_mode": se_data.get("eval_mode", "per_passage"),
        "inner_sft_articles": se_data.get("inner_sft_articles", 1),
    }


# Backward-compatible alias
plot_matrix = plot_se_gen_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot SE-gen (full) and/or continual_self_edits (lower-tri) heatmaps"
    )
    p.add_argument(
        "--results_dir",
        "--se_results_dir",
        dest="se_results_dir",
        type=str,
        default=None,
        help="continual_self_edit_gen_forgetting results directory",
    )
    p.add_argument("--summary", "--se_summary", dest="se_summary", type=str, default=None)
    p.add_argument(
        "--inner_results_dir",
        type=str,
        default=None,
        help="continual_self_edits results directory",
    )
    p.add_argument("--inner_summary", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG when plotting a single matrix (default: forgetting_heatmap*.png)",
    )
    p.add_argument("--se_output", type=str, default=None)
    p.add_argument("--inner_output", type=str, default=None)
    p.add_argument(
        "--combined_output",
        type=str,
        default=None,
        help="Side-by-side figure when both summaries are provided",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Self-edit generation forgetting (held-out val)",
    )
    p.add_argument(
        "--inner_title",
        type=str,
        default="Inner TTT forgetting (continual_self_edits)",
    )
    return p.parse_args()


def _default_combined_path(args: argparse.Namespace) -> Path:
    if args.combined_output:
        return Path(args.combined_output)
    if args.se_results_dir:
        return Path(args.se_results_dir) / "forgetting_heatmaps_combined.png"
    if args.se_summary:
        return Path(args.se_summary).parent / "forgetting_heatmaps_combined.png"
    if args.inner_results_dir:
        return Path(args.inner_results_dir) / "forgetting_heatmaps_combined.png"
    return Path("forgetting_heatmaps_combined.png")


def main() -> None:
    args = parse_args()

    has_se = args.se_results_dir or args.se_summary
    has_inner = args.inner_results_dir or args.inner_summary
    if not has_se and not has_inner:
        raise SystemExit(
            "Provide at least one of: --results_dir/--se_results_dir, --summary, "
            "--inner_results_dir, --inner_summary"
        )

    se_data: Optional[dict] = None
    se_out: Optional[Path] = None
    if has_se:
        if args.se_summary:
            se_data = json.loads(Path(args.se_summary).read_text(encoding="utf-8"))
            se_out = Path(
                args.se_output
                or args.output
                or Path(args.se_summary).with_suffix(".png")
            )
        else:
            se_dir = Path(args.se_results_dir)
            se_data = load_latest_summary(se_dir)
            se_out = Path(
                args.se_output
                or args.output
                or se_dir / (
                    "forgetting_heatmap.png"
                    if not has_inner
                    else "forgetting_heatmap_se_gen.png"
                )
            )

    inner_data: Optional[dict] = None
    inner_out: Optional[Path] = None
    if has_inner:
        if args.inner_summary:
            inner_data = json.loads(
                Path(args.inner_summary).read_text(encoding="utf-8")
            )
            inner_out = Path(
                args.inner_output
                or args.output
                or Path(args.inner_summary).with_suffix(".png")
            )
        else:
            inner_dir = Path(args.inner_results_dir)
            inner_data = load_latest_summary(inner_dir)
            inner_out = Path(
                args.inner_output
                or args.output
                or inner_dir / "forgetting_heatmap_inner_ttt.png"
            )

    if se_data and inner_data:
        plot_combined(
            se_data,
            inner_data,
            _default_combined_path(args),
            se_title=args.title,
            inner_title=args.inner_title,
        )

    if se_data and se_out is not None:
        plot_se_gen_matrix(
            se_data["mean_over_sequences"],
            se_data.get("std_over_sequences"),
            se_out,
            title=args.title,
            n_sequences=se_data.get("n_sequences", 1),
            **_se_plot_kwargs(se_data),
        )

    if inner_data and inner_out is not None:
        n_dp = inner_data.get(
            "n_datapoints", len(inner_data["mean_over_sequences"][0])
        )
        plot_inner_ttt_matrix(
            inner_data["mean_over_sequences"],
            inner_data.get("std_over_sequences"),
            inner_out,
            title=args.inner_title,
            n_sequences=inner_data.get("n_sequences", 1),
            n_datapoints=n_dp,
        )


if __name__ == "__main__":
    main()
