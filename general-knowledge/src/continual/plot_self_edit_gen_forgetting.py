# general-knowledge/src/continual/plot_self_edit_gen_forgetting.py
"""
Plot lower-triangular forgetting heatmaps (Figure 6 style) from summary JSON files.

Usage:
  python general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \\
    --results_dir general-knowledge/results/continual_self_edit_gen_forgetting/run0

  python general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \\
    --summary path/to/summary_123.json --output figure6.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

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
    return json.loads(max(candidates, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))


def plot_matrix(
    mean_mat: List[List[float]],
    std_mat: Optional[List[List[float]]],
    out_path: Path,
    *,
    title: str,
    n_sequences: int,
    subtitle: str = "",
) -> None:
    data = np.array(mean_mat, dtype=float)
    R, K = data.shape

    mask = np.zeros((R, K), dtype=bool)
    for r in range(1, R):
        mask[r, r:] = True
    plot_data = np.ma.array(data, mask=mask)

    fig, ax = plt.subplots(figsize=(max(7, K * 0.95), max(5.5, R * 0.75)))
    im = ax.imshow(plot_data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adapter accuracy (fresh SE + inner TTT)")

    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            val = data[r, i]
            if np.isnan(val):
                continue
            std_txt = ""
            if std_mat is not None and i < len(std_mat[r]) and not np.isnan(std_mat[r][i]):
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

    ax.set_xlabel("Task index (datapoint)")
    ax.set_ylabel("After merge steps (row 0 = initial model)")
    ax.set_title(title)
    ax.set_xticks(range(K))
    ax.set_yticks(range(R))
    ax.set_xticklabels([f"d{i}" for i in range(K)])
    ylabels = ["base (no merge)"] + [f"after merge 0..{i}" for i in range(1, R)]
    ax.set_yticklabels(ylabels)

    note = (
        "Each cell: regenerate self-edit with current generator, per-task TTT, "
        f"eval on that task only. Averaged over {n_sequences} sequence(s)."
    )
    if subtitle:
        note = subtitle + "\n" + note
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot self-edit generation forgetting heatmap")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--results_dir", type=str, help="Directory with summary_*.json")
    g.add_argument("--summary", type=str, help="Path to a single summary JSON")
    p.add_argument("--output", type=str, default=None, help="Output PNG path")
    p.add_argument(
        "--title",
        type=str,
        default="Self-edit generation forgetting (Figure 6 style)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary:
        data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        out = Path(args.output or Path(args.summary).with_suffix(".png"))
    else:
        results_dir = Path(args.results_dir)
        data = load_latest_summary(results_dir)
        out = Path(
            args.output
            or results_dir / "forgetting_heatmap.png"
        )

    mean_mat = data["mean_over_sequences"]
    std_mat = data.get("std_over_sequences")
    n_seq = data.get("n_sequences", 1)

    plot_matrix(
        mean_mat,
        std_mat,
        out,
        title=args.title,
        n_sequences=n_seq,
    )


if __name__ == "__main__":
    main()
