#!/usr/bin/env python3
"""Recover checkpoint_summary.json from a Baseline O-LoRA driver stdout log."""
from __future__ import annotations

import argparse
from pathlib import Path

from .baseline_olora_checkpoint import recover_checkpoint_from_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover checkpoint_summary.json from baseline_olora driver log"
    )
    p.add_argument("--log", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--max_seq",
        type=int,
        required=True,
        help="Recover sequences with seq_idx < max_seq",
    )
    p.add_argument("--n_sequences", type=int, default=8)
    p.add_argument("--n_merge", type=int, default=8)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--olora_mode", default="adaptive", choices=["standard", "adaptive"])
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--lambda_fixed", type=float, default=1.0)
    p.add_argument("--experiment", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment:
        experiment_name = args.experiment
    elif args.olora_mode == "standard":
        experiment_name = "baseline2_standard_olora"
    else:
        experiment_name = "baseline3_gpt4_adaptive_olora"

    out = recover_checkpoint_from_log(
        Path(args.log),
        Path(args.output_dir),
        max_seq_exclusive=args.max_seq,
        n_sequences=args.n_sequences,
        n_merge=args.n_merge,
        n_val=args.n_val,
        experiment_name=experiment_name,
        args_overrides={
            "olora_mode": args.olora_mode,
            "tau": args.tau,
            "gamma": args.gamma,
            "lambda_fixed": args.lambda_fixed,
        },
    )
    print(f"Recovered checkpoint → {out}")


if __name__ == "__main__":
    main()
