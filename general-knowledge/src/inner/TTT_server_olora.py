# general-knowledge/src/inner/TTT_server_olora.py
"""
Inner-loop TTT server with O-LoRA orthogonal penalty on LoRA A matrices.

Loss = L_SFT + lambda_t * gamma * L_ortho(A_t, U_hist)

JSON request adds optional ``olora`` block:
    {
        "enabled": true,
        "lambda_t": 1.0,
        "gamma": 1.0,
        "U_hist_dir": "/path/to/U_hist/seq0"
    }
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import zmq
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..continual.olora_utils import compute_ortho_loss, load_u_hist
from ..lora_config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES
from ..utils import (
    extract_final_answer,
    format_answer_prompts,
    format_grade_prompts,
    generate,
    grade_with_gpt4,
    load_adapter,
    score_proxy_with_gpt4,
    set_vllm_api_url,
    strip_think_blocks,
    unload_adapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger()


class OLoRATrainer(Trainer):
    """HF Trainer with optional O-LoRA orthogonal regularizer on lora_A."""

    def __init__(
        self,
        *args,
        u_hist=None,
        lambda_t: float = 1.0,
        lambda_weights: Optional[List[float]] = None,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.u_hist = u_hist or []
        self.lambda_t = float(lambda_t)
        self.lambda_weights = lambda_weights
        self.gamma = float(gamma)
        self.last_sft_loss: Optional[float] = None
        self.last_ortho_loss: Optional[float] = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        sft_loss = outputs.loss
        if self.lambda_weights is not None:
            ortho = compute_ortho_loss(
                model, self.u_hist, lambda_weights=self.lambda_weights
            )
            loss = sft_loss + self.gamma * ortho
        else:
            ortho = compute_ortho_loss(model, self.u_hist)
            loss = sft_loss + self.lambda_t * self.gamma * ortho
        self.last_sft_loss = float(sft_loss.detach().item())
        self.last_ortho_loss = float(ortho.detach().item())
        return (loss, outputs) if return_outputs else loss


def accuracy_and_texts(
    questions: List[Dict[str, str]],
    answer_model_ref: str,
    sampling: Dict[str, Any],
    stop_ids: List[int],
    instruct_model: bool,
    chain_of_thought: bool = False,
    thinking_mode: bool = False,
) -> tuple[float, List[str], List[bool]]:
    ans_out = generate(
        format_answer_prompts(
            questions,
            instruct_model=instruct_model,
            chain_of_thought=chain_of_thought,
            thinking_mode=thinking_mode,
        ),
        answer_model_ref,
        sampling,
        stop_ids,
    ) or []
    preds = [strip_think_blocks(o.get("text", "")) for o in ans_out]
    if chain_of_thought:
        preds = [extract_final_answer(p) for p in preds]

    verdicts: List[bool] = [False] * len(preds)
    q_sub, p_sub, idx_sub = [], [], []
    for i, (q, p) in enumerate(zip(questions, preds)):
        if p.strip():
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)
    if q_sub:
        graded = grade_with_gpt4(format_grade_prompts(q_sub, p_sub))
        for i, v in zip(idx_sub, graded):
            verdicts[i] = v
    acc = sum(verdicts) / len(questions) if questions else 0.0
    return acc, preds, verdicts


def _parse_olora(msg: Dict[str, Any]) -> Dict[str, Any]:
    olora = msg.get("olora") or {}
    enabled = bool(olora.get("enabled", False))
    lambda_t = float(olora.get("lambda_t", 1.0))
    gamma = float(olora.get("gamma", 1.0))
    u_hist_dir = olora.get("U_hist_dir") or olora.get("u_hist_dir")
    u_hist_path = olora.get("U_hist_path") or olora.get("u_hist_path")
    u_hist_loc = u_hist_dir or u_hist_path
    u_hist = load_u_hist(u_hist_loc) if enabled and u_hist_loc else []
    init_adapter_path = olora.get("init_adapter_path")
    reuse_mode = bool(olora.get("reuse_mode", False))
    lambda_weights_raw = olora.get("lambda_weights")
    lambda_weights = (
        [float(x) for x in lambda_weights_raw]
        if lambda_weights_raw is not None
        else None
    )
    return {
        "enabled": enabled,
        "lambda_t": lambda_t,
        "lambda_weights": lambda_weights,
        "gamma": gamma,
        "u_hist_dir": u_hist_dir,
        "u_hist_path": u_hist_path,
        "u_hist": u_hist,
        "init_adapter_path": init_adapter_path,
        "reuse_mode": reuse_mode,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zmq_port", type=int, default=5555)
    p.add_argument("--vllm_api_url", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
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
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=1.0)
    p.add_argument("--eval_max_tokens", type=int, default=64)
    p.add_argument(
        "--keep_adapter_dir",
        action="store_true",
        help="Keep tmp adapter dir for outer driver merge / A extraction.",
    )
    args = p.parse_args()
    if args.thinking_mode and not args.instruct_model:
        raise SystemExit("[!] --thinking_mode requires --instruct_model")

    set_vllm_api_url(args.vllm_api_url)

    LOG.info("Loading base model %s (O-LoRA TTT)...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    if args.instruct_model:
        stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    else:
        stop_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    ctx, sock = zmq.Context(), None
    try:
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{args.zmq_port}")
        LOG.info("O-LoRA TTT ZMQ listening at tcp://*:%d", args.zmq_port)
        step = 0
        while True:
            LOG.info("Waiting for request...")
            msg = sock.recv_json()
            LOG.info("Received request keys: %s", list(msg.keys()))

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "bye"})
                break

            recv_start = time.time()
            reply: Dict[str, Any] = {}
            try:
                seed = (int(time.time() * 1e6) + step) & 0xFFFFFFFF
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                LOG.info("Step %d seed %d", step, seed)

                train_sequences = msg.get("train_sequences")
                questions = msg.get("eval_questions", [])
                lora_rank = msg.get("lora_rank", LORA_RANK)
                lora_alpha = msg.get("lora_alpha", LORA_ALPHA)
                lora_dropout = msg.get("lora_dropout", LORA_DROPOUT)
                finetune_epochs = msg.get("finetune_epochs", 10)
                finetune_lr = msg.get("finetune_lr", 1e-3)
                batch_size = msg.get("batch_size", 1)
                gradient_accumulation_steps = msg.get(
                    "gradient_accumulation_steps", 1
                )
                end_mask_substring = msg.get("end_mask_substring")
                baseline_eval = bool(msg.get("baseline_eval", True))
                chain_of_thought = bool(msg.get("chain_of_thought", False))
                reward_mode = msg.get("reward_mode", "ttt")
                completion_raw = msg.get("comp_raw", "")
                olora_cfg = _parse_olora(msg)

                sampling_cfg = {
                    "n": 1,
                    "temperature": args.eval_temperature,
                    "top_p": args.eval_top_p,
                    "max_tokens": args.eval_max_tokens,
                }

                title, article_context = "", ""
                if questions:
                    title = questions[0].get("title", "") or ""
                    article_context = questions[0].get("context", "") or ""

                if reward_mode in ("proxy", "both"):
                    try:
                        proxy_scores = score_proxy_with_gpt4(
                            title=title,
                            context=article_context,
                            completion=completion_raw,
                        )
                    except Exception:
                        proxy_scores = {
                            "length": 1,
                            "diversity": 1,
                            "quality": 1,
                            "correctness": 1,
                            "final": 4,
                        }
                else:
                    proxy_scores = {}

                if reward_mode == "proxy":
                    reply = {
                        "baseline_accuracy": 0.0,
                        "adapter_accuracy": 0.0,
                        "adapter_gain": 0.0,
                        "baseline_texts": [""] * len(questions),
                        "adapter_texts": [""] * len(questions),
                        "baseline_correct": [False] * len(questions),
                        "adapter_correct": [False] * len(questions),
                        "gains": [0] * len(questions),
                        "proxy_scores": proxy_scores,
                    }
                elif baseline_eval:
                    base_acc, base_texts, base_ok = accuracy_and_texts(
                        questions,
                        answer_model_ref=args.model,
                        sampling=sampling_cfg,
                        stop_ids=stop_ids,
                        instruct_model=args.instruct_model,
                        chain_of_thought=chain_of_thought,
                        thinking_mode=args.thinking_mode,
                    )
                else:
                    base_acc, base_texts, base_ok = (
                        0.0,
                        [""] * len(questions),
                        [False] * len(questions),
                    )

                if reward_mode != "proxy" and not train_sequences:
                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy": round(base_acc, 4),
                        "adapter_gain": 0.0,
                        "baseline_texts": base_texts,
                        "adapter_texts": base_texts,
                        "baseline_correct": base_ok,
                        "adapter_correct": base_ok,
                        "gains": [0] * len(base_ok),
                        "olora": {
                            "enabled": olora_cfg["enabled"],
                            "lambda_t": olora_cfg["lambda_t"],
                            "gamma": olora_cfg["gamma"],
                            "u_hist_size": len(olora_cfg["u_hist"]),
                        },
                    }
                elif reward_mode != "proxy":
                    tmp_tag = f"inner_TTT_{step}"
                    tmp_dir = Path(f"models/tmp_{args.zmq_port}_{tmp_tag}")
                    os.makedirs(tmp_dir, exist_ok=True)

                    sub_ids = (
                        tokenizer.encode(end_mask_substring, add_special_tokens=False)
                        if end_mask_substring
                        else []
                    )
                    rows = []
                    for idx, seq in enumerate(train_sequences):
                        tok = tokenizer(
                            seq,
                            truncation=True,
                            max_length=args.max_seq_length,
                            padding="max_length",
                        )
                        labels = tok["input_ids"].copy()
                        if sub_ids:
                            m = len(sub_ids)
                            for i in range(len(labels) - m + 1):
                                if labels[i : i + m] == sub_ids:
                                    for j in range(i + m):
                                        labels[j] = -100
                                    break
                        if idx < 3 and not sub_ids:
                            LOG.info("TRAIN[%d] %s", idx, seq[:200])
                        rows.append(
                            {
                                "input_ids": tok["input_ids"],
                                "attention_mask": tok["attention_mask"],
                                "labels": labels,
                            }
                        )

                    ds = HFDataset.from_list(rows)
                    collator = DataCollatorWithPadding(tokenizer)
                    init_adapter_path = olora_cfg.get("init_adapter_path")
                    if init_adapter_path and Path(init_adapter_path).exists():
                        LOG.info(
                            "Warm-starting LoRA from %s (reuse_mode=%s)",
                            init_adapter_path,
                            olora_cfg.get("reuse_mode", False),
                        )
                        lora_model = PeftModel.from_pretrained(
                            base_model,
                            str(init_adapter_path),
                            is_trainable=True,
                        )
                    else:
                        if init_adapter_path:
                            LOG.warning(
                                "init_adapter_path missing (%s); fresh LoRA init",
                                init_adapter_path,
                            )
                        lora_cfg = LoraConfig(
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            target_modules=LORA_TARGET_MODULES,
                        )
                        lora_model = get_peft_model(base_model, lora_cfg)

                    trainer_cls = OLoRATrainer if olora_cfg["enabled"] else Trainer
                    trainer_kwargs: Dict[str, Any] = {}
                    if olora_cfg["enabled"]:
                        trainer_kwargs = {
                            "u_hist": olora_cfg["u_hist"],
                            "lambda_t": olora_cfg["lambda_t"],
                            "gamma": olora_cfg["gamma"],
                        }
                        if olora_cfg.get("lambda_weights") is not None:
                            trainer_kwargs["lambda_weights"] = olora_cfg[
                                "lambda_weights"
                            ]

                    trainer = trainer_cls(
                        model=lora_model,
                        args=TrainingArguments(
                            output_dir=str(tmp_dir),
                            per_device_train_batch_size=batch_size,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            num_train_epochs=finetune_epochs,
                            learning_rate=finetune_lr,
                            logging_steps=1,
                            save_strategy="no",
                            report_to="none",
                            remove_unused_columns=False,
                            fp16=False,
                            bf16=torch.cuda.is_available()
                            and torch.cuda.is_bf16_supported(),
                            seed=seed,
                        ),
                        train_dataset=ds,
                        data_collator=collator,
                        **trainer_kwargs,
                    )
                    trainer.train()

                    adapter_path = tmp_dir / "final_adapter"
                    lora_model.save_pretrained(str(adapter_path))

                    olora_metrics = {
                        "enabled": olora_cfg["enabled"],
                        "lambda_t": olora_cfg["lambda_t"],
                        "lambda_weights": olora_cfg.get("lambda_weights"),
                        "gamma": olora_cfg["gamma"],
                        "u_hist_size": len(olora_cfg["u_hist"]),
                        "init_adapter_path": olora_cfg.get("init_adapter_path"),
                        "reuse_mode": olora_cfg.get("reuse_mode", False),
                        "sft_loss": None,
                        "ortho_loss": None,
                    }
                    if olora_cfg["enabled"] and isinstance(trainer, OLoRATrainer):
                        olora_metrics["sft_loss"] = trainer.last_sft_loss
                        olora_metrics["ortho_loss"] = trainer.last_ortho_loss
                        LOG.info(
                            "O-LoRA step %d  sft=%s ortho=%s  λ=%.3f γ=%.3f  |U|=%d",
                            step,
                            olora_metrics["sft_loss"],
                            olora_metrics["ortho_loss"],
                            olora_cfg["lambda_t"],
                            olora_cfg["gamma"],
                            len(olora_cfg["u_hist"]),
                        )

                    adapter_name = tmp_tag
                    load_adapter(str(adapter_path), adapter_name)
                    adapter_acc, adapter_texts, adapter_ok = accuracy_and_texts(
                        questions,
                        answer_model_ref=adapter_name,
                        sampling=sampling_cfg,
                        stop_ids=stop_ids,
                        instruct_model=args.instruct_model,
                        chain_of_thought=chain_of_thought,
                        thinking_mode=args.thinking_mode,
                    )
                    gains = [
                        1 if a and not b else -1 if b and not a else 0
                        for b, a in zip(base_ok, adapter_ok)
                    ]
                    unload_adapter(adapter_name)

                    if not args.keep_adapter_dir:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    gc.collect()
                    torch.cuda.empty_cache()

                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy": round(adapter_acc, 4),
                        "adapter_gain": round(adapter_acc - base_acc, 4),
                        "baseline_texts": base_texts,
                        "adapter_texts": adapter_texts,
                        "baseline_correct": base_ok,
                        "adapter_correct": adapter_ok,
                        "gains": gains,
                        "olora": olora_metrics,
                    }
                    if reward_mode in ("proxy", "both"):
                        reply["proxy_scores"] = proxy_scores

                    LOG.info(
                        "Step %d base %.3f adapter %.3f (%.2fs)",
                        step,
                        base_acc,
                        adapter_acc,
                        time.time() - recv_start,
                    )
            except Exception as e:
                LOG.exception("Error processing request.")
                reply = {"error": f"{type(e).__name__}: {e}"}
            finally:
                sock.send_json(reply)
                step += 1
    finally:
        if sock:
            sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
