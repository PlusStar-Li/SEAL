# general-knowledge/src/EM/train_SFT.py
"""
SFT trainer (ReST-EM outer loop).

Dataset format expected:
{"prompt": "...", "completion": "..."}

After training, saves:
  1. {output_dir}_lora_adapter/  — outer LoRA (W_SE) before merge, for O-LoRA U_hist
  2. {output_dir}/               — merged full model for inference / next RL round
"""
import os
import sys
import argparse
from pathlib import Path

from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel

# Allow flat imports when launched as a script path via accelerate.
_EM_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EM_DIR.parent
for _p in (str(_EM_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import apply_qwen3_thinking_prefix, strip_think_blocks  # noqa: E402
from lora_config import (  # noqa: E402
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    lora_target_modules_csv,
)
from lora_checkpoint_utils import default_lora_adapter_dir, save_outer_lora_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
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
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=LORA_RANK)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default=lora_target_modules_csv(),
    )
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument(
        "--lora_adapter_dir",
        default=None,
        help="Where to save outer LoRA before merge (default: {output_dir}_lora_adapter)",
    )
    p.add_argument(
        "--no_save_lora_adapter",
        action="store_true",
        help="Skip saving pre-merge LoRA adapter and A matrices",
    )
    return p.parse_args()


def _unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def _normalize_example(example, *, instruct_model: bool, thinking_mode: bool):
    prompt = apply_qwen3_thinking_prefix(
        example["prompt"],
        instruct_model=instruct_model,
        thinking_mode=thinking_mode,
    )
    completion = strip_think_blocks(example["completion"])
    return {"prompt": prompt, "completion": completion}


def longest_seq_len(dataset, tok):
    return max(
        len(tok(example["prompt"] + example["completion"]).input_ids)
        for example in dataset
    )


def main() -> None:
    args = parse_args()
    if args.thinking_mode and not args.instruct_model:
        raise SystemExit("[!] --thinking_mode requires --instruct_model")

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(
        lambda ex: _normalize_example(
            ex,
            instruct_model=args.instruct_model,
            thinking_mode=args.thinking_mode,
        )
    )
    print(
        f"[SFT] instruct_model={args.instruct_model} thinking_mode={args.thinking_mode} "
        f"n={len(dataset)}"
    )
    if len(dataset) > 0:
        print("[SFT] prompt tail:", repr(dataset[0]["prompt"][-120:]))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=longest_seq_len(dataset, tokenizer),
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    trainer.train()
    peft_model = _unwrap_model(trainer.model)
    if not isinstance(peft_model, PeftModel):
        raise TypeError(f"Expected PeftModel after SFT, got {type(peft_model)}")

    should_save = not args.no_save_lora_adapter and trainer.is_world_process_zero()
    lora_adapter_dir = args.lora_adapter_dir or default_lora_adapter_dir(args.output_dir)
    if should_save:
        save_outer_lora_checkpoint(
            peft_model,
            lora_adapter_dir,
            metadata={
                "role": "outer_w_se",
                "base_model": args.model_name_or_path,
                "merged_model_dir": args.output_dir,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules.split(","),
                "train_file": args.train_file,
                "instruct_model": args.instruct_model,
                "thinking_mode": args.thinking_mode,
            },
        )
        print(f"Saved outer LoRA adapter → {lora_adapter_dir}")

    merged_model = peft_model.merge_and_unload()
    if should_save:
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
