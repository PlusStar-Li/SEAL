export DATA_DIR=/data/SEAL/few-shot/data
export MODEL_DIR=~/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct
export TTI_DIR=/data/SEAL/few-shot/experiments/ttt
export LORA_DIR=/data/SEAL/few-shot/loras
# export VLLM_USE_V1=0
# export VLLM_USE_TRITON_LORA=0
# 强制 vLLM 使用非 Triton 的 LoRA 实现
# export VLLM_USE_TRITON_LORA=0
# # 彻底禁用报错的 Punica 封装路径
# export VLLM_LORA_SKIP_PUNICA=1


python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/training_set_iteration_1 \
    --pretrained_checkpoint=${MODEL_DIR} \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/training_set_iteration_1 \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=11 \
    --n_self_edits=15