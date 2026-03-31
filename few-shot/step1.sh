export DATA_DIR=/data/SEAL/few-shot/data
export MODEL_DIR=~/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_TRITON_FLASH_ATTN=1
# rm -rf ~/.triton/cache/*
# export VLLM_GPU_MEMORY_UTILIZATION=0.7


python self-edit.py \
    --experiment_name=training_set_iteration_1 \
    --challenge_file=${DATA_DIR}/arc-agi_training_challenges_filtered_1B_training_set.json \
    --solution_file=${DATA_DIR}/arc-agi_training_solutions_filtered_1B_training_set.json \
    --model_name=${MODEL_DIR} \
    --n_tasks=12 \
    --n_self_edits_per_task=15