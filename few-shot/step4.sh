export DATA_DIR=/data/SEAL/few-shot/data
export LORA_DIR=/data/SEAL/few-shot/loras

python self-edit.py \
    --experiment_name=eval_RL_iteration_1_8_epoch \
    --challenge_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --model_name=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --n_tasks=10 \
    --n_self_edits_per_task=5