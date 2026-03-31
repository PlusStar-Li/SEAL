export DATA_DIR=/data/SEAL/few-shot/data
export TTI_DIR=/data/SEAL/few-shot/experiments/ttt
export LORA_DIR=/data/SEAL/few-shot/loras


python eval-self-edits.py \
    --experiment_folder=${TTI_DIR}/eval_set_RL_iteration_1_8_epoch \
    --pretrained_checkpoint=${LORA_DIR}/self-edit/training_set_iteration_1/RL_trained_model_iteration_1_8_epoch \
    --lora_checkpoints_folder=${LORA_DIR}/self-edit/eval_RL_iteration_1_8_epoch \
    --temperature=0 \
    --n_sample=1 \
    --data_file=${DATA_DIR}/arc-agi_evaluation_challenges_filtered_1B_eval_set.json \
    --solution_file=${DATA_DIR}/arc-agi_evaluation_solutions_filtered_1B_eval_set.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format \
    --num_examples=9 \
    --n_self_edits=5