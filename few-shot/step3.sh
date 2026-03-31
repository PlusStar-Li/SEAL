export MODEL_DIR=~/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct
export LORA_DIR=/data/SEAL/few-shot/loras

python BC-self-edit.py \
    --configs_and_indices=${LORA_DIR}/self-edit/training_set_iteration_1/final_configs_and_indices.json \
    --results=/data/SEAL/few-shot/final_results.json \
    --model_name=${MODEL_DIR} \
    --lora_rank=16 \
    --lora_alpha=16 \
    --num_train_epochs=8 \
    --per_device_train_batch_size=5 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5