#!/bin/bash
#SBATCH --job-name=se_gen_forget
#SBATCH --output=logs/%A_%x.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
# source ~/.bashrc
# conda activate seal_env
# cd ~/SEAL
# set -a && source .env && set +a

# -------- User-editable (matches continual_self_edits.sh) ------------ #
INDEX=0

MODEL_NAME="Qwen/Qwen2.5-7B"
DATASET="general-knowledge/data/squad_val.json"
OUTPUT_DIR="general-knowledge/results/continual_self_edit_gen_forgetting/run${INDEX}"
mkdir -p "${OUTPUT_DIR}"

LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_EPOCHS=10
FINETUNE_LR=1e-3
BATCH_SIZE=1
GRAD_ACC=1

VLLM_SERVER_GPUS="0"
PY_DRIVER_GPU="1"
PORT=$((8001 + INDEX))
ZMQ_PORT=$((5555 + INDEX))
SEED=$((42 + INDEX))

MAX_TOKENS=8192
TEMPERATURE=1.0
TOP_P=0.95

N_SEQUENCES=8
N_DATAPOINTS=8
# --------------------------------------------------------------------- #

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}

echo "Self-edit generation forgetting experiment"
python3 -u -m general-knowledge.src.continual.continual_self_edit_gen_forgetting \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_lr ${FINETUNE_LR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --n_sequences ${N_SEQUENCES} \
    --n_datapoints ${N_DATAPOINTS} \
    --output_dir "${OUTPUT_DIR}" \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED}

echo "Plotting heatmap from latest summary..."
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
    --results_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/forgetting_heatmap.png"

echo "Job finished."
