#!/bin/bash
#SBATCH --job-name=b2_olora
#SBATCH --output=logs/%A_%x.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
# source ~/.bashrc
# conda activate seal_env
# cd ~/SEAL
# set -a && source .env && set +a

# ====================================================================== #
#  Baseline 2: Standard O-LoRA (λ≡1.0) — aligned with Baseline 1 shell
# ====================================================================== #
INDEX=0
MODEL_NAME="models/iter2"
DATASET="general-knowledge/data/squad_val.json"

LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_LR=1e-3

VLLM_SERVER_GPUS="0"
PY_DRIVER_GPU="1"
PORT=$((8001 + INDEX))
ZMQ_PORT=$((5555 + INDEX))
SEED=$((42 + INDEX))

MAX_TOKENS=8192
TEMPERATURE=1.0
TOP_P=0.95

N_SEQUENCES=3
N_MERGE=8
N_VAL=8

INNER_SFT_ARTICLES=1
K_COMP=1
FINETUNE_EPOCHS=10
BATCH_SIZE=1
GRAD_ACC=1
SPLIT_NEWLINES=1

# --- O-LoRA (Baseline 2 only) ---
OLORA_MODE="standard"
LAMBDA_FIXED=1.0
GAMMA=1.0
U_SE_PATH="models/iter2_lora_adapter/lora_A_matrices.pt"
SPLITS_DIR="general-knowledge/results/continual_self_edit_gen_forgetting/single/run0/splits"

OUTPUT_DIR="general-knowledge/results/baselines/baseline2_standard_olora/run${INDEX}"
mkdir -p "${OUTPUT_DIR}"

# --------------------------------------------------------------------- #

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}
export OMP_NUM_THREADS=1

echo "======== Baseline 2 Standard O-LoRA ========"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}  U_SE=${U_SE_PATH}"
echo "  seq=${N_SEQUENCES}  merge=${N_MERGE}  val=${N_VAL}  seed=${SEED}"
echo "  λ=${LAMBDA_FIXED}  γ=${GAMMA}  mode=${OLORA_MODE}"
echo "  splits_dir=${SPLITS_DIR}"
echo "  inner: r=${LORA_RANK} α=${LORA_ALPHA} epochs=${FINETUNE_EPOCHS} lr=${FINETUNE_LR}"
echo "==========================================="

SN_FLAG=""
if [[ "${SPLIT_NEWLINES}" == "1" ]]; then
    SN_FLAG="--split_newlines"
elif [[ "${SPLIT_NEWLINES}" == "0" ]]; then
    SN_FLAG="--no_split_newlines"
fi

SPLITS_ARG=""
if [[ -n "${SPLITS_DIR}" && -d "${SPLITS_DIR}" ]]; then
    SPLITS_ARG="--splits_dir ${SPLITS_DIR}"
fi

python3 -u -m general-knowledge.src.continual.baseline2_standard_olora \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --u_se_path "${U_SE_PATH}" \
    ${SPLITS_ARG} \
    --olora_mode "${OLORA_MODE}" \
    --lambda_fixed "${LAMBDA_FIXED}" \
    --gamma "${GAMMA}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_lr ${FINETUNE_LR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --n_sequences ${N_SEQUENCES} \
    --n_merge ${N_MERGE} \
    --n_val ${N_VAL} \
    --inner_sft_articles ${INNER_SFT_ARTICLES} \
    --k_completions ${K_COMP} \
    ${SN_FLAG} \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED}

echo "Plotting heatmap..."
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
    --results_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/forgetting_heatmap_se_gen.png"

echo "Job finished."
