#!/bin/bash
#SBATCH --job-name=b3_adaptive_olora
#SBATCH --output=logs/%A_%x.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
# source ~/.bashrc
# conda activate seal_env
# cd ~/SEAL
# set -a && source .env && set +a

# ====================================================================== #
#  Baseline 3: GPT-4.1 Adaptive O-LoRA (BGE ablation via LAMBDA_SOURCE)
# ====================================================================== #

export HF_HOME=/mnt/afs/visitor38/cache
export HF_ENDPOINT=https://hf-mirror.com
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
INSTRUCT_MODEL=1     # Qwen instruct chat template
THINKING_MODE=1      # Qwen3 thinking (requires INSTRUCT_MODEL=1)

N_SEQUENCES=8
N_MERGE=8
N_VAL=8
# Resume: set to the sequence index to re-run from (0 = fresh start).
# Interrupted during seq5 with checkpoint_summary.json present → START_SEQ=5.
START_SEQ=6

INNER_SFT_ARTICLES=1
K_COMP=1
FINETUNE_EPOCHS=10
BATCH_SIZE=1
GRAD_ACC=1
SPLIT_NEWLINES=1
# Inner TTT: 1 = self-edit only; 0 = SEAL default (append passage row)
NO_ADD_CONTEXT=0

# --- Adaptive O-LoRA (Baseline 3) ---
OLORA_MODE="adaptive"
LAMBDA_SOURCE="gpt4"               # gpt4 (Full) | embedding (ablation)
EMBED_MODEL="BAAI/bge-large-en-v1.5"
TAU=0.000001
GAMMA=1.0
# Optional: set to e.g. 1.0 to skip GPT-4/embedding and use fixed lambdas for all U_hist slots
# FIXED_LAMBDA_T=""
U_SE_PATH="models/iter2_lora_adapter/lora_A_matrices.pt"
SPLITS_DIR="general-knowledge/baselines/results/baseline1_vanilla_sft/run0/splits"

if [[ "${LAMBDA_SOURCE}" == "gpt4" ]]; then
    OUTPUT_DIR="general-knowledge/results/baselines/baseline3_gpt4_noreuse_full_gamma/run${INDEX}"
else
    OUTPUT_DIR="general-knowledge/results/baselines/baseline3_embedding_adaptive_olora/run${INDEX}"
fi
mkdir -p "${OUTPUT_DIR}"

# --------------------------------------------------------------------- #

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}
export OMP_NUM_THREADS=1

echo "======== Baseline 3 Adaptive O-LoRA ========"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  model=${MODEL_NAME}  U_SE=${U_SE_PATH}"
echo "  seq=${N_SEQUENCES}  merge=${N_MERGE}  val=${N_VAL}  seed=${SEED}"
echo "  start_seq=${START_SEQ}"
echo "  λ_source=${LAMBDA_SOURCE}  τ=${TAU}  γ=${GAMMA}"
echo "  fixed_lambda_t=${FIXED_LAMBDA_T:-<adaptive>}"
echo "  splits_dir=${SPLITS_DIR}"
echo "  inner: r=${LORA_RANK} α=${LORA_ALPHA} epochs=${FINETUNE_EPOCHS} lr=${FINETUNE_LR}"
echo "  no_add_context=${NO_ADD_CONTEXT}"
echo "==========================================="

SN_FLAG=""
if [[ "${SPLIT_NEWLINES}" == "1" ]]; then
    SN_FLAG="--split_newlines"
elif [[ "${SPLIT_NEWLINES}" == "0" ]]; then
    SN_FLAG="--no_split_newlines"
fi

NO_CTX_FLAG=""
if [[ "${NO_ADD_CONTEXT}" == "1" ]]; then
    NO_CTX_FLAG="--no_add_context"
fi

SPLITS_ARG=""
if [[ -n "${SPLITS_DIR}" && -d "${SPLITS_DIR}" ]]; then
    SPLITS_ARG="--splits_dir ${SPLITS_DIR}"
fi

FIXED_LAMBDA_ARG=""
if [[ -n "${FIXED_LAMBDA_T:-}" ]]; then
    FIXED_LAMBDA_ARG="--fixed_lambda_t ${FIXED_LAMBDA_T}"
fi

START_SEQ_ARG=""
if [[ "${START_SEQ}" -gt 0 ]]; then
    START_SEQ_ARG="--start_seq ${START_SEQ}"
fi

python3 -u -m general-knowledge.src.continual.baseline_olora \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --u_se_path "${U_SE_PATH}" \
    ${SPLITS_ARG} \
    --olora_mode "${OLORA_MODE}" \
    --lambda_source "${LAMBDA_SOURCE}" \
    --embed_model "${EMBED_MODEL}" \
    --tau ${TAU} \
    --gamma ${GAMMA} \
    ${FIXED_LAMBDA_ARG} \
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
    ${NO_CTX_FLAG} \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    ${INSTRUCT_MODEL:+--instruct_model} \
    ${THINKING_MODE:+--thinking_mode} \
    ${START_SEQ_ARG}

echo "Plotting heatmap..."
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
    --results_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/forgetting_heatmap_se_gen.png"

echo "Job finished."
