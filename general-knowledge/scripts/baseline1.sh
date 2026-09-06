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

# ====================================================================== #
#  Switch experiment here:  INNER_MODE="single"  |  INNER_MODE="cpt"
# ====================================================================== #
INNER_MODE="single" # "single", "cpt"

INDEX=0
MODEL_NAME="models/qwen3_iter2" # Qwen/Qwen2.5-7B, models/iter1, models/iter2
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

# -------- Presets (aligned with continual_self_edits vs CPT.sh) -------- #
case "${INNER_MODE}" in
    single)
        # Baseline 1: vanilla LoRA continual SE-gen forgetting (single passage)
        INNER_SFT_ARTICLES=1
        K_COMP=1
        FINETUNE_EPOCHS=10
        BATCH_SIZE=1
        GRAD_ACC=1
        SPLIT_NEWLINES=1
        OUTPUT_DIR="general-knowledge/results/baselines/qwen3_baseline1_vanilla_sft/run${INDEX}"
        ;;
    cpt)
        # CPT.sh base_5c: N-article corpus per task, k=5, split_newlines=0
        INNER_SFT_ARTICLES=100
        K_COMP=5
        FINETUNE_EPOCHS=3
        BATCH_SIZE=4
        GRAD_ACC=2
        SPLIT_NEWLINES=0
        OUTPUT_DIR="general-knowledge/results/continual_self_edit_gen_forgetting/cpt/run${INDEX}"
        ;;
    *)
        echo "[!] Unknown INNER_MODE='${INNER_MODE}'. Use: single | cpt"
        exit 1
        ;;
esac

# Inner TTT: 1 = self-edit implications only; 0 = SEAL default (also train on passage)
NO_ADD_CONTEXT=1

mkdir -p "${OUTPUT_DIR}"

# Optional: force split_newlines after preset (leave unset to use preset above)
# SPLIT_NEWLINES_OVERRIDE=0

if [[ -n "${SPLIT_NEWLINES_OVERRIDE:-}" ]]; then
    SPLIT_NEWLINES="${SPLIT_NEWLINES_OVERRIDE}"
fi

# --------------------------------------------------------------------- #

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}
export OMP_NUM_THREADS=1

echo "======== SE gen forgetting ========"
echo "  INNER_MODE=${INNER_MODE}  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  seq=${N_SEQUENCES}  merge=${N_MERGE}  val=${N_VAL}"
echo "  inner_sft_articles=${INNER_SFT_ARTICLES}  k_completions=${K_COMP}"
echo "  finetune: epochs=${FINETUNE_EPOCHS} lr=${FINETUNE_LR} bs=${BATCH_SIZE} ga=${GRAD_ACC}"
echo "  split_newlines=${SPLIT_NEWLINES}  no_add_context=${NO_ADD_CONTEXT}"
echo "==================================="

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
    --n_merge ${N_MERGE} \
    --n_val ${N_VAL} \
    --inner_sft_articles ${INNER_SFT_ARTICLES} \
    --k_completions ${K_COMP} \
    ${SN_FLAG} \
    ${NO_CTX_FLAG} \
    --output_dir "${OUTPUT_DIR}" \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED} \
    ${INSTRUCT_MODEL:+--instruct_model} \
    ${THINKING_MODE:+--thinking_mode}

echo "Plotting heatmaps..."
python3 general-knowledge/src/continual/plot_self_edit_gen_forgetting.py \
    --results_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/forgetting_heatmap_se_gen.png" \
    --combined_output "${OUTPUT_DIR}/forgetting_heatmaps_combined.png"

echo "Job finished."
