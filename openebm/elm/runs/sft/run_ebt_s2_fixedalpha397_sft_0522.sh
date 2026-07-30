#!/bin/bash

################################################################################
# EBT d26 S2 fixed-alpha397 SFT training script (offline mode)
#
# Fine-tunes a base model trained by:
#   openebm/elm/runs/rjob/run_ebt_s2_fixedalpha397_2node_8gpu.sh
#
# Usage:
#   bash openebm/elm/runs/sft/run_ebt_s2_fixedalpha397_sft_0522.sh
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NOVA_HOME="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ -z "${NOVA_HOME:-}" ]]; then
    if [[ -f "${PWD}/openebm/elm/train.py" && -f "${PWD}/openebm/elm/runs/utils/exp_layout.sh" ]]; then
        NOVA_HOME="${PWD}"
    else
        NOVA_HOME="${SCRIPT_NOVA_HOME}"
    fi
fi
RUNS_DIR="${NOVA_HOME}/openebm/elm/runs"
TRAIN_SCRIPT="${NOVA_HOME}/openebm/elm/train.py"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
    export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
fi

cd "${NOVA_HOME}"
export PYTHONPATH="${NOVA_HOME}:${PYTHONPATH}"
export EBT_RUNS_ROOT="${EBT_RUNS_ROOT:-${NOVA_HOME}/logs/ebt_runs}"

################################################################################
# Basic config
################################################################################

# EXP_ID points to the corresponding base_train experiment family.
export EXP_ID="${EXP_ID:-d26-ctx2048-20260522-s2-fixedalpha397}"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

################################################################################
# Base checkpoint
################################################################################

# Keep this explicit to avoid accidentally SFT-ing from a different base checkpoint.
PRETRAIN_CKPT="/mnt/shared-storage-user/luyudong/nova/logs/checkpoints/s2-fixedalpha397-2node-8gpu-bf16mixed_0522_0702_d26_ctx2048_bs512_lr0.0012_2nodes_8gpus/s=step=6999-d26-ctx2048-lr0.0012-bs1x32-muon_adamw-valid_loss=valid_loss=2.5550.ckpt"

################################################################################
# Environment
################################################################################

HOME="/mnt/shared-storage-user/luyudong/nanochat"
export NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"
export NANOCHAT_SFT_DATA_DIR="${NANOCHAT_SFT_DATA_DIR:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="offline"

# Offline mode must be set before torchrun starts.
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

################################################################################
# EBT core hyperparameters aligned with run_ebt_s2_fixedalpha397_2node_8gpu.sh
################################################################################

MCMC_STEP_SIZE=397.0
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
NO_MCMC_DETACH=false
TRUNCATE_MCMC=true

################################################################################
# Batch config
################################################################################

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

CONTEXT_LENGTH=2048
DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32

################################################################################
# SFT learning-rate config
################################################################################

# Base LR is 0.0012 in run_ebt_s2_fixedalpha397_2node_8gpu.sh; SFT uses a 5x lower main LR.
PEAK_LR=0.00024

# Muon / AdamW group LRs are absolute values, not scaled by PEAK_LR.
SFT_MUON_LR=0.002
SFT_EMBEDDING_LR=0.03
SFT_VOCAB_TO_EMBED_LR=0.001
SFT_SCALAR_LR=0.004

################################################################################
# Optimizer config
################################################################################

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# SFT steps
################################################################################

MAX_STEPS=3000
MAX_SCHEDULING_STEPS=3000

################################################################################
# Validation and checkpointing
################################################################################

VAL_CHECK_INTERVAL=500
LIMIT_VAL_BATCHES=50
SAVE_TOP_K=2

################################################################################
# Optional flags
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0057 --warmdown_ratio 0.5 --final_lr_frac 0.05 \
--optimizer muon_adamw --muon_lr ${SFT_MUON_LR} --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr ${SFT_EMBEDDING_LR} --adamw_vocab_to_embed_lr ${SFT_VOCAB_TO_EMBED_LR} --adamw_scalar_lr ${SFT_SCALAR_LR} --adamw_dmodel_lr_scaling \
--muon_momentum_warmup_steps 300 --ffn_dim_multiplier 2.67"

COMPILE_FLAGS="--compile_model --compile_mode transformer_only"
WANDB_FLAGS=""

################################################################################
# Offline dataset and checkpoint checks
################################################################################

echo ""
echo "=================================="
echo "  Check offline SFT datasets"
echo "=================================="
echo "Dataset path: ${NANOCHAT_SFT_DATA_DIR}"

MISSING_DATASETS=()

[ ! -d "${NANOCHAT_SFT_DATA_DIR}/smol-smoltalk" ] && MISSING_DATASETS+=("smol-smoltalk")
[ ! -d "${NANOCHAT_SFT_DATA_DIR}/mmlu" ] && MISSING_DATASETS+=("mmlu")
[ ! -d "${NANOCHAT_SFT_DATA_DIR}/gsm8k" ] && MISSING_DATASETS+=("gsm8k")
[ ! -f "${NANOCHAT_SFT_DATA_DIR}/words_alpha.txt" ] && MISSING_DATASETS+=("words_alpha.txt")

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo "Missing offline datasets:"
    for dataset in "${MISSING_DATASETS[@]}"; do
        echo "  - ${dataset}"
    done
    echo ""
    echo "Set NANOCHAT_SFT_DATA_DIR to an existing offline SFT data directory."
    exit 1
fi

echo "Found SmolTalk: ${NANOCHAT_SFT_DATA_DIR}/smol-smoltalk"
echo "Found MMLU: ${NANOCHAT_SFT_DATA_DIR}/mmlu"
echo "Found GSM8K: ${NANOCHAT_SFT_DATA_DIR}/gsm8k"
echo "Found words_alpha.txt: ${NANOCHAT_SFT_DATA_DIR}/words_alpha.txt"

if [ -z "${PRETRAIN_CKPT}" ]; then
    echo "PRETRAIN_CKPT is required."
    echo "Usage:"
    echo "  PRETRAIN_CKPT=/path/to/base.ckpt bash $0"
    exit 1
fi

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "Checkpoint does not exist: ${PRETRAIN_CKPT}"
    exit 1
fi
echo "Found checkpoint: ${PRETRAIN_CKPT}"
echo ""

################################################################################
# Experiment layout
################################################################################

source "${RUNS_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${PRETRAIN_CKPT}"
exp_init_sft "$0"
EXP_STAGE_DIR="$(dirname "$(dirname "${EXP_LOG_FILE}")")"
export RUN_NAME="${EXP_ID}-sft"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

exp_save_hparams "${EXP_STAGE_DIR}" \
    "model_size=${MODEL_SIZE}" \
    "context_length=${CONTEXT_LENGTH}" \
    "peak_lr=${PEAK_LR}" \
    "sft_muon_lr=${SFT_MUON_LR}" \
    "sft_embedding_lr=${SFT_EMBEDDING_LR}" \
    "sft_vocab_to_embed_lr=${SFT_VOCAB_TO_EMBED_LR}" \
    "sft_scalar_lr=${SFT_SCALAR_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "device_batch_size=${DEVICE_BATCH_SIZE}" \
    "grad_accum=${GRAD_ACCUM}" \
    "num_gpus=${NUM_GPUS}" \
    "max_steps=${MAX_STEPS}" \
    "mcmc_step_size=${MCMC_STEP_SIZE}" \
    "mcmc_num_steps=${MCMC_NUM_STEPS}" \
    "truncate_mcmc=${TRUNCATE_MCMC}" \
    "no_mcmc_detach=${NO_MCMC_DETACH}" \
    "pretrain_ckpt=${PRETRAIN_CKPT}" \
    "optimizer=muon_adamw"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# Display config
################################################################################

echo "=================================="
echo "  EBT d26 S2 fixed-alpha397 SFT"
echo "=================================="
echo "NOVA_HOME:     ${NOVA_HOME}"
echo "Train script:  ${TRAIN_SCRIPT}"
echo "Pretrain ckpt: ${PRETRAIN_CKPT}"
echo "Dataset:       nanochat_sft"
echo "Peak LR:       ${PEAK_LR}"
echo "Max steps:     ${MAX_STEPS}"
echo "Log file:      ${LOG_FILE}"
echo ""
read -p "Press Enter to start training, or Ctrl+C to cancel..."

################################################################################
# Log header
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 S2 fixed-alpha397 SFT training log
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Pretrain From:   ${PRETRAIN_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
################################################################################

LOG_HEADER

{
echo "================================================================================"
echo "[SYSTEM INFO]"
echo "================================================================================"
echo "Hostname:         $(hostname)"
echo "User:             $(whoami)"
echo "Python:           $(python3 --version 2>&1)"
echo "PyTorch:          $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA Available:   $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU Count:        $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""
echo "================================================================================"
echo "[SFT CONFIG]"
echo "================================================================================"
echo "Pretrain Checkpoint: ${PRETRAIN_CKPT}"
echo "Dataset:             nanochat_sft"
echo "Peak LR:             ${PEAK_LR}"
echo "Max Steps:           ${MAX_STEPS}"
echo "MCMC Step Size:      ${MCMC_STEP_SIZE}"
echo "MCMC Steps:          ${MCMC_NUM_STEPS}"
echo "Truncate MCMC:       ${TRUNCATE_MCMC}"
echo "No MCMC Detach:      ${NO_MCMC_DETACH}"
echo ""
echo "================================================================================"
echo "[START TRAINING]"
echo "================================================================================"
echo ""

set +e

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${TRAIN_SCRIPT}" \
--run_name "${RUN_NAME}_${current_time}" \
--checkpoint_dir "${EXP_CKPT_DIR}" \
--wandb_save_dir "${EXP_WANDB_DIR}" \
--modality "NLP" \
--model_name "${MODEL_NAME}" \
--model_size "${MODEL_SIZE}" \
--pretokenize_dataset \
--normalize_initial_condition \
--ebt_type "${EBT_TYPE}" \
--denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
--mcmc_step_size "${MCMC_STEP_SIZE}" \
--mcmc_num_steps "${MCMC_NUM_STEPS}" \
--truncate_mcmc \
--context_length "${CONTEXT_LENGTH}" \
--gpus "-1" \
--peak_learning_rate "${PEAK_LR}" \
--batch_size_per_device "${DEVICE_BATCH_SIZE}" \
--accumulate_grad_batches "${GRAD_ACCUM}" \
--gradient_clip_val "${GRADIENT_CLIP_VAL}" \
--weight_decay "${WEIGHT_DECAY}" \
--beta1 "${BETA1}" \
--beta2 "${BETA2}" \
--max_steps "${MAX_STEPS}" \
--max_scheduling_steps "${MAX_SCHEDULING_STEPS}" \
--execution_mode "finetune" \
--dataset_name "nanochat_sft" \
--val_check_interval "${VAL_CHECK_INTERVAL}" \
--limit_val_batches "${LIMIT_VAL_BATCHES}" \
--val_sanity 1 \
--validation_split_pct 0.0027 \
--wandb_project 'nlp_sft' \
--log_model_archi \
--set_matmul_precision "medium" \
--save_top_k_ckpts "${SAVE_TOP_K}" \
--save_periodic_steps "${VAL_CHECK_INTERVAL}" \
--float_precision "bf16-mixed" \
--finetuning_model_ckpt "${PRETRAIN_CKPT}" \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_STAGE_DIR}" "$(basename "${EXP_STAGE_DIR}")" "${TRAIN_EXIT_CODE}"

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "SFT training completed successfully"
else
    echo "SFT training failed (exit code: ${TRAIN_EXIT_CODE})"
fi

exit "${TRAIN_EXIT_CODE}"
} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"
PIPE_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Log saved to: ${LOG_FILE}"
exit "${PIPE_EXIT_CODE}"
