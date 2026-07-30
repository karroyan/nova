#!/bin/bash

################################################################################
# EBT d26 free-embedding direct-unembed Sudoku SFT - 2 nodes x 8 GPUs
#
# This script combines:
#   - the multi-node torchrun/rjob runtime from run_ebt_freeembed_2node_8gpu.sh
#   - the Sudoku mixed SFT v3 data and optimizer schedule
#   - the free-embedding direct-unembed architecture used by the source checkpoint
################################################################################

set -e

OPENEBM_HOME_DEFAULT="/mnt/shared-storage-user/puyuan/code/OpenEBM"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV_PATH}/bin/torchrun}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    if ! conda activate "${CONDA_ENV_PATH}"; then
        echo "Warning: conda activate failed for ${CONDA_ENV_PATH}; continuing with explicit python/torchrun paths." >&2
    fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
    exit 1
fi

export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="${NOVA_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
TRAIN_SCRIPT="${NOVA_HOME}/openebm/elm/train.py"
EXP_LAYOUT_SCRIPT="${NOVA_HOME}/openebm/elm/runs/utils/exp_layout.sh"

cd "${NOVA_HOME}"

export PYTHONPATH="${NOVA_HOME}/nanochat:${NOVA_HOME}:${PYTHONPATH:-}"
export HOME="${NOVA_HOME}"
export NANOCHAT_BASE_DIR="${NOVA_HOME}/data"
export NANOCHAT_SFT_DATA_DIR="${NOVA_HOME}/data/sft_data"
export HF_HOME="${NOVA_HOME}/data/hf_home"
export HF_DATASETS_CACHE="${NOVA_HOME}/data/hf_datasets_cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${MPLCONFIGDIR}"

# GPU workers are offline. All datasets/checkpoints must already be visible on
# shared storage before this script starts.
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

################################################################################
# Distributed runtime
################################################################################

NODE_COUNT=${NODE_COUNT:-2}
PROC_PER_NODE=${PROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
JOB_ID=${JOB_ID:-$$}
MASTER_PORT=$(( 20000 + 0x$(echo "${JOB_ID}" | md5sum | head -c 4) % 10000 ))

NUM_NODES=${NODE_COUNT}
GPUS_PER_NODE=${PROC_PER_NODE}
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
WORLD_SIZE=${NUM_GPUS}

################################################################################
# Model/checkpoint
################################################################################

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-/mnt/shared-storage-user/luyudong/nova-sft/nova/logs/ebt_runs/d26-ctx2048-20260605-s2-step1-learnablealpha-init300-freeembed-direct/sft_train.v4/checkpoints/s=step=1687-d26-ctx2048-lr0.00024-bs1x32-muon_adamw-valid_loss=valid_loss=0.6590.ckpt}"

################################################################################
# Free-embedding direct-unembed architecture
################################################################################

MCMC_STEP_SIZE="${MCMC_STEP_SIZE:-300.0}"
MCMC_STEP_SIZE_LR_MULTIPLIER="${MCMC_STEP_SIZE_LR_MULTIPLIER:-1500}"
MCMC_NUM_STEPS="${MCMC_NUM_STEPS:-1}"
RANDOMIZE_MCMC_NUM_STEPS="${RANDOMIZE_MCMC_NUM_STEPS:-0}"
EBT_TYPE="${EBT_TYPE:-time_embed}"
DENOISING_INITIAL_CONDITION="${DENOISING_INITIAL_CONDITION:-random_noise}"

TF_HEAD_TYPE="${TF_HEAD_TYPE:-direct_unembed}"
TF_HEAD_LAYERS="${TF_HEAD_LAYERS:-1}"
TF_HEAD_N_HEADS="${TF_HEAD_N_HEADS:-0}"
TF_HEAD_FFN_MULT="${TF_HEAD_FFN_MULT:-4.0}"
FREE_EMBED_NOISE_SCALE="${FREE_EMBED_NOISE_SCALE:-1.0}"
USE_SDPA_ATTENTION="${USE_SDPA_ATTENTION:-false}"

################################################################################
# Sudoku mixed SFT v3 data and training hyperparameters
################################################################################

SUDOKU_DATA_DIR_V2="${SUDOKU_DATA_DIR_V2:-${NOVA_HOME}/openebm/elm/data/sudoku_cache_v2}"
SUDOKU_RATIO="${SUDOKU_RATIO:-0.6}"
SUDOKU_RATIO_SCHEDULE="${SUDOKU_RATIO_SCHEDULE:-three_phase_with_guard}"
SUDOKU_DIFFICULTY_SCHEDULE="${SUDOKU_DIFFICULTY_SCHEDULE:-three_phase}"
SUDOKU_BLANK_LOSS_WEIGHT="${SUDOKU_BLANK_LOSS_WEIGHT:-2.0}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"

PEAK_LR="${PEAK_LR:-0.00005}"
SFT_MUON_LR="${SFT_MUON_LR:-0.002}"
SFT_EMBEDDING_LR="${SFT_EMBEDDING_LR:-0.03}"
SFT_VOCAB_TO_EMBED_LR="${SFT_VOCAB_TO_EMBED_LR:-0.001}"
SFT_SCALAR_LR="${SFT_SCALAR_LR:-0.004}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA1="${BETA1:-0.8}"
BETA2="${BETA2:-0.95}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"

MAX_STEPS="${MAX_STEPS:-3000}"
MAX_SCHEDULING_STEPS="${MAX_SCHEDULING_STEPS:-3000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-25}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.05 --warmdown_ratio 0.2 --final_lr_frac 0.0 \
--optimizer muon_adamw --muon_lr ${SFT_MUON_LR} --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr ${SFT_EMBEDDING_LR} --adamw_vocab_to_embed_lr ${SFT_VOCAB_TO_EMBED_LR} --adamw_scalar_lr ${SFT_SCALAR_LR} \
--adamw_dmodel_lr_scaling"

COMPILE_FLAGS="${COMPILE_FLAGS:-"--compile_model --compile_mode transformer_only"}"
WANDB_FLAGS="${WANDB_FLAGS:-}"
SDPA_FLAGS=""
if [[ "${USE_SDPA_ATTENTION}" == "true" ]]; then
    SDPA_FLAGS="--use_sdpa_attention"
fi

################################################################################
# Experiment directories
################################################################################

source "${EXP_LAYOUT_SCRIPT}"

RUN_PREFIX="${RUN_PREFIX:-d26-ctx2048-sudoku-sft-freeembed-direct-2node-8gpu}"
export EXP_ID="${EXP_ID:-${RUN_PREFIX}-${JOB_ID}}"
export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"
STAGE_DIR="${EXP_DIR}/sft_train"
export EXP_CKPT_DIR="${STAGE_DIR}/checkpoints"
export EXP_WANDB_DIR="${STAGE_DIR}/logs"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "${STAGE_DIR}/config" "${EXP_CKPT_DIR}" "${EXP_WANDB_DIR}"
cp "$0" "${STAGE_DIR}/config/run_script.sh" 2>/dev/null || true
_exp_write_meta "${EXP_DIR}" "sft_train"
_exp_git_info "${STAGE_DIR}/config/git_info.txt"

LOG_FILE="${EXP_WANDB_DIR}/train_rank${NODE_RANK}.log"
export RUN_NAME="${EXP_ID}-rank${NODE_RANK}"

if [[ "${NODE_RANK}" == "0" ]]; then
    exp_save_hparams "${STAGE_DIR}" \
        "model_size=${MODEL_SIZE}" \
        "context_length=${CONTEXT_LENGTH}" \
        "peak_lr=${PEAK_LR}" \
        "sft_muon_lr=${SFT_MUON_LR}" \
        "sft_embedding_lr=${SFT_EMBEDDING_LR}" \
        "weight_decay=${WEIGHT_DECAY}" \
        "device_batch_size=${DEVICE_BATCH_SIZE}" \
        "grad_accum=${GRAD_ACCUM}" \
        "num_nodes=${NUM_NODES}" \
        "gpus_per_node=${GPUS_PER_NODE}" \
        "max_steps=${MAX_STEPS}" \
        "mcmc_step_size=${MCMC_STEP_SIZE}" \
        "mcmc_lr_multiplier=${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
        "mcmc_num_steps=${MCMC_NUM_STEPS}" \
        "randomize_mcmc_num_steps=${RANDOMIZE_MCMC_NUM_STEPS}" \
        "tf_head_type=${TF_HEAD_TYPE}" \
        "free_embedding_mcmc=true" \
        "pretrain_ckpt=${PRETRAIN_CKPT}" \
        "dataset=sudoku_mixed_v3" \
        "sudoku_ratio_initial=${SUDOKU_RATIO}" \
        "sudoku_ratio_schedule=${SUDOKU_RATIO_SCHEDULE}" \
        "sudoku_difficulty_schedule=${SUDOKU_DIFFICULTY_SCHEDULE}" \
        "sudoku_blank_loss_weight=${SUDOKU_BLANK_LOSS_WEIGHT}" \
        "save_top_k=${SAVE_TOP_K}"
fi

exec > >(awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}") 2>&1

################################################################################
# Preflight checks
################################################################################

echo "=== Distributed config ==="
echo "NODE_RANK:      ${NODE_RANK}"
echo "NODE_COUNT:     ${NODE_COUNT}"
echo "MASTER_ADDR:    ${MASTER_ADDR}"
echo "MASTER_PORT:    ${MASTER_PORT}"
echo "PROC_PER_NODE:  ${PROC_PER_NODE}"
echo "WORLD_SIZE:     ${WORLD_SIZE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA:-}"
echo "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}"

echo ""
echo "=== Paths ==="
echo "NOVA_HOME:             ${NOVA_HOME}"
echo "TRAIN_SCRIPT:          ${TRAIN_SCRIPT}"
echo "NANOCHAT_BASE_DIR:     ${NANOCHAT_BASE_DIR}"
echo "NANOCHAT_SFT_DATA_DIR: ${NANOCHAT_SFT_DATA_DIR}"
echo "SUDOKU_DATA_DIR_V2:    ${SUDOKU_DATA_DIR_V2}"
echo "PRETRAIN_CKPT:         ${PRETRAIN_CKPT}"
echo "CONDA_ENV_PATH:        ${CONDA_ENV_PATH}"
echo "PYTHON_BIN:            ${PYTHON_BIN}"
echo "TORCHRUN_BIN:          ${TORCHRUN_BIN}"
echo "EXP_DIR:               ${EXP_DIR}"
echo "EXP_CKPT_DIR:          ${EXP_CKPT_DIR}"
echo "LOG_FILE:              ${LOG_FILE}"

MISSING=()
[[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_train.pt" ]] && MISSING+=("rrn_train.pt")
[[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_val.pt" ]] && MISSING+=("rrn_val.pt")
[[ ! -f "${SUDOKU_DATA_DIR_V2}/satnet_test.pt" ]] && MISSING+=("satnet_test.pt")
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Missing Sudoku V2 cache files in ${SUDOKU_DATA_DIR_V2}: ${MISSING[*]}" >&2
    exit 1
fi
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Checkpoint not found: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY_PREFLIGHT'
import sys
import torch
import wandb

try:
    import lightning as pl
    pl_name = "lightning"
except ImportError:
    import pytorch_lightning as pl
    pl_name = "pytorch_lightning"

import openebm.elm.train

print(
    f"Python preflight ok: {sys.executable} | "
    f"torch={torch.__version__} | {pl_name}={pl.__version__}"
)
PY_PREFLIGHT

echo ""
echo "=== Training config ==="
echo "Dataset:                    sudoku_mixed"
echo "Sudoku ratio initial:       ${SUDOKU_RATIO}"
echo "Sudoku ratio schedule:      ${SUDOKU_RATIO_SCHEDULE}"
echo "Difficulty schedule:        ${SUDOKU_DIFFICULTY_SCHEDULE}"
echo "Blank loss weight:          ${SUDOKU_BLANK_LOSS_WEIGHT}"
echo "MCMC steps:                 ${MCMC_NUM_STEPS}"
echo "MCMC step size init:        ${MCMC_STEP_SIZE}"
echo "MCMC step size lr mult:     ${MCMC_STEP_SIZE_LR_MULTIPLIER}"
echo "TF head type:               ${TF_HEAD_TYPE}"
echo "Free embed noise scale:     ${FREE_EMBED_NOISE_SCALE}"
echo "Peak LR:                    ${PEAK_LR}"
echo "Max steps:                  ${MAX_STEPS}"
echo "Val interval:               ${VAL_CHECK_INTERVAL}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-60}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-20}"

set +e

"${TORCHRUN_BIN}" \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${JOB_ID}" \
  "${TRAIN_SCRIPT}" \
  --run_name "${RUN_NAME}" \
  --checkpoint_dir "${EXP_CKPT_DIR}" \
  --wandb_save_dir "${EXP_WANDB_DIR}" \
  --modality "NLP" \
  --model_name "${MODEL_NAME}" \
  --model_size "${MODEL_SIZE}" \
  --truncate_mcmc \
  --pretokenize_dataset \
  --normalize_initial_condition \
  --ebt_type "${EBT_TYPE}" \
  --denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
  --mcmc_step_size_learnable \
  --mcmc_step_size "${MCMC_STEP_SIZE}" \
  --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
  --mcmc_num_steps "${MCMC_NUM_STEPS}" \
  --randomize_mcmc_num_steps "${RANDOMIZE_MCMC_NUM_STEPS}" \
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
  --dataset_name "sudoku_mixed" \
  --sudoku_ratio "${SUDOKU_RATIO}" \
  --sudoku_ratio_schedule "${SUDOKU_RATIO_SCHEDULE}" \
  --sudoku_difficulty_schedule "${SUDOKU_DIFFICULTY_SCHEDULE}" \
  --sudoku_blank_loss_weight "${SUDOKU_BLANK_LOSS_WEIGHT}" \
  --num_workers "${NUM_WORKERS}" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_sanity 1 \
  --validation_split_pct 0.1 \
  --wandb_project "nlp_sudoku_sft_freeembed" \
  --log_model_archi \
  --set_matmul_precision "medium" \
  --save_top_k_ckpts "${SAVE_TOP_K}" \
  --save_periodic_steps "${VAL_CHECK_INTERVAL}" \
  --float_precision "bf16-mixed" \
  --finetuning_model_ckpt "${PRETRAIN_CKPT}" \
  --manual_gc_collect_every_n_steps -1 \
  --use_tf_head \
  --tf_head_type "${TF_HEAD_TYPE}" \
  --tf_head_layers "${TF_HEAD_LAYERS}" \
  --tf_head_n_heads "${TF_HEAD_N_HEADS}" \
  --tf_head_ffn_mult "${TF_HEAD_FFN_MULT}" \
  --free_embedding_mcmc \
  --free_embed_noise_scale "${FREE_EMBED_NOISE_SCALE}" \
  ${WANDB_FLAGS} \
  ${OPTION_FLAGS} \
  ${COMPILE_FLAGS} \
  ${SDPA_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

if [[ "${NODE_RANK}" == "0" ]]; then
    exp_save_status "${STAGE_DIR}" "sudoku_sft_freeembed_train" "${TRAIN_EXIT_CODE}"
fi

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    echo "Training completed successfully on node rank ${NODE_RANK}"
else
    echo "Training failed with exit code ${TRAIN_EXIT_CODE} on node rank ${NODE_RANK}"
fi

echo "Log file: ${LOG_FILE}"
exit "${TRAIN_EXIT_CODE}"
