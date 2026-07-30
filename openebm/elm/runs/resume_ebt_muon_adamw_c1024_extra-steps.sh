#!/bin/bash

################################################################################
# EBT d26 恢复训练脚本 - 从 checkpoint 继续训练
# 基于: run_ebt_muon_adamw_0324.sh
# 新增: checkpoint 恢复 + 磁盘空间管理
################################################################################

### 基础配置 ###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"
export RUN_NAME="ebt-d26-ctx1024-from0419"

export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="d26"

### 恢复训练配置 ###

# context 1024
RESUME_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-ctx2048-muon-adamw-0413_20260413_123504/s=step=27937-d26-ctx1024-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.6296.ckpt"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# EBT 核心超参数
################################################################################
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置
################################################################################
# NUM_GPUS=8
# DEVICE_BATCH_SIZE=4
# GRAD_ACCUM=8
# CONTEXT_LENGTH=512

# NUM_GPUS=8
# DEVICE_BATCH_SIZE=2
# GRAD_ACCUM=16
# CONTEXT_LENGTH=1024

NUM_GPUS=8
DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=1024

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
TARGET_TOTAL_TOKENS=7340032000
# Override for resume: total 56000 steps (28000 original + 28000 continuation)
MAX_STEPS=56000
MAX_SCHEDULING_STEPS=56000

# TODO =============
# MAX_STEPS=559990
# MAX_SCHEDULING_STEPS=559990

################################################################################
# 学习率配置
################################################################################
PEAK_LR=0.000125
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置
################################################################################
WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################
VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8

################################################################################
# 优化选项配置
################################################################################
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.25 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.01 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.15 --adamw_vocab_to_embed_lr 0.005 --adamw_scalar_lr 0.02 --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################
# 重要：必须与原始训练使用相同的 compile 模式，否则 state_dict key 不匹配
# 原始训练使用 full 模式，恢复时也必须使用 full
COMPILE_FLAGS="--compile_model --compile_mode full"
WANDB_FLAGS=""

################################################################################
# Checkpoint 管理配置 (减少磁盘占用)
################################################################################
SAVE_TOP_K=2  # 只保留最好的 2 个 checkpoint (原来是 10)

################################################################################
# 日志配置
################################################################################
current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_${MODEL_NAME}_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_muon-adamw_gpu${NUM_GPUS}_resume.log"
mkdir -p logs

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  恢复训练配置"
echo "=================================="
echo "Resume from: ${RESUME_CKPT}"
echo "Save top K:  ${SAVE_TOP_K} (减少磁盘占用)"
echo "Log file:    ${LOG_FILE}"
echo ""
read -p "按 Enter 开始恢复训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 Muon+AdamW 恢复训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Resume From:     ${RESUME_CKPT}
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
echo "[恢复训练配置]"
echo "================================================================================"
echo "Resume Checkpoint: ${RESUME_CKPT}"
echo "Save Top K:        ${SAVE_TOP_K}"
echo ""
echo "================================================================================"
echo "[开始训练]"
echo "================================================================================"
echo ""

set +e

torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/train.py \
--run_name ${RUN_NAME}_${current_time} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
--pretokenize_dataset \
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
--denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps ${MCMC_NUM_STEPS} \
--context_length ${CONTEXT_LENGTH} \
--gpus "-1" \
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
--weight_decay ${WEIGHT_DECAY} \
--beta1 ${BETA1} \
--beta2 ${BETA2} \
--min_lr_scale ${MIN_LR_SCALE} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--warm_up_steps ${WARM_UP_STEPS} \
--warm_up_base_lr_divider ${WARM_UP_BASE_LR_DIVIDER} \
--dataset_name "nanochat" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
--float_precision bf16-mixed \
--manual_gc_collect_every_n_steps -1 \
--save_top_k_ckpts ${SAVE_TOP_K} \
--resume_training_ckpt ${RESUME_CKPT} \
--resume_warmup_steps 2000 \
--save_periodic_steps 1000 \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
