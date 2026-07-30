#!/bin/bash

################################################################################
# EBT d26 精确断点续训脚本 (Exact Resume)
# 从 step=27937 精确续训到 step=28000
# 训练曲线与不中断完全一致 (数据流 + RNG 精确恢复)
#
# 关键: 所有超参数与原始 run_ebt_muon_adamw_c1024.sh 完全一致
################################################################################

### 基础配置 ###
# EXP_ID 必须指定，指向原始 base_train 的实验目录
# 例如: export EXP_ID="d26-ctx1024-muon_adamw-20260413-1235"
export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 恢复训练配置 ###
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
# EBT 核心超参数 (与原始训练完全一致)
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
# Batch 配置 (与原始训练完全一致)
################################################################################
NUM_GPUS=8
DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=1024

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

# 与原始训练完全一致: 28000 步
MAX_STEPS=28000
MAX_SCHEDULING_STEPS=28000

################################################################################
# 学习率配置 (与原始训练完全一致)
################################################################################
PEAK_LR=0.00025
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置 (与原始训练完全一致)
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
SAVE_TOP_K=2

################################################################################
# 优化选项配置 (与原始训练完全一致: warmdown_ratio=0.5)
################################################################################
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置 (与原始训练一致)
################################################################################
COMPILE_FLAGS="--compile_model --compile_mode full"
WANDB_FLAGS=""

################################################################################
# 实验目录布局 - 在 base_train/ 下创建 resume_v{N}/ 子目录
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export RESUME_CKPT="${RESUME_CKPT}"
exp_init_resume "$0"
export RUN_NAME="${EXP_ID}-resume"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  精确断点续训 (Exact Resume)"
echo "=================================="
echo "Resume from:     ${RESUME_CKPT}"
echo "Target steps:    ${MAX_STEPS} (与原始训练一致)"
echo "Warmdown ratio:  0.5 (与原始训练一致)"
echo "resume_warmup:   0 (不做额外 warmup)"
echo "val_sanity:      0 (跳过，避免 RNG 干扰)"
echo "Log file:        ${LOG_FILE}"
echo ""
read -p "按 Enter 开始精确续训，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 精确断点续训日志 (Exact Resume)
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Resume From:     ${RESUME_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Target:          step 27937 → step 28000 (63 steps remaining)
#
# 关键: 所有超参数与原始训练完全一致，确保训练曲线连续
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
echo "[精确续训配置]"
echo "================================================================================"
echo "Resume Checkpoint: ${RESUME_CKPT}"
echo "Max Steps:         ${MAX_STEPS} (与原始训练一致)"
echo "Scheduling Steps:  ${MAX_SCHEDULING_STEPS}"
echo "Peak LR:           ${PEAK_LR}"
echo "Warmdown Ratio:    0.5 (与原始训练一致)"
echo "Resume Warmup:     0 (精确续训，不做 warmup)"
echo "Val Sanity:        0 (跳过，避免 RNG 干扰)"
echo ""
echo "================================================================================"
echo "[开始训练]"
echo "================================================================================"
echo ""

set +e

torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/train.py \
--run_name ${RUN_NAME}_${current_time} \
--checkpoint_dir "${EXP_CKPT_DIR}" \
--wandb_save_dir "${EXP_WANDB_DIR}" \
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
--val_sanity 0 \
--validation_split_pct 0.0027 \
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
--float_precision bf16-mixed \
--manual_gc_collect_every_n_steps -1 \
--save_top_k_ckpts ${SAVE_TOP_K} \
--resume_training_ckpt ${RESUME_CKPT} \
--resume_warmup_steps 0 \
--save_periodic_steps 100 \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${RESUME_DIR}" "resume" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ 精确续训成功完成\033[0m"
else
    echo -e "\033[0;31m✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
