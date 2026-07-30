#!/bin/bash

################################################################################
# EBT d26 SFT 训练脚本 (支持 Offline 模式)
# 基于预训练模型进行监督微调
################################################################################

set -e

### 基础配置 ###
# EXP_ID 必须指定，指向 base_train 的实验目录
# 例如: export EXP_ID="d26-ctx1024-muon_adamw-20260413-1235"
export EXP_ID="d26-ctx2048-20260426"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 预训练权重 ###
# base_train bpb 0.81
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-muon-adamw-0318_20260324_164538_2026-03-24_16-46-34_/last.ckpt"
# base_train bpb 0.81
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt"
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt"
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/e=epoch=0-s=step=55999-lr0.00025-bs4x8-muon_adamw-valid_loss=valid_loss=2.6877.ckpt"
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260407_001616/e=epoch=0-s=step=812-lr5e-05-bs4x16-muon_adamw-valid_loss=valid_loss=1.2609.ckpt"

# base_train c1024 bpb 0.78
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-ctx2048-muon-adamw-0413_20260413_123504/s=step=27999-d26-ctx1024-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.6294.ckpt"
# TODO
# base_train c2048 bpb 0.xx 
# PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep9562_20260424_211606/s=step=10937-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7801.ckpt"

# base_train c2048 bpb 0.86
PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep9562_20260424_211606/s=step=12374-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7007.ckpt"

### 环境变量 (对齐 resume_ebt_muon_adamw.sh + Offline 支持) ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_SFT_DATA_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

# Offline 模式环境变量 (必须在 torchrun 前设置)
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# mkdir -p logs/slurm/nlp/
# module purge

################################################################################
# EBT 核心超参数 (对齐预训练配置)
################################################################################
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=750
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置 (自动检测 GPU 数量)
################################################################################
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

CONTEXT_LENGTH=1024  # 受限于 base_train 阶段的上下文长度
# DEVICE_BATCH_SIZE=2

CONTEXT_LENGTH=2048  # 受限于 base_train 阶段的上下文长度
DEVICE_BATCH_SIZE=1

GRAD_ACCUM=32



################################################################################
# SFT 学习率配置
################################################################################
# SFT 使用较低学习率，防止灾难性遗忘
# 预训练 PEAK_LR=0.00025，SFT 降 5x
PEAK_LR=0.00005
# 注意: 使用 --linear_warmdown 调度时 --warm_up_steps / --warm_up_base_lr_divider / --min_lr_scale
# 均不生效 (WarmUpLinearWarmdownLR 只接受 warmup_ratio)。warmup 由 --warmup_ratio 控制。
#
# 重要: --muon_lr / --adamw_*_lr 是绝对值，不受 PEAK_LR 缩放！
# 预训练值: muon_lr=0.02, embedding_lr=0.3, vocab_to_embed_lr=0.01, scalar_lr=0.04
# SFT 需要同比例降低 (÷10)，否则会导致梯度爆炸 → loss NaN
SFT_MUON_LR=0.002           # 预训练 0.02 ÷ 10
SFT_EMBEDDING_LR=0.03       # 预训练 0.3 ÷ 10
SFT_VOCAB_TO_EMBED_LR=0.001 # 预训练 0.01 ÷ 10
SFT_SCALAR_LR=0.004         # 预训练 0.04 ÷ 10

################################################################################
# 优化器配置 (对齐预训练)
################################################################################
WEIGHT_DECAY=0.0
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# SFT 训练步数
################################################################################
# SFT 数据 ~856K 条，根据实际需求调整

MAX_STEPS=3000
MAX_SCHEDULING_STEPS=3000

################################################################################
# 验证与数据加载配置
################################################################################
VAL_CHECK_INTERVAL=500
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8

################################################################################
# 优化选项配置 (SFT: 所有绝对 LR 同比降低)
################################################################################
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.05 --warmdown_ratio 0.2 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr ${SFT_MUON_LR} --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr ${SFT_EMBEDDING_LR} --adamw_vocab_to_embed_lr ${SFT_VOCAB_TO_EMBED_LR} --adamw_scalar_lr ${SFT_SCALAR_LR} --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置 (对齐预训练)
################################################################################
COMPILE_FLAGS="--compile_model --compile_mode full"
WANDB_FLAGS=""

################################################################################
# Checkpoint 管理配置
################################################################################
SAVE_TOP_K=2

################################################################################
# 数据集检查 (Offline 模式)
################################################################################
echo ""
echo "=================================="
echo "  检查 Offline 数据集"
echo "=================================="
echo "数据集路径: ${NANOCHAT_SFT_DATA_DIR}"

MISSING_DATASETS=()

# 检查必需的数据集
[ ! -d "${NANOCHAT_SFT_DATA_DIR}/smol-smoltalk" ] && MISSING_DATASETS+=("smol-smoltalk")
[ ! -d "${NANOCHAT_SFT_DATA_DIR}/mmlu" ] && MISSING_DATASETS+=("mmlu")
[ ! -d "${NANOCHAT_SFT_DATA_DIR}/gsm8k" ] && MISSING_DATASETS+=("gsm8k")
[ ! -f "${NANOCHAT_SFT_DATA_DIR}/words_alpha.txt" ] && MISSING_DATASETS+=("words_alpha.txt")

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo "✗ 缺少以下离线数据集:"
    for dataset in "${MISSING_DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "请先在联网环境下载数据集:"
    echo "  cd /mnt/shared-storage-user/puyuan/code/nanochat"
    echo "  python runs/download_sft_datasets.py"
    echo ""
    echo "或设置环境变量指向已下载的数据集:"
    echo "  export NANOCHAT_SFT_DATA_DIR=/path/to/sft_data"
    exit 1
fi

echo "✓ SmolTalk: ${NANOCHAT_SFT_DATA_DIR}/smol-smoltalk"
echo "✓ MMLU: ${NANOCHAT_SFT_DATA_DIR}/mmlu"
echo "✓ GSM8K: ${NANOCHAT_SFT_DATA_DIR}/gsm8k"
echo "✓ words_alpha.txt: ${NANOCHAT_SFT_DATA_DIR}/words_alpha.txt"

# 检查 checkpoint
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "✗ Checkpoint 不存在: ${PRETRAIN_CKPT}"
    exit 1
fi
echo "✓ Checkpoint: ${PRETRAIN_CKPT}"
echo ""

################################################################################
# 实验目录布局 - 统一输出到 ebt_runs/<exp_id>/sft_train/
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${PRETRAIN_CKPT}"
exp_init_sft "$0"
export RUN_NAME="${EXP_ID}-sft"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

exp_save_hparams "${EXP_DIR}/sft_train" \
    "model_size=${MODEL_SIZE}" \
    "context_length=${CONTEXT_LENGTH}" \
    "peak_lr=${PEAK_LR}" \
    "sft_muon_lr=${SFT_MUON_LR}" \
    "sft_embedding_lr=${SFT_EMBEDDING_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "device_batch_size=${DEVICE_BATCH_SIZE}" \
    "grad_accum=${GRAD_ACCUM}" \
    "num_gpus=${NUM_GPUS}" \
    "max_steps=${MAX_STEPS}" \
    "mcmc_step_size=${MCMC_STEP_SIZE}" \
    "mcmc_lr_multiplier=${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
    "pretrain_ckpt=${PRETRAIN_CKPT}" \
    "optimizer=muon_adamw"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  EBT d26 SFT 训练"
echo "=================================="
echo "Pretrain ckpt: ${PRETRAIN_CKPT}"
echo "Dataset:       nanochat_sft"
echo "Peak LR:       ${PEAK_LR}"
echo "Max steps:     ${MAX_STEPS}"
echo "Log file:      ${LOG_FILE}"
echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 SFT 训练日志
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
echo "[SFT 训练配置]"
echo "================================================================================"
echo "Pretrain Checkpoint: ${PRETRAIN_CKPT}"
echo "Dataset:             nanochat_sft"
echo "Peak LR:             ${PEAK_LR}"
echo "Max Steps:           ${MAX_STEPS}"
echo ""
echo "================================================================================"
echo "[开始训练]"
echo "================================================================================"
echo ""

set -e

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
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--dataset_name "nanochat_sft" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
--wandb_project 'nlp_sft' \
--log_model_archi \
--set_matmul_precision "medium" \
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps ${VAL_CHECK_INTERVAL} \
--float_precision "bf16-mixed" \
--finetuning_model_ckpt ${PRETRAIN_CKPT} \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_DIR}/sft_train" "sft_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ SFT 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ SFT 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
