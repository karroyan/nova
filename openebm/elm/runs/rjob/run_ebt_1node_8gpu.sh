#!/bin/bash

################################################################################
# EBT d26 训练脚本 - 1节点 × 8卡
#
# 用法:
#   通过 rjob 提交 (见配套的 rjob_ebt_1node_8gpu.sh)
#   本地调试: NODE_RANK=0 NODE_COUNT=1 MASTER_ADDR=127.0.0.1 PROC_PER_NODE=8 bash run_ebt_1node_8gpu.sh
#
################################################################################

# Conda 环境激活（仅远程集群需要，本地调试可跳过）
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
    export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
fi

### 路径配置 ###
# 自动推导 NOVA_HOME：基于本脚本所在位置向上 4 级到 nova 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="${NOVA_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
TRAIN_SCRIPT="${NOVA_HOME}/openebm/elm/train.py"

NANOCHAT_HOME="/mnt/shared-storage-user/luyudong/nanochat"

# 进入工作目录
cd "${NOVA_HOME}"

# 确保 openebm 包可被 Python 导入
export PYTHONPATH="${NOVA_HOME}:${PYTHONPATH}"

### 基础配置 ###
# 设定当前实验名称
RUN_PREFIX="1node-8gpu-bf16mixed"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="${NANOCHAT_HOME}"
export NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# WandB 配置（offline 模式可以不提供 api key，同步时再提供即可）
export WANDB_MODE="offline"

################################################################################
# NCCL 配置
# 每节点 GPU 数 < 8 时，必须通过 nccl_auto_config.py 自动配置 NCCL 参数
# 请勿手动覆盖 NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX /
# CUDA_VISIBLE_DEVICES，这些由平台注入或 nccl_auto_config.py 自动设置
################################################################################

eval $(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=20

################################################################################
# 多机节点与 GPU 配置
#
#   以下变量由平台通过 -e DISTRIBUTED_JOB=true 自动注入:
#     NODE_RANK / NODE_COUNT / MASTER_ADDR / PROC_PER_NODE / JOB_ID
#
#   rendezvous 方式: c10d (--rdzv_backend=c10d)
#     - 使用 JOB_ID 作为 --rdzv_id，确保同 job 各节点汇聚到同一 rendezvous
################################################################################

NODE_COUNT=${NODE_COUNT:-2}
PROC_PER_NODE=${PROC_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
JOB_ID=${JOB_ID:-$$}
MASTER_PORT=$(( 20000 + 0x$(echo "${JOB_ID}" | md5sum | head -c 4) % 10000 ))

NUM_NODES=${NODE_COUNT}
GPUS_PER_NODE=${PROC_PER_NODE}
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
WORLD_SIZE=${NUM_GPUS}

echo "=== 分布式配置 ==="
echo "NODE_RANK:      ${NODE_RANK}"
echo "NODE_COUNT:     ${NODE_COUNT}"
echo "MASTER_ADDR:    ${MASTER_ADDR}"
echo "MASTER_PORT:    ${MASTER_PORT}"
echo "PROC_PER_NODE:  ${PROC_PER_NODE}"
echo "WORLD_SIZE:     ${WORLD_SIZE}"

# EBT 核心超参数
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"               # 官方建议: can almost always stay
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置与训练步数自动计算
################################################################################

DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=2048

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

TARGET_TOTAL_TOKENS=7340032000
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置
################################################################################

PEAK_LR=0.00025
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置 (Muon+AdamW 模式)
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
SAVE_TOP_K=2

################################################################################
# 优化选项
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.05 \
--optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling \
--muon_momentum_warmup_steps 0"

################################################################################
# torch.compile 配置
################################################################################

COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

################################################################################
# WandB 训练标志
################################################################################

WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"

################################################################################
# 日志配置
################################################################################

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
CONFIG_TAG="${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}_${NUM_NODES}nodes_${GPUS_PER_NODE}gpus"
export RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}"

LOG_DIR="${NOVA_HOME}/logs_base_train/${DATE_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

################################################################################
# 启动训练
################################################################################

echo "=== 启动训练 ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "LOG_FILE: ${LOG_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

set +e

torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${JOB_ID}" \
  "${TRAIN_SCRIPT}" \
  --run_name "${RUN_NAME}" \
  --modality "NLP" \
  --model_name "${MODEL_NAME}" \
  --model_size "${MODEL_SIZE}" \
  \
  --pretokenize_dataset \
  \
  --normalize_initial_condition \
  --ebt_type "${EBT_TYPE}" \
  --denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
  --mcmc_step_size_learnable \
  --mcmc_step_size "${MCMC_STEP_SIZE}" \
  --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
  --mcmc_num_steps "${MCMC_NUM_STEPS}" \
  \
  --context_length "${CONTEXT_LENGTH}" \
  \
  --gpus "-1" \
  \
  --peak_learning_rate "${PEAK_LR}" \
  --batch_size_per_device "${DEVICE_BATCH_SIZE}" \
  --accumulate_grad_batches "${GRAD_ACCUM}" \
  --gradient_clip_val "${GRADIENT_CLIP_VAL}" \
  \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --min_lr_scale "${MIN_LR_SCALE}" \
  --max_steps "${MAX_STEPS}" \
  --max_scheduling_steps "${MAX_SCHEDULING_STEPS}" \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --warm_up_base_lr_divider "${WARM_UP_BASE_LR_DIVIDER}" \
  \
  --dataset_name "nanochat" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_sanity 1 \
  --validation_split_pct 0.0027 \
  \
  --wandb_project 'nlp_pretrain' \
  --log_model_archi \
  --set_matmul_precision "medium" \
  --float_precision "bf16-mixed" \
  --manual_gc_collect_every_n_steps -1 \
  --save_top_k_ckpts "${SAVE_TOP_K}" \
  --save_periodic_steps 1000 \
  ${WANDB_FLAGS} \
  ${OPTION_FLAGS} \
  ${COMPILE_FLAGS}
  
# --use_ve \

TRAIN_EXIT_CODE=$?
set -e

################################################################################
# 训练结束处理
################################################################################

echo ""
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✓ 训练成功完成 (rank ${NODE_RANK})"
else
    echo "✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE, rank: ${NODE_RANK})"
fi

echo "日志文件: ${LOG_FILE}"
