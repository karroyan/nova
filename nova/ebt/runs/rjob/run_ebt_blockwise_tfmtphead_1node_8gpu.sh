#!/bin/bash

################################################################################
# EBT blockwise + TF-MTP-head (Gemma-drafter style) 训练脚本 - 1节点 × 8卡
#
# 默认配置:
#   * MODEL_SIZE=d26
#   * BLOCK_MODE=blockwise
#   * TRAIN_BLOCK_SIZE=2
#   * BLOCK_LATENT_HEAD_TYPE=per_offset_tf_transformer  (E4-b, "tfmtphead")
#   * BLOCK_LATENT_HEAD_LAYERS=1                        ("L1")
#   * CONTEXT_LENGTH=1024
#
# 所有 head 参数都是 env var, 可外部覆盖 (例如想试 L=2, per_offset_tf_linear, ...).
#
# 用法:
#   通过 rjob 提交 (见配套的 rjob_ebt_blockwise_tfmtphead_1node_8gpu.sh).
#   本地调试: NODE_RANK=0 NODE_COUNT=1 MASTER_ADDR=127.0.0.1 PROC_PER_NODE=8 \
#             bash run_ebt_blockwise_tfmtphead_1node_8gpu.sh
################################################################################

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/lixueyan/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/lixueyan/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
umask 022

### 路径配置 ###
# Points at the nova_dev tree (dev-blockwise / dev-blockwise-c1024 branch
# where the TF heads + bidirectional + E1-E4 code lives). Override
# NOVA_HOME if you want to point at another checkout.
NOVA_HOME="${NOVA_HOME:-/mnt/shared-storage-user/lixueyan/nar/nova_dev/nova}"
NANOCHAT_HOME="${NANOCHAT_HOME:-/mnt/shared-storage-user/lixueyan/nar/nanochat}"
TRAIN_SCRIPT="${NOVA_HOME}/ebt/train.py"

cd "${NOVA_HOME%/nova}"  # one level up so 'nova/ebt/train.py' is reachable

### 基础配置 ###
RUN_PREFIX="${RUN_PREFIX:-1node-8gpu-bf16mixed-blockwise-tfmtphead}"

export MODEL_NAME="${MODEL_NAME:-ebt}"
export MODEL_SIZE="${MODEL_SIZE:-d26}"

### Blockwise 配置 ###
BLOCK_MODE="${BLOCK_MODE:-blockwise}"
TRAINING_OBJECTIVE="${TRAINING_OBJECTIVE:-blockwise}"
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"
TRAIN_CONTEXT_LENGTH="${TRAIN_CONTEXT_LENGTH:-}"

case "${BLOCK_MODE}" in
  mtp_mcmc|future_latent_non_causal|blockwise|future_latent_bidirectional)
    ;;
  *)
    echo "ERROR: unknown BLOCK_MODE='${BLOCK_MODE}'" >&2
    echo "Allowed: mtp_mcmc | future_latent_non_causal | blockwise | future_latent_bidirectional" >&2
    exit 2
    ;;
esac

### Per-offset head 配置 (E2/E4) ###
# Default to the experimentally best combo at xxs: per_offset_tf_transformer
# (Gemma-drafter TF head) with 1 transformer layer ("tfmtphead L1"). Overrideable.
BLOCK_LATENT_HEAD_TYPE="${BLOCK_LATENT_HEAD_TYPE:-per_offset_tf_transformer}"
BLOCK_LATENT_HEAD_LAYERS="${BLOCK_LATENT_HEAD_LAYERS:-1}"
BLOCK_LATENT_HEAD_FFN_MULT="${BLOCK_LATENT_HEAD_FFN_MULT:-4.0}"
BLOCK_LATENT_HEAD_N_HEADS="${BLOCK_LATENT_HEAD_N_HEADS:-0}"

### Optional E1 / E3 knobs (default off / no-op) ###
OFFSET_LOSS_WEIGHTS="${OFFSET_LOSS_WEIGHTS:-}"
MCMC_CAUSAL_DETACH="${MCMC_CAUSAL_DETACH:-}"
MCMC_STAGGERED="${MCMC_STAGGERED:-}"

### 环境变量 ###
HOME="${NANOCHAT_HOME}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-offline}"

################################################################################
# NCCL 配置
################################################################################

eval $(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export)
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=20

################################################################################
# 分布式配置
################################################################################

NODE_COUNT=${NODE_COUNT:-1}
PROC_PER_NODE=${PROC_PER_NODE:-8}
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

################################################################################
# EBT / blockwise 核心超参数
################################################################################

MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500
MCMC_NUM_STEPS="${MCMC_NUM_STEPS:-2}"
EBT_TYPE="time_embed"
DENOISING_INITIAL_CONDITION="random_noise"

################################################################################
# Batch 配置与训练步数自动计算
################################################################################

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
# Blockwise at K=2 expands the trunk sequence to ~1 + S + S*K = 3073 tokens
# for S=1024. d26 + bf16-mixed + gradient_checkpointing should fit on H200/140G.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
TARGET_TOTAL_TOKENS="${TARGET_TOTAL_TOKENS:-7340032000}"
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率 / 优化器 / 验证
################################################################################

# Conservative LR + non-zero warmup vs origin's dense d26 recipe: the TF
# heads add K newly-initialised matrices that need gentle entry into the
# optimization basin. Tested working at PEAK_LR=0.0001 / WARM_UP_STEPS=1000.
PEAK_LR="${PEAK_LR:-0.0001}"
WARM_UP_STEPS="${WARM_UP_STEPS:-1000}"
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-2000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
SAVE_TOP_K="${SAVE_TOP_K:-2}"

# Default is AdamW. Muon+AdamW empirically diverges with TF heads at d26
# (loss climbs to ~500 within 60 steps): Muon's spectral-normalised update
# at lr=0.02 over-perturbs the freshly-initialised K head matrices. AdamW's
# adaptive moment update doesn't suffer from this. Set OPTIMIZER=muon_adamw
# to opt into Muon (requires param-group routing work to be safe with TF
# heads — not done yet).
OPTIMIZER="${OPTIMIZER:-adamw}"
case "${OPTIMIZER}" in
  muon_adamw)
    OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 \
--optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"
    ;;
  adamw)
    OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 \
--optimizer adamw"
    ;;
  *)
    echo "ERROR: unknown OPTIMIZER='${OPTIMIZER}', allowed: muon_adamw | adamw" >&2
    exit 2
    ;;
esac

# torch.compile is currently incompatible with EBT's double backward in
# training (create_graph=True); leave off until inference-only paths.
COMPILE_FLAGS=""

# Escape hatches if something looks numerically off: set
# GRADIENT_CHECKPOINTING=false (saves only marginal memory but rules out
# double-backward + GC interaction) or FLOAT_PRECISION=32-true (slow but
# definitive bf16 diagnostic).
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FLOAT_PRECISION="${FLOAT_PRECISION:-bf16-mixed}"

WANDB_FLAGS=""

################################################################################
# 日志配置
################################################################################

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
# Encode head type in the config tag so multiple TF / non-TF runs don't collide.
HEAD_TAG="${BLOCK_LATENT_HEAD_TYPE}_L${BLOCK_LATENT_HEAD_LAYERS}"
CONFIG_TAG="${MODEL_SIZE}_${BLOCK_MODE}_k${TRAIN_BLOCK_SIZE}_${HEAD_TAG}_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}_${NUM_NODES}nodes_${GPUS_PER_NODE}gpus"
export RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}}"

LOG_DIR="${NOVA_HOME%/nova}/logs_base_train/${DATE_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

echo "=== 启动训练 ==="
echo "RUN_NAME:              ${RUN_NAME}"
echo "LOG_FILE:              ${LOG_FILE}"
echo "BLOCK_MODE:            ${BLOCK_MODE}"
echo "TRAIN_BLOCK_SIZE:      ${TRAIN_BLOCK_SIZE}"
echo "BLOCK_LATENT_HEAD_TYPE: ${BLOCK_LATENT_HEAD_TYPE}"
echo "BLOCK_LATENT_HEAD_LAYERS: ${BLOCK_LATENT_HEAD_LAYERS}"

exec > >(tee -a "${LOG_FILE}") 2>&1

set +e

TORCHRUN_CMD=(
  torchrun
  --nnodes="${NUM_NODES}"
  --nproc_per_node="${GPUS_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  --rdzv_id="${JOB_ID}"
  "${TRAIN_SCRIPT}"
  --run_name "${RUN_NAME}"
  --modality "NLP"
  --model_name "${MODEL_NAME}"
  --model_size "${MODEL_SIZE}"
  --pretokenize_dataset
  --normalize_initial_condition
  --ebt_type "${EBT_TYPE}"
  --denoising_initial_condition "${DENOISING_INITIAL_CONDITION}"
  --mcmc_step_size_learnable
  --mcmc_step_size "${MCMC_STEP_SIZE}"
  --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}"
  --mcmc_num_steps "${MCMC_NUM_STEPS}"
  --training_objective "${TRAINING_OBJECTIVE}"
  --block_mode "${BLOCK_MODE}"
  --context_length "${CONTEXT_LENGTH}"
  --train_block_size "${TRAIN_BLOCK_SIZE}"
  # E2/E4 per-offset head
  --block_latent_head_type "${BLOCK_LATENT_HEAD_TYPE}"
  --block_latent_head_layers "${BLOCK_LATENT_HEAD_LAYERS}"
  --block_latent_head_ffn_mult "${BLOCK_LATENT_HEAD_FFN_MULT}"
  --block_latent_head_n_heads "${BLOCK_LATENT_HEAD_N_HEADS}"
  --gpus "-1"
  --peak_learning_rate "${PEAK_LR}"
  --batch_size_per_device "${DEVICE_BATCH_SIZE}"
  --accumulate_grad_batches "${GRAD_ACCUM}"
  --gradient_clip_val "${GRADIENT_CLIP_VAL}"
  --weight_decay "${WEIGHT_DECAY}"
  --beta1 "${BETA1}"
  --beta2 "${BETA2}"
  --min_lr_scale "${MIN_LR_SCALE}"
  --max_steps "${MAX_STEPS}"
  --max_scheduling_steps "${MAX_SCHEDULING_STEPS}"
  --warm_up_steps "${WARM_UP_STEPS}"
  --warm_up_base_lr_divider "${WARM_UP_BASE_LR_DIVIDER}"
  --dataset_name "nanochat"
  --val_check_interval "${VAL_CHECK_INTERVAL}"
  --limit_val_batches "${LIMIT_VAL_BATCHES}"
  --val_sanity 1
  --validation_split_pct 0.0027
  --wandb_project "nlp_pretrain"
  --log_model_archi
  --set_matmul_precision "medium"
  --float_precision "${FLOAT_PRECISION}"
  --manual_gc_collect_every_n_steps -1
  --save_top_k_ckpts "${SAVE_TOP_K}"
  --save_periodic_steps 1000
)

# --gradient_checkpointing is store_true on train.py side; only append
# when the env var requests it. Default ON to match origin's d26 recipe.
case "${GRADIENT_CHECKPOINTING,,}" in
  ""|0|false|no|off)
    echo "[diagnostic] gradient_checkpointing DISABLED"
    ;;
  *)
    TORCHRUN_CMD+=(--gradient_checkpointing)
    ;;
esac

# Optional E1.1 weights (only forwarded when set, else uniform default).
if [[ -n "${OFFSET_LOSS_WEIGHTS}" ]]; then
  TORCHRUN_CMD+=(--offset_loss_weights "${OFFSET_LOSS_WEIGHTS}")
fi
# Optional E3 ablation flags (store_true on train.py side, empty = off).
if [[ -n "${MCMC_CAUSAL_DETACH}" ]]; then
  TORCHRUN_CMD+=(--mcmc_causal_detach)
fi
if [[ -n "${MCMC_STAGGERED}" ]]; then
  TORCHRUN_CMD+=(--mcmc_staggered)
fi

if [[ -n "${TRAIN_CONTEXT_LENGTH}" ]]; then
  TORCHRUN_CMD+=(--train_context_length "${TRAIN_CONTEXT_LENGTH}")
fi
if [[ -n "${WANDB_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  TORCHRUN_CMD+=(${WANDB_FLAGS})
fi
if [[ -n "${OPTION_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  TORCHRUN_CMD+=(${OPTION_FLAGS})
fi
if [[ -n "${COMPILE_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  TORCHRUN_CMD+=(${COMPILE_FLAGS})
fi

printf ' %q' "${TORCHRUN_CMD[@]}"
echo
"${TORCHRUN_CMD[@]}"

TRAIN_EXIT_CODE=$?
set -e

echo ""
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
  echo "训练成功结束"
else
  echo "训练失败，退出码: ${TRAIN_EXIT_CODE}"
fi

exit "${TRAIN_EXIT_CODE}"
