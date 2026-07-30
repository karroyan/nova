#!/bin/bash

################################################################################
# EBT d26 训练脚本 - Muon+AdamW 混合优化器 (对齐 NanoChat)
#
# 日期: 2026-03-19
# 策略:
#   1. 优化器: Muon+AdamW (复用 nanochat/optim.py)
#      - Transformer 矩阵参数 → Muon (LR=0.02)
#      - Embedding/Scalar/Alpha → AdamW (beta1=0.8, beta2=0.95)
#   2. Weight Decay: 0.2 + 动态衰减到 0 (对齐 base_train.py)
#   3. LR 调度: Linear Warmdown (后50%衰减到0, 对齐 base_train.py)
#   4. EBT 特有参数保持官方推荐 (MCMC step_size, alpha LR 等)
#
# 参考:
#   - NanoChat: scripts/base_train.py, nanochat/optim.py, nanochat/gpt.py
#   - EBT 官方: https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-d26-stable
#SBATCH --output=logs/slurm/nlp/ebt-d26-stable_%A-%a.log

### 基础配置 ###
# RUN_PREFIX 用于 exp_id 生成（如需手动指定 EXP_ID，设置 EXP_ID 环境变量）
RUN_PREFIX="${RUN_PREFIX:-ebt-d26-ctx2048-ve}"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"
# export MODEL_SIZE="medium"



### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# WandB 配置
# export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# EBT 核心超参数 (严格遵循官方建议)
################################################################################
# 来源: https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py
#
# 官方说明:
# "keeping mcmc_step_size_lr_multiplier = 3x mcmc_step_size is safe and
#  what works well so the most important and arguably only really necessary
#  to tune hparam is mcmc_step_size"
#
# 关键要点:
# 1. mcmc_step_size 是唯一需要调优的超参数
# 2. mcmc_step_size_lr_multiplier 应该是 step_size 的 3 倍
# 3. 其他参数保持默认即可
################################################################################

# EBT 特定参数
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"               # 官方建议: can almost always stay
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false
USE_MCMC_TIME_EMBED=false          # false: shared transition kernel, supports arbitrary inference steps

# 不使用的高级参数 (官方: not recommended for getting started)
# ebt_norm, ebt_act_func, dyt_alpha_init, mcmc_replay_buffer 等
# 均保持代码默认值,不在命令行覆盖

################################################################################
# Batch 配置与训练步数自动计算 - 针对 d26 模型优化
################################################################################

# 1. 硬件与 Batch 基础配置
# NUM_GPUS=8
# DEVICE_BATCH_SIZE=4
# GRAD_ACCUM=8
# CONTEXT_LENGTH=512

# 可运行
# NUM_GPUS=8
# DEVICE_BATCH_SIZE=1
# GRAD_ACCUM=2
# CONTEXT_LENGTH=2048

# --float_precision "bf16-mixed" \
# --gradient_checkpointing \

# 可运行
NUM_GPUS=8
DEVICE_BATCH_SIZE=1
# DEVICE_BATCH_SIZE=2
# GRAD_ACCUM=24
# GRAD_ACCUM=12
# GRAD_ACCUM=1
GRAD_ACCUM=32

CONTEXT_LENGTH=2048

# NUM_GPUS=8
# DEVICE_BATCH_SIZE=1
# GRAD_ACCUM=48
# CONTEXT_LENGTH=2048


# 2. 计算每步的有效 Token 数 (Tokens per step)
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# 当前配置: 8 × 4 × 8 × 512 = 131,072 tokens/step

# 3. 设置目标总 Token 数
# NanoChat d26 (speedrun.sh --depth=26):
#   tokens/step = 16 × 2048 × 4(accum) × 8(GPUs) = 1,048,576
#   7000 step × 1,048,576 = 7,340,032,000 ≈ 7.34B tokens → bpb=0.747
TARGET_TOTAL_TOKENS=7340032000 # 约 7.34B tokens

# 注意: 由于EBT训练时显存消耗更多，目前 NanoChat 的 batch size 更大 ,
# 大 batch 梯度噪声更低, 每步更新更有效,
# 所以即使 token 总量相同, EBT 可能需要更多步才能达到同等效果

# 4. 自动计算总 Steps 数 (向下取整)
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置 (基于模型规模推荐)
################################################################################
# 来源: https://github.com/openway-ai/nova/blob/main/nova/ebt/utils.py
#
# d26 模型规模分析:
#   - 26 layers, 13 heads, 1664 dim
#   - 参数量: ~1031M (1.03B)
#
# 模型规模对比:
#   small (162M):   LR = 0.0006
#   medium (405M):  LR = 0.0003
#   large (833M):   LR = 0.00025  ← 最接近
#   d26 (1031M):    LR = 0.00025  ← 推荐 (介于large和xl之间,更接近large)
#   xl (1413M):     LR = 0.0002
#
# 重要: d26 比 large 大 24%, 但远小于 xl (仅为 xl 的 73%)
#       因此应该使用 large 的 LR, 而不是 small 的 LR!
#
# 注意: Alpha 的实际 LR = PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER
#       = 0.00025 × 1500 = 0.375 ✓ (官方推荐范围 0.3-0.9)
################################################################################

PEAK_LR=0.00025  # 与 large 一致 (d26 参数量更接近 large 而非 small)

# Warmup 配置: 无 warmup (对齐 NanoChat warmup_ratio=0.0)
# 注意: 使用 --linear_warmdown 时, WARM_UP_STEPS 和 MIN_LR_SCALE 不生效
# 实际由 --warmup_ratio 和 --warmdown_ratio 控制
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10

# LR 衰减: 由 --linear_warmdown 控制 (后 50% 线性衰减到 0)
# 以下参数仅在不使用 --linear_warmdown 时生效 (cosine annealing fallback)
MIN_LR_SCALE=50

################################################################################
# 优化器配置 (Muon+AdamW 模式, 对齐 NanoChat base_train.py)
################################################################################

# Weight Decay: NanoChat 默认 0.2, 配合 --dynamic_wd 线性衰减到 0
# 参考 base_train.py:63: --weight-decay 0.2
# 参考 base_train.py:366: weight_decay_scaled * (1 - it / num_iterations)
WEIGHT_DECAY=0.2

# Adam Beta 参数: 对齐 NanoChat (用于 AdamW 部分: embedding/scalar/alpha)
# 参考 base_train.py:66-67: --adam-beta1 0.8, --adam-beta2 0.95
# 注意: 这些 beta 是 NanoChat 配合 Muon 验证过的组合
BETA1=0.8
BETA2=0.95

# 梯度裁剪: 标准设置
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

# VAL_CHECK_INTERVAL=1000
VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
# NUM_WORKERS: passed to train.py --num_workers but NOT used by nanochat dataloader
# (hardcoded num_workers=0 in dataset.py because nanochat generator holds GPU state).
# Kept for compatibility with the CLI arg parser.
NUM_WORKERS=19
SAVE_TOP_K=2

################################################################################
# 优化选项配置
################################################################################
#
# 当前启用: Muon+AdamW 混合优化器 + 动态 Weight Decay + Linear Warmdown
#
# 可选优化 (逐个测试):
#   1. --layered_lr: 分层学习率 (仅用于纯 AdamW 模式, muon_adamw 已内置分层)
#   2. --dynamic_wd: 动态 Weight Decay (线性衰减到0, 对齐 NanoChat base_train.py:366)
#   3. --linear_warmdown: NanoChat 风格 LR 调度 (对齐 base_train.py:347)
#   4. --optimizer muon_adamw: Muon+AdamW 混合优化器 (复用 nanochat/optim.py)
#      Muon 处理 transformer 矩阵参数 (LR=0.02), AdamW 处理 embedding/scalar/alpha
#
# 注意:
#   - --layered_lr 与 --optimizer muon_adamw 不要同时使用 (muon_adamw 已内置分层)
#   - --dynamic_wd 和 --linear_warmdown 可以与 muon_adamw 组合使用
################################################################################

# === Baseline: 纯 AdamW, 不启用任何高级优化 ===
# OPTION_FLAGS=""

# === 纯 AdamW + 分层学习率 ===
# OPTION_FLAGS="--layered_lr"

# === Muon+AdamW 全套优化 (对齐 NanoChat) ===
# --optimizer muon_adamw: Muon 处理 transformer 矩阵, AdamW 处理其余
# --dynamic_wd: Weight Decay 线性衰减到 0 (对齐 base_train.py:366)
# --linear_warmdown: 线性 warmdown LR 调度 (对齐 base_train.py:347)
#
# AdamW 独立绝对 LR (对齐 NanoChat, 不再从 peak_lr 派生):
#   embedding:       0.3 (对齐 NanoChat base_train.py:61, 不在 MCMC 循环内, 可用高 LR)
#   vocab_to_embed:  0.01 (EBT 特有, 在 MCMC 循环内有二阶梯度, 需保守)
#   scalar:          0.04 (RMSNorm 等, 在 MCMC 循环内, 适度保守)
#   alpha:           保持 PEAK_LR × MCMC_LR_MULT = 0.375 (EBT 特有, 不变)
# --adamw_dmodel_lr_scaling: 对 AdamW LR 做 (dim/768)^-0.5 缩放 (对齐 gpt.py:362)
#   d26 dim=1664: scale = (1664/768)^-0.5 ≈ 0.679
#   实际 embedding LR = 0.3 × 0.679 = 0.204
#   实际 scalar LR = 0.04 × 0.679 = 0.027
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"


################################################################################
# torch.compile 配置
################################################################################
# EBT 使用 autograd.grad 进行 MCMC 更新,与 fullgraph 模式不兼容
# 推荐: transformer_only 模式,仅编译 transformer 部分

# 目前这个会报错
# COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

# 不使用 compile，steps 较少时适用
# COMPILE_FLAGS="--compile_model --compile_mode disabled" 
# 启用 compile，steps 较多时适用
# full mode causes repeated recompilation with create_graph=True + bf16-mixed
COMPILE_FLAGS="--compile_model --compile_mode full"
# COMPILE_FLAGS="--compile_model --compile_mode disabled"

################################################################################
# WandB 配置 (训练参数)
################################################################################
# 默认: 只记录 loss 等基础 metric, 不记录 gradients/activations
# 如需 debug 梯度, 取消下面注释切换到全量记录
#
# 选项说明:
#   不传 --wandb_watch        → 只记录 scalar metrics (最快, 默认)
#   --wandb_watch              → 记录 parameters only (中等开销)
#   --wandb_watch --wandb_watch_level all → 记录 gradients+parameters+activations (最慢, debug 用)
#   --disable_wandb            → 完全关闭 wandb
#
WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level parameters"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level all"

################################################################################
# 实验目录布局 - 统一输出到 ebt_runs/<exp_id>/base_train/
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

exp_init_base_train "$0" "muon_adamw"
export RUN_NAME="${EXP_ID}"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

exp_save_hparams "${EXP_DIR}/base_train" \
    "model_size=${MODEL_SIZE}" \
    "context_length=${CONTEXT_LENGTH}" \
    "peak_lr=${PEAK_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "beta1=${BETA1}" \
    "beta2=${BETA2}" \
    "device_batch_size=${DEVICE_BATCH_SIZE}" \
    "grad_accum=${GRAD_ACCUM}" \
    "num_gpus=${NUM_GPUS}" \
    "effective_batch_size=${EFFECTIVE_BATCH_SIZE}" \
    "max_steps=${MAX_STEPS}" \
    "target_total_tokens=${TARGET_TOTAL_TOKENS}" \
    "mcmc_step_size=${MCMC_STEP_SIZE}" \
    "mcmc_lr_multiplier=${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
    "use_mcmc_time_embed=${USE_MCMC_TIME_EMBED}" \
    "optimizer=muon_adamw"

LOG_FILE="${EXP_LOG_FILE}"

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# 日志函数
print_separator() {
    local char=${1:-"═"}
    local width=${2:-80}
    printf "${BLUE}"
    printf '%*s' "$width" | tr ' ' "$char"
    printf "${NC}\n"
}

print_header() {
    local title=$1
    echo ""
    print_separator "═"
    echo -e "${BOLD}${GREEN}  $title${NC}"
    print_separator "═"
}

print_kv() {
    local key=$1
    local value=$2
    local width=${3:-30}
    printf "  ${DIM}%-${width}s${NC} %s\n" "$key:" "$value"
}

################################################################################
# 显示配置
################################################################################

print_header "EBT d26 Muon+AdamW 训练配置"

echo ""
echo -e "${CYAN}▶ 优化器策略 (对齐 NanoChat)${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} Muon+AdamW 混合优化器 (nanochat/optim.py)"
echo "  ${GREEN}✓${NC} Transformer 矩阵参数 → Muon (LR=0.02)"
echo "  ${GREEN}✓${NC} Embedding/Scalar/Alpha → AdamW"
echo "  ${GREEN}✓${NC} Adam Beta: (${BOLD}${BETA1}, ${BETA2}${NC}) (对齐 NanoChat)"
echo "  ${GREEN}✓${NC} Weight Decay: ${BOLD}${WEIGHT_DECAY}${NC} + 动态衰减到 0"
echo "  ${GREEN}✓${NC} LR 调度: Linear Warmdown (后50%衰减到0)"

echo ""
echo -e "${CYAN}▶ EBT 核心参数 (官方推荐)${NC}"
print_separator "─" 60
print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)"
print_kv "MCMC Num Steps" "${MCMC_NUM_STEPS}"
print_kv "EBT Type" "${EBT_TYPE}"
print_kv "Normalize Init Condition" "${NORMALIZE_INITIAL_CONDITION}"
print_kv "Denoising Init Condition" "${DENOISING_INITIAL_CONDITION}"

echo ""
echo -e "${CYAN}▶ 模型配置${NC}"
print_separator "─" 60
print_kv "Model Size" "${MODEL_SIZE}"
print_kv "Context Length" "${CONTEXT_LENGTH}"
print_kv "估计参数量" "~1031M (1.03B) - 介于large和xl之间"

echo ""
echo -e "${CYAN}▶ Batch 配置${NC}"
print_separator "─" 60
print_kv "Device Batch Size" "${DEVICE_BATCH_SIZE}"
print_kv "Gradient Accumulation" "${GRAD_ACCUM}"
print_kv "Num GPUs" "${NUM_GPUS}"
print_kv "Effective Batch Size" "${EFFECTIVE_BATCH_SIZE} tokens/step"

echo ""
echo -e "${CYAN}▶ 学习率配置${NC}"
print_separator "─" 60
print_kv "Peak LR (Base)" "${PEAK_LR}"
local alpha_lr=$(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
print_kv "Alpha Effective LR" "${BOLD}${alpha_lr}${NC} (=${PEAK_LR}×${MCMC_STEP_SIZE_LR_MULTIPLIER})"
print_kv "Warmup Steps" "${WARM_UP_STEPS} (5%)"
print_kv "Min LR Scale" "${MIN_LR_SCALE}"
local final_lr=$(awk "BEGIN {printf \"%.8f\", ${PEAK_LR}/${MIN_LR_SCALE}}")
print_kv "Final LR" "${final_lr}"

echo ""
echo -e "${CYAN}▶ 优化器配置${NC}"
print_separator "─" 60
print_kv "Weight Decay" "${WEIGHT_DECAY}"
print_kv "Adam Beta1" "${BETA1}"
print_kv "Adam Beta2" "${BETA2}"
print_kv "Gradient Clip" "${GRADIENT_CLIP_VAL}"

echo ""
echo -e "${CYAN}▶ 训练进度${NC}"
print_separator "─" 60
print_kv "Target Tokens" "7.34B"
print_kv "Max Steps" "${MAX_STEPS} (自适应计算)"
print_kv "Max Sched Steps" "${MAX_SCHEDULING_STEPS}"
# 注意：在 bash 脚本的全局作用域中不应使用 local 关键字，直接赋值即可
total_tokens_b=$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")
print_kv "Actual Total Tokens" "~${total_tokens_b}B"
print_kv "Val Check Interval" "${VAL_CHECK_INTERVAL}"

echo ""
echo -e "${CYAN}▶ 输出配置${NC}"
print_separator "─" 60
print_kv "Run Name" "${RUN_NAME}"
print_kv "Log File" "${LOG_FILE}"
print_kv "WandB Mode" "${WANDB_MODE}"

echo ""
echo -e "${YELLOW}⚠ 重要说明${NC}"
echo "  1. EBT 核心参数 (MCMC) 保持官方推荐值不变"
echo "  2. 优化器/LR调度/Weight Decay 对齐 NanoChat base_train.py"
echo "  3. Alpha LR 仍由 MCMC_STEP_SIZE_LR_MULTIPLIER × PEAK_LR 控制 (EBT 特有)"
echo "  4. 监控关键指标: train_loss, Alpha_MCMC, Global_LR"

echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 Muon+AdamW 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 优化器策略 (对齐 NanoChat base_train.py):
#   1. Muon+AdamW 混合优化器 (nanochat/optim.py)
#   2. Adam Beta: (${BETA1}, ${BETA2})
#   3. Weight Decay: ${WEIGHT_DECAY} + 动态衰减到 0
#   4. LR 调度: Linear Warmdown (后50%衰减到0)
#   5. EBT 特有: Alpha LR = PEAK_LR × MCMC_LR_MULT (保持不变)
#
################################################################################

================================================================================
[SYSTEM INFO]
================================================================================
Hostname:         $(hostname)
User:             $(whoami)
Python:           $(python3 --version 2>&1)
PyTorch:          $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA Available:   $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
GPU Count:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "N/A")
GPU Model:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

================================================================================
[EBT 核心参数 - 官方推荐]
================================================================================
MCMC Step Size:           ${MCMC_STEP_SIZE}
MCMC LR Multiplier:       ${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)
Alpha Effective LR:       $(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
MCMC Num Steps:           ${MCMC_NUM_STEPS}
EBT Type:                 ${EBT_TYPE}

================================================================================
[训练配置]
================================================================================
Model Size:               ${MODEL_SIZE}
Context Length:           ${CONTEXT_LENGTH}
Device Batch Size:        ${DEVICE_BATCH_SIZE}
Gradient Accumulation:    ${GRAD_ACCUM}
Effective Batch Size:     ${EFFECTIVE_BATCH_SIZE} tokens/step

Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Warmup Steps:             ${WARM_UP_STEPS}
Max Steps:                ${MAX_STEPS}
Total Tokens:             ${total_tokens_b}

Option Flags:             ${OPTION_FLAGS:-"None (Baseline)"}
Compile Flags:            ${COMPILE_FLAGS:-"None"}

================================================================================
[训练输出]
================================================================================

LOG_HEADER

################################################################################
# 自动重定向所有输出到日志文件 (替代手动 tee)
################################################################################
exec > >(tee -a "${LOG_FILE}") 2>&1

################################################################################
# 启动训练
################################################################################

echo ""
print_header "开始训练"
echo ""

set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/train.py \
--run_name ${RUN_NAME} \
--checkpoint_dir "${EXP_CKPT_DIR}" \
--wandb_save_dir "${EXP_WANDB_DIR}" \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
--denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps ${MCMC_NUM_STEPS} \
$([ "$USE_MCMC_TIME_EMBED" = true ] && echo "--use_mcmc_time_embed") \
\
--context_length ${CONTEXT_LENGTH} \
\
--gpus "-1" \
\
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
\
--weight_decay ${WEIGHT_DECAY} \
--beta1 ${BETA1} \
--beta2 ${BETA2} \
--min_lr_scale ${MIN_LR_SCALE} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--warm_up_steps ${WARM_UP_STEPS} \
--warm_up_base_lr_divider ${WARM_UP_BASE_LR_DIVIDER} \
\
--dataset_name "nanochat" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
\
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--use_ve \
--set_matmul_precision "medium" \
--float_precision "bf16-mixed" \
--manual_gc_collect_every_n_steps -1 \
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps 1000 \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

# --use_ve \

# --manual_gc_collect_every_n_steps -1 \

# --float_precision "bf16-mixed" \
# --cpu_offload_optimizer \
# --gradient_checkpointing \
# --manual_gc_collect_every_n_steps 50 \


TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_DIR}/base_train" "base_train" "$TRAIN_EXIT_CODE"

################################################################################
# 训练结束处理
################################################################################

echo ""
print_header "训练结束"

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "${RED}✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)${NC}"

    # 检查常见问题
    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
        echo -e "${YELLOW}⚠ 检测到 OOM - 建议减小 DEVICE_BATCH_SIZE${NC}"
    fi

    if grep -q "nan\|NaN\|inf\|Inf" "${LOG_FILE}" 2>/dev/null | head -5; then
        echo -e "${YELLOW}⚠ 检测到 NaN/Inf - 可能需要进一步降低学习率${NC}"
    fi
fi

echo ""
echo "日志文件: ${LOG_FILE}"
echo ""

################################################################################
# 监控建议
################################################################################

print_header "训练监控建议"
cat << MONITOR_HELP

实时监控命令 (在另一个终端运行):
  tail -f ${LOG_FILE} | grep -E "(train_loss|Alpha_MCMC|Global_LR)"

关键指标检查:
  1. train_loss: 应该平稳下降,避免突然飙升
  2. Alpha_MCMC: MCMC步长应该稳定在合理范围 (如 400-600)
  3. Global_LR: 学习率应该平滑衰减

异常诊断:
  - 如果 loss 突然飙升: 检查 Alpha_MCMC 是否异常
  - 如果 Alpha_MCMC 震荡剧烈: 考虑进一步降低 MCMC_STEP_SIZE_LR_MULTIPLIER
  - 如果 NaN 出现: 降低 PEAK_LR 或启用 gradient checkpointing

对比基准 (之前失败的配置):
  - 之前: Alpha LR = 1.2, 多次 loss 暴涨
  - 现在: Alpha LR = 0.375, 预期稳定

MONITOR_HELP

echo ""
