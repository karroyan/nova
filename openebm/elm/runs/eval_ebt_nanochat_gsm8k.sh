#!/bin/bash
################################################################################
# EBT 评估入口脚本
#
# 统一配置参数 → 调度子任务 → 双向输出(屏幕+日志) → 自动清理
# 运行: bash runs/eval_ebt_nanochat_gsm8k.sh
################################################################################

set -e
set -o pipefail

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
REPO_ROOT="$( cd "$EBT_DIR/../.." && pwd )"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"

# 确保 pip-installed 的 nvidia cuBLAS 优先于系统库，避免多 GPU 下 CUBLAS_STATUS_INVALID_VALUE
export LD_LIBRARY_PATH=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
# mp.spawn 子进程通过管道输出时 stdout 全缓冲，导致看不到实时进度
export PYTHONUNBUFFERED=1

################################################################################
# 通用配置 (所有子脚本共享)
################################################################################

# Checkpoint 路径
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt}"
# base_train bpb 0.81
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt}"

# ctx2048 sft-train
export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-20260426/sft_train.v4/checkpoints/s=step=562-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.2264.ckpt}"


if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 未找到 checkpoint: $CKPT_PATH"
    exit 1
fi

# Tokenizer 路径
export TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"

# GPU 和批量配置
# GPUS=-1 表示自动检测所有可用 GPU
export GPUS="${GPUS:--1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"

# 自动检测 GPU 数量 (参考 eval_ebt_core.sh)
if [ "$GPUS" = "-1" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "自动检测到 $GPUS 个 GPU"
    else
        GPUS=1
    fi
    export GPUS
fi

# WandB 配置
export USE_WANDB="${USE_WANDB:-false}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

################################################################################
# NanoChat 分片评估配置
################################################################################

export EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,15}"
export MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-5}"
export ENABLE_GENERATION="${ENABLE_GENERATION:-true}"
export GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"
export MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"

################################################################################
# GSM8K 评估配置
################################################################################

export EVAL_TASK="${EVAL_TASK:-gsm8k}"

################################################################################
# 日志目录设置
################################################################################

source "${SCRIPT_DIR}/utils/exp_layout.sh"

if [ -n "$EXP_ID" ]; then
    # EXP_ID 模式: 输出到 logs/ebt_runs/<EXP_ID>/nanochat_gsm8k_eval/
    exp_init_nanochat_gsm8k_eval "$0"
    MASTER_LOG="$EVAL_LOG_FILE"
else
    # Fallback 模式: 输出到 $REPO_ROOT/logs/nanochat_gsm8k_eval/
    RUN_NAME=$(basename "$(dirname "$CKPT_PATH")")
    RUN_SHORT=$(echo "$RUN_NAME" | sed 's/_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}_\?$//')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    export EVAL_RUN_DIR="$REPO_ROOT/logs/nanochat_gsm8k_eval/${RUN_SHORT}_${TIMESTAMP}"
    mkdir -p "$EVAL_RUN_DIR"

    MASTER_LOG="$EVAL_RUN_DIR/master.log"
    export NANOCHAT_EVAL_LOG="$EVAL_RUN_DIR/nanochat_shards.log"
    export GSM8K_EVAL_LOG="$EVAL_RUN_DIR/gsm8k.log"
fi

################################################################################
# 自动清理: 只保留最近 MAX_KEEP 个运行目录
################################################################################

MAX_KEEP=10

# 清理的基目录与 EVAL_RUN_DIR 的父目录保持一致
EVAL_BASE_DIR="$(dirname "$EVAL_RUN_DIR")"
cleanup_old_run_dirs() {
    # 仅在 fallback 模式下清理 (EXP_ID 模式由实验管理)
    [ -n "$EXP_ID" ] && return 0
    local dirs
    dirs=$(ls -1dt "$EVAL_BASE_DIR"/${RUN_SHORT}_[0-9]* 2>/dev/null || true)
    local count
    count=$(echo "$dirs" | grep -c . 2>/dev/null || true)
    if [ "$count" -gt "$MAX_KEEP" ]; then
        echo "$dirs" | tail -n +$((MAX_KEEP + 1)) | xargs rm -rf
        echo "  🧹 清理 $(( count - MAX_KEEP )) 个旧运行目录"
    fi
}

cleanup_old_run_dirs

################################################################################
# 主逻辑 (包裹在函数中，通过管道 tee 实现双向输出)
################################################################################

main() {
    echo "======================================================================"
    echo "  EBT 模型评估 - $RUN_SHORT"
    echo "  时间: $(date)"
    echo "======================================================================"
    echo ""
    echo "📦 Checkpoint : $CKPT_PATH"
    echo "📦 Tokenizer  : $TOKENIZER_PATH"
    echo "📦 GPUs: $GPUS | Batch: $BATCH_SIZE | LimitBatches: $LIMIT_TEST_BATCHES"
    echo ""
    echo "📁 运行目录   : $EVAL_RUN_DIR"
    echo "   主日志     : $(basename "$MASTER_LOG")"
    echo "   NanoChat   : $(basename "$NANOCHAT_EVAL_LOG")"
    echo "   GSM8K      : $(basename "$GSM8K_EVAL_LOG")"
    echo ""

    ############################################################################
    # 任务 1: NanoChat 分片评估
    ############################################################################

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  任务 1/2: NanoChat 分片评估 (shard $EVAL_SHARD_INDICES)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    TASK1_START=$(date +%s)

    if bash "$SCRIPT_DIR/eval_nanochat_shards.sh"; then
        TASK1_STATUS="✅ 成功"
    else
        TASK1_STATUS="❌ 失败 (退出码: $?)"
    fi

    TASK1_END=$(date +%s)
    TASK1_DURATION=$(( TASK1_END - TASK1_START ))

    echo ""
    echo "  ⏱ NanoChat 耗时: ${TASK1_DURATION}s | $TASK1_STATUS"
    echo ""

    ############################################################################
    # 任务 2: GSM8K 数学推理
    ############################################################################

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  任务 2/2: GSM8K 数学推理"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    TASK2_START=$(date +%s)

    if bash "$SCRIPT_DIR/eval_ebt.sh"; then
        TASK2_STATUS="✅ 成功"
    else
        TASK2_STATUS="❌ 失败 (退出码: $?)"
    fi

    TASK2_END=$(date +%s)
    TASK2_DURATION=$(( TASK2_END - TASK2_START ))

    echo ""
    echo "  ⏱ GSM8K 耗时: ${TASK2_DURATION}s | $TASK2_STATUS"
    echo ""

    ############################################################################
    # 汇总报告
    ############################################################################

    TOTAL_DURATION=$(( TASK2_END - TASK1_START ))

    echo ""
    echo "======================================================================"
    echo "  评估完成 - 汇总"
    echo "======================================================================"
    echo ""
    echo "  模型      : $RUN_SHORT"
    echo "  NanoChat  : $TASK1_STATUS  (${TASK1_DURATION}s)"
    echo "  GSM8K     : $TASK2_STATUS  (${TASK2_DURATION}s)"
    echo "  总耗时    : ${TOTAL_DURATION}s"
    echo ""

    # 提取 NanoChat 关键指标 (从 [EVAL_SUMMARY] 标记行)
    if [ -f "$NANOCHAT_EVAL_LOG" ]; then
        NANOCHAT_SUMMARY=$(grep -E "\[EVAL_SUMMARY\]" "$NANOCHAT_EVAL_LOG" 2>/dev/null | tail -1 || true)
        if [ -n "$NANOCHAT_SUMMARY" ]; then
            echo "  NanoChat 指标:"
            echo "    $NANOCHAT_SUMMARY"
            echo ""
        fi
        # Also show PPL statistics if present
        PPL_LINE=$(grep -E "Mean:.*[0-9]" "$NANOCHAT_EVAL_LOG" 2>/dev/null | head -4 || true)
        if [ -n "$PPL_LINE" ]; then
            echo "  NanoChat PPL/Loss 详情:"
            echo "$PPL_LINE" | sed 's/^/    /'
            echo ""
        fi
    fi

    # 提取 GSM8K 关键指标
    if [ -f "$GSM8K_EVAL_LOG" ]; then
        GSM8K_SUMMARY=$(grep -E "\[EVAL_SUMMARY\]" "$GSM8K_EVAL_LOG" 2>/dev/null | tail -1 || true)
        if [ -n "$GSM8K_SUMMARY" ]; then
            echo "  GSM8K 指标:"
            echo "    $GSM8K_SUMMARY"
            echo ""
        fi
    fi

    echo "  📁 完整日志:"
    echo "    $EVAL_RUN_DIR"
    echo ""
    echo "======================================================================"
}

# 运行主逻辑: stdout/stderr 同时输出到屏幕和主日志文件
main 2>&1 | tee "$MASTER_LOG"
exit ${PIPESTATUS[0]}
