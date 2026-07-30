#!/bin/bash
################################################################################
# EBT 模型 CORE 评估脚本
# 基于 nanochat 的 base_eval 逻辑，适配 EBT 模型
################################################################################

set -e
set -o pipefail

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
REPO_ROOT="$( cd "$EBT_DIR/../.." && pwd )"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"

# =============================================================================
# 环境配置
# =============================================================================

HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_OFFLINE_MODE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 确保 pip-installed 的 nvidia cuBLAS 优先于系统库，避免版本不兼容导致 CUBLAS_STATUS_INVALID_VALUE
export LD_LIBRARY_PATH=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
# mp.spawn 子进程通过管道输出时 stdout 全缓冲，导致看不到实时进度
export PYTHONUNBUFFERED=1

# 确保 pip-installed 的 nvidia 库优先于系统库，避免 cuBLAS 版本冲突导致 CUBLAS_STATUS_INVALID_VALUE
export LD_LIBRARY_PATH=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

# mp.spawn 子进程通过管道输出时 stdout 默认全缓冲，导致 | tee 场景下看不到实时输出
export PYTHONUNBUFFERED=1

# =============================================================================
# 参数配置
# =============================================================================

# base-train ckpt
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-ctx2048-muon-adamw-0413_20260413_123504/s=step=27999-d26-ctx1024-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.6294.ckpt}"


# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt}"
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260407_001616/e=epoch=0-s=step=812-lr5e-05-bs4x16-muon_adamw-valid_loss=valid_loss=1.2609.ckpt}"

# ctx1048 sft-train
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0420-from0413_20260420_193427/s=step=1953-d26-ctx1024-lr5e-05-bs2x32-muon_adamw-valid_loss=valid_loss=2.1394.ckpt}"
# export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0420-from0413_20260420_193427/periodic-s=step=2999-d26-ctx1024.ckpt}"

# ctx2048 sft-train
export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-20260426/sft_train.v4/checkpoints/s=step=562-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.2264.ckpt}"



if [ -z "$CKPT_PATH" ]; then
    echo "错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt bash runs/eval_ebt_core.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

# 评估配置
EVAL_MODES="${EVAL_MODES:-core}"
MAX_PER_TASK="${MAX_PER_TASK:--1}"
TASK_SAMPLES="${TASK_SAMPLES:-}"
# DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-24}"
NUM_GPUS="${NUM_GPUS:--1}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"
DTYPE="${DTYPE:-bfloat16}"

# 自动检测 GPU
if [ "$NUM_GPUS" = "-1" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "自动检测到 $NUM_GPUS 个 GPU"
    else
        NUM_GPUS=1
    fi
fi

# =============================================================================
# 输出配置 - 统一到 ebt_runs/<exp_id>/core_eval/<eval_run_id>/
# =============================================================================

source "${SCRIPT_DIR}/utils/exp_layout.sh"

if [ -n "$EXP_ID" ]; then
    exp_init_eval "$0"
    OUTPUT_DIR="$EVAL_OUTPUT_DIR"
    LOG_FILE="$EVAL_LOG_FILE"

    exp_save_eval_params "$EVAL_RUN_DIR" \
        "ckpt_path=${CKPT_PATH}" \
        "eval_modes=${EVAL_MODES}" \
        "max_per_task=${MAX_PER_TASK}" \
        "device_batch_size=${DEVICE_BATCH_SIZE}" \
        "num_gpus=${NUM_GPUS}" \
        "dtype=${DTYPE}"
else
    # 兼容旧模式: 不传 EXP_ID 时使用原始路径
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
    RUN_SHORT=$(echo "$RUN_NAME" | sed 's/_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}_\?$//')
    CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
    OUTPUT_DIR="$REPO_ROOT/logs/core_eval/${RUN_SHORT}_${TIMESTAMP}/${CKPT_FILENAME}"
    LOG_FILE="$REPO_ROOT/logs/core_eval/${RUN_SHORT}_${TIMESTAMP}/core_eval.log"
fi

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# 打印配置
# =============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  EBT 模型 CORE 评估"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Checkpoint: $CKPT_PATH"
echo "Tokenizer:  $TOKENIZER_PATH"
echo "评估模式:   $EVAL_MODES"
echo "样本数:     $MAX_PER_TASK (-1 = 全部)"
if [ -n "$TASK_SAMPLES" ]; then
    echo "任务样本:   $TASK_SAMPLES"
fi
echo "Batch Size: $DEVICE_BATCH_SIZE"
echo "GPU 数量:   $NUM_GPUS"
echo "输出目录:   $OUTPUT_DIR"
echo "日志文件:   $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# =============================================================================
# 环境检查
# =============================================================================

if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "错误: 评估数据集 (eval_bundle) 不存在"
    echo "请先运行: cd /mnt/shared-storage-user/puyuan/code/nanochat && bash runs/download_eval_bundle.sh"
    exit 1
fi

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "错误: Tokenizer 不存在: $TOKENIZER_PATH"
    exit 1
fi

# =============================================================================
# 运行评估
# =============================================================================

echo "开始 CORE 评估..."
echo ""

cd "$REPO_ROOT"

# 构建任务样本数参数
TASK_SAMPLES_ARG=""
if [ -n "$TASK_SAMPLES" ]; then
    TASK_SAMPLES_ARG="--task-samples '$TASK_SAMPLES'"
fi

# 确定 Python
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [ -x "$(command -v python3)" ]; then
    PYTHON="python3"
else
    PYTHON="python"
fi

EVAL_CMD="$PYTHON -m openebm.elm.scripts.ebt_core_eval \
    --ckpt-path '$CKPT_PATH' \
    --tokenizer-path '$TOKENIZER_PATH' \
    --eval-bundle-dir '$NANOCHAT_BASE_DIR/eval_bundle' \
    --eval-modes '$EVAL_MODES' \
    --max-per-task $MAX_PER_TASK \
    $TASK_SAMPLES_ARG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --output-dir '$OUTPUT_DIR' \
    --gpus $NUM_GPUS \
    --dtype $DTYPE"

{
    eval $EVAL_CMD
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# 结果摘要
# =============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  CORE 评估结果摘要"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "状态: 成功"
    echo ""

    # 提取 [EVAL_SUMMARY] 行 (Python 端生成)
    SUMMARY_LINE=$(grep -E "\[EVAL_SUMMARY\]" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    if [ -n "$SUMMARY_LINE" ]; then
        echo "  $SUMMARY_LINE"
        echo ""
    fi

    # 提取 CORE 分数 (兼容旧格式)
    CORE_SCORE=$(grep "CORE Metric:" "$LOG_FILE" | tail -1 | awk '{print $3}')
    if [ -n "$CORE_SCORE" ]; then
        echo "  CORE 分数: $CORE_SCORE"
    fi
    echo ""

    # 显示各任务结果
    echo "  各任务详细结果:"
    grep -A 100 "Task Results:" "$LOG_FILE" | grep -E "^\s+[a-z_]+:" | head -20 || echo "  未找到详细结果"
    echo ""

    # 输出文件
    echo "  输出文件:"
    [ -f "$OUTPUT_DIR/core_results.json" ] && echo "    - $OUTPUT_DIR/core_results.json"
    [ -f "$OUTPUT_DIR/core_results.csv" ] && echo "    - $OUTPUT_DIR/core_results.csv"
    echo "    - $LOG_FILE"
else
    echo "状态: 失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "错误信息:"
    tail -30 "$LOG_FILE"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

# 保存评估状态
if [ -n "$EXP_ID" ] && [ -n "$EVAL_RUN_DIR" ]; then
    exp_save_status "$EVAL_RUN_DIR" "core_eval" "$EXIT_CODE"
fi

exit $EXIT_CODE