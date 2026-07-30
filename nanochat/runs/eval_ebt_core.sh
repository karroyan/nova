#!/bin/bash
################################################################################
# EBT 模型 CORE 评估脚本
# 基于 nanochat 的 base_eval 逻辑，适配 EBT 模型
################################################################################

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 环境配置
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_OFFLINE_MODE=1
export OMP_NUM_THREADS=1

# EBT checkpoint 路径（必须指定）
CKPT_PATH="${CKPT_PATH:-}"
if [ -z "$CKPT_PATH" ]; then
    echo "❌ 错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt bash runs/eval_ebt_core.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

# 评估配置
EVAL_MODES="${EVAL_MODES:-core}"  # 默认只评估 CORE，可选 core,bpb,sample
MAX_PER_TASK="${MAX_PER_TASK:--1}"  # -1 表示评估所有样本
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-1}"

# Tokenizer 路径
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"

# 输出配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
OUTPUT_DIR="../../openebm/elm/logs/core_eval/${RUN_NAME}/${CKPT_FILENAME}"
LOG_FILE="../../openebm/elm/logs/core_eval_${TIMESTAMP}.log"

# 创建输出目录
cd /mnt/shared-storage-user/puyuan/code/nanochat
cd openebm/elm
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p "logs"
cd ../..

# =============================================================================
# 检查环境准备
# =============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  EBT 模型 CORE 评估"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📦 Checkpoint: $CKPT_PATH"
echo "📦 Tokenizer: $TOKENIZER_PATH"
echo "📊 评估模式: $EVAL_MODES"
echo "📊 每任务最大样本数: $MAX_PER_TASK"
echo "📊 Batch Size: $DEVICE_BATCH_SIZE"
echo "📊 GPU 数量: $NUM_GPUS"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📁 日志文件: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 检查 eval_bundle 是否存在
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "❌ 错误: 评估数据集 (eval_bundle) 不存在"
    echo "请先下载评估数据集:"
    echo "  cd /mnt/shared-storage-user/puyuan/code/nanochat"
    echo "  bash runs/download_eval_bundle.sh"
    exit 1
fi

echo "✅ 评估数据集已就绪: $NANOCHAT_BASE_DIR/eval_bundle"
echo ""

# 检查 tokenizer 是否存在
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "❌ 错误: Tokenizer 不存在: $TOKENIZER_PATH"
    exit 1
fi

echo "✅ Tokenizer 已就绪"
echo ""

# =============================================================================
# 激活环境
# =============================================================================

echo "🔧 激活虚拟环境..."
source .venv/bin/activate 2>/dev/null || source /mnt/shared-storage-user/puyuan/code/nanochat/.venv/bin/activate
export PYTHONPATH="${PWD}/nanochat:${PWD}:${PYTHONPATH:-}"

echo "✅ 环境已激活"
echo ""

# =============================================================================
# 运行 CORE 评估
# =============================================================================

echo "🚀 开始 CORE 评估..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 构建评估命令
cd openebm/elm

EVAL_CMD="python -m openebm.elm.scripts.ebt_core_eval \
    --ckpt-path '$CKPT_PATH' \
    --tokenizer-path '$TOKENIZER_PATH' \
    --eval-bundle-dir '$NANOCHAT_BASE_DIR/eval_bundle' \
    --eval-modes '$EVAL_MODES' \
    --max-per-task $MAX_PER_TASK \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --output-dir '$OUTPUT_DIR' \
    --gpus $NUM_GPUS"

echo "执行命令:"
echo "$EVAL_CMD"
echo ""

# 执行评估（将输出同时保存到日志和显示在屏幕）
{
    eval $EVAL_CMD
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

cd ../..

# =============================================================================
# 显示结果摘要
# =============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  CORE 评估结果摘要"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 状态: 成功"
    echo ""

    # 提取 CORE 分数
    CORE_SCORE=$(grep "CORE Metric:" "$LOG_FILE" | tail -1 | awk '{print $3}')
    if [ ! -z "$CORE_SCORE" ]; then
        echo "📊 CORE 分数: $CORE_SCORE"
    fi
    echo ""

    # 显示各任务结果
    echo "📋 各任务详细结果:"
    grep -A 100 "Task Results:" "$LOG_FILE" | grep -E "^\s+[a-z_]+:" | head -20 || echo "  未找到详细结果"
    echo ""

    # 输出文件
    echo "📁 输出文件:"
    if [ -f "$OUTPUT_DIR/core_results.json" ]; then
        echo "  - CORE 结果: $OUTPUT_DIR/core_results.json"
    fi
    if [ -f "$OUTPUT_DIR/core_results.csv" ]; then
        echo "  - CSV 报告: $OUTPUT_DIR/core_results.csv"
    fi
    echo "  - 完整日志: $LOG_FILE"

else
    echo "❌ 状态: 失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "📋 错误信息:"
    tail -30 "$LOG_FILE"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

exit $EXIT_CODE
