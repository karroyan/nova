#!/bin/bash
################################################################################
# EBT Web 对话服务启动脚本
#
# 通过网页端与训练好的 EBT 模型进行交互式对话，支持流式输出和 /xx 命令
#
# 使用方法:
#   bash runs/chat_ebt_web.sh                        # 默认模式
#   bash runs/chat_ebt_web.sh --show-mcmc            # 展示 MCMC 步骤
#   bash runs/chat_ebt_web.sh --port 8080            # 指定端口
#   bash runs/chat_ebt_web.sh --help                 # 显示帮助
#
# 环境变量:
#   CKPT_PATH      - Checkpoint 路径 (可选)
#   TEMPERATURE    - 生成温度 (默认: 0.8)
#   TOP_P          - Top-P 采样 (默认: 0.9)
#   MAX_TOKENS     - 最大生成 tokens (默认: 512)
#   PORT           - 服务端口 (默认: 8000)
################################################################################

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
REPO_ROOT="$( cd "$EBT_DIR/../.." && pwd )"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"

# 默认配置 (与 chat_ebt.sh 保持一致)
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt"
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260408_230436/periodic-s=step=2999-lr5e-05.ckpt"
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260408_230436/periodic-s=step=1499-lr5e-05.ckpt"

DEFAULT_TOKENIZER="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer"

# 从环境变量读取配置
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-512}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# 解析命令行参数
SHOW_MCMC_FLAG=""
VERBOSE_FLAG=""
SHOW_ENERGY_FLAG=""
SHOW_DIST_FLAG=""
OVERRIDE_MCMC_STEPS=""
OVERRIDE_NOISE_STD=""
OVERRIDE_ALPHA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --show-mcmc|-m)
            SHOW_MCMC_FLAG="--show-mcmc"
            shift
            ;;
        --verbose|-v)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        --show-energy|-e)
            SHOW_ENERGY_FLAG="--show-energy"
            shift
            ;;
        --show-distribution|-d)
            SHOW_DIST_FLAG="--show-distribution"
            shift
            ;;
        --checkpoint|-c)
            CKPT_PATH="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p|-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --override-mcmc-steps)
            OVERRIDE_MCMC_STEPS="$2"
            shift 2
            ;;
        --override-noise-std)
            OVERRIDE_NOISE_STD="$2"
            shift 2
            ;;
        --override-alpha)
            OVERRIDE_ALPHA="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "EBT Web 对话服务启动脚本"
            echo ""
            echo "使用方法:"
            echo "  bash runs/chat_ebt_web.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --checkpoint PATH          Checkpoint 路径"
            echo "  -t, --temperature VAL          生成温度 (默认: 0.8)"
            echo "  -p, --top-p VAL                Top-P 采样 (默认: 0.9)"
            echo "  --max-tokens VAL               最大生成 tokens (默认: 512)"
            echo "  --port PORT                    服务端口 (默认: 8000)"
            echo "  --host HOST                    绑定地址 (默认: 0.0.0.0)"
            echo "  -m, --show-mcmc                展示 MCMC 步骤过程"
            echo "  -v, --verbose                  详细模式"
            echo "  -e, --show-energy              展示能量值变化"
            echo "  -d, --show-distribution        展示概率分布变化"
            echo "  --override-mcmc-steps N        覆盖 MCMC 步数"
            echo "  --override-noise-std VAL       覆盖 Langevin 噪声"
            echo "  --override-alpha VAL           覆盖 MCMC 步长 alpha"
            echo "  --dtype TYPE                   数据类型 (float32/bfloat16)"
            echo "  --device DEVICE                设备 (cuda/cpu)"
            echo "  -h, --help                     显示此帮助"
            echo ""
            echo "示例:"
            echo "  bash runs/chat_ebt_web.sh --port 8080"
            echo "  bash runs/chat_ebt_web.sh --show-mcmc --override-mcmc-steps 10"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 从环境变量设置标志
if [ "${SHOW_MCMC:-false}" = "true" ]; then
    SHOW_MCMC_FLAG="--show-mcmc"
fi

if [ "${VERBOSE:-false}" = "true" ]; then
    VERBOSE_FLAG="--verbose"
fi

# 验证 checkpoint 存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: 找不到 checkpoint: $CKPT_PATH"
    echo ""
    echo "请设置 CKPT_PATH 环境变量或使用 --checkpoint 参数指定正确的路径"
    exit 1
fi

# 设置环境变量
export NANOCHAT_OFFLINE_MODE=1
export HF_HUB_OFFLINE=1
export NANOCHAT_BASE_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 清除分布式训练环境变量
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# 激活 conda 环境
if command -v conda &> /dev/null; then
    CONDA_ENV_PATH="/mnt/shared-storage-user/puyuan/conda_envs/nanochat"
    if [ -d "$CONDA_ENV_PATH" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV_PATH" 2>/dev/null || true
    fi
fi

# 显示配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EBT Web Chat Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint: $CKPT_PATH"
echo "温度: $TEMPERATURE | Top-P: $TOP_P | 最大 Tokens: $MAX_TOKENS"
echo "端口: $PORT | 地址: $HOST"
echo "MCMC 显示: ${SHOW_MCMC_FLAG:-关闭} | 详细模式: ${VERBOSE_FLAG:-关闭}"
[ -n "$OVERRIDE_MCMC_STEPS" ] && echo "覆盖 MCMC 步数: $OVERRIDE_MCMC_STEPS"
[ -n "$OVERRIDE_ALPHA" ] && echo "覆盖 Alpha: $OVERRIDE_ALPHA"
[ -n "$OVERRIDE_NOISE_STD" ] && echo "覆盖 Noise Std: $OVERRIDE_NOISE_STD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 构建 override 参数
OVERRIDE_FLAGS=""
[ -n "$OVERRIDE_MCMC_STEPS" ] && OVERRIDE_FLAGS="$OVERRIDE_FLAGS --override-mcmc-steps $OVERRIDE_MCMC_STEPS"
[ -n "$OVERRIDE_NOISE_STD" ] && OVERRIDE_FLAGS="$OVERRIDE_FLAGS --override-noise-std $OVERRIDE_NOISE_STD"
[ -n "$OVERRIDE_ALPHA" ] && OVERRIDE_FLAGS="$OVERRIDE_FLAGS --override-alpha $OVERRIDE_ALPHA"

# 切换到 repo root（绝对 import 需要从 repo root 运行）
cd "$REPO_ROOT"

# 运行 Python Web 服务
PYTHON="${CONDA_ENV_PATH}/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi
$PYTHON -m openebm.elm.scripts.chat_ebt_web \
    --checkpoint "$CKPT_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --port "$PORT" \
    --host "$HOST" \
    $OVERRIDE_FLAGS \
    $SHOW_MCMC_FLAG \
    $VERBOSE_FLAG \
    $SHOW_ENERGY_FLAG \
    $SHOW_DIST_FLAG
