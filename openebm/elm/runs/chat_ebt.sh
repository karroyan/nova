#!/bin/bash
################################################################################
# EBT 交互式对话脚本
#
# 与训练好的 EBT 模型进行交互式对话，支持展示 MCMC 迭代过程
#
# 使用方法:
#   bash runs/chat_ebt.sh                    # 默认模式
#   bash runs/chat_ebt.sh --show-mcmc        # 展示 MCMC 步骤
#   bash runs/chat_ebt.sh --verbose          # 详细模式
#   bash runs/chat_ebt.sh --help             # 显示帮助
#
# 环境变量:
#   CKPT_PATH      - Checkpoint 路径 (可选)
#   TEMPERATURE    - 生成温度 (默认: 0.8)
#   TOP_P          - Top-P 采样 (默认: 0.9)
#   MAX_TOKENS     - 最大生成 tokens (默认: 256)
#   SHOW_MCMC      - 是否展示 MCMC (true/false, 默认: false)
#   VERBOSE        - 详细模式 (true/false, 默认: false)
################################################################################

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
REPO_ROOT="$( cd "$EBT_DIR/../.." && pwd )"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"

# 默认配置
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt"

# base_train bpb 0.81
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt"
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-muon-adamw-0327_20260327_140553_2026-03-27_14-06-11_/last.ckpt"

# base_train bpb 0.85 sft_train 'valid_loss' reached 8.15313 
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-sft_20260331_001308_2026-03-31_00-13-29_/e=epoch=0-s=step=2312-lr5e-05-bs4x8-muon_adamw-valid_loss=valid_loss=8.1531.ckpt"
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints_cp/ebt-d26-sft_20260331_001308_2026-03-31_00-13-29_/last.ckpt"

# sft_train val_loss=2.0933
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260406_222136/e=epoch=0-s=step=62-lr5e-05-bs4x8-muon_adamw-valid_loss=valid_loss=0.5019.ckpt"

# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260407_001616/e=epoch=0-s=step=812-lr5e-05-bs4x16-muon_adamw-valid_loss=valid_loss=1.2609.ckpt"
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260408_230436/periodic-s=step=2999-lr5e-05.ckpt"
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0406-from0327-v2_20260408_230436/periodic-s=step=1499-lr5e-05.ckpt"

# sft_train val_loss=1.88
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0420-from0413_20260420_193427/periodic-s=step=2999-d26-ctx1024.ckpt"
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-sft-0420-from0413_20260420_193427/s=step=1953-d26-ctx1024-lr5e-05-bs2x32-muon_adamw-valid_loss=valid_loss=2.1394.ckpt"

# ctx2048 sft-train/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/chat_ebt.sh
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-20260426/sft_train.v4/checkpoints/s=step=562-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.2264.ckpt"

DEFAULT_CKPT="/mnt/shared-storage-user/luyudong/nova-sft/nova/logs/ebt_runs/d26-ctx2048-20260422/sft_train/checkpoints/s=step=2984-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.5680.ckpt"

# sudoku
# DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-20260430/sft_train.v3/checkpoints/periodic-s=step=399-d26-ctx2048.ckpt"

# sudoku
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-0.6-20260508/sft_train/checkpoints/s=step=2987-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4782.ckpt"

DEFAULT_TOKENIZER="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer"

# 从环境变量读取配置
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER}"
TEMPERATURE="${TEMPERATURE:-0.8}"
# TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.9}"
# MAX_TOKENS="${MAX_TOKENS:-512}"
# MAX_TOKENS="${MAX_TOKENS:-1024}"

MAX_TOKENS="${MAX_TOKENS:-2048}"

# MAX_TOKENS="${MAX_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

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
        --help|-h)
            echo "EBT 交互式对话脚本"
            echo ""
            echo "使用方法:"
            echo "  bash runs/chat_ebt.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --checkpoint PATH          Checkpoint 路径"
            echo "  -t, --temperature VAL          生成温度 (默认: 0.8)"
            echo "  -p, --top-p VAL                Top-P 采样 (默认: 0.9)"
            echo "  --max-tokens VAL               最大生成 tokens (默认: 256)"
            echo "  -m, --show-mcmc                展示 MCMC 步骤过程"
            echo "  -v, --verbose                  详细模式"
            echo "  -e, --show-energy              展示能量值变化"
            echo "  -d, --show-distribution        展示概率分布变化"
            echo "  --override-mcmc-steps N        覆盖 MCMC 步数 (默认: 训练值)"
            echo "  --override-noise-std VAL       覆盖 Langevin 噪声"
            echo "  --override-alpha VAL           覆盖 MCMC 步长 alpha"
            echo "  --dtype TYPE                   数据类型 (float32/bfloat16)"
            echo "  --device DEVICE                设备 (cuda/cpu)"
            echo "  -h, --help                     显示此帮助"
            echo ""
            echo "示例:"
            echo "  bash runs/chat_ebt.sh --show-mcmc --verbose"
            echo "  bash runs/chat_ebt.sh --show-mcmc --override-mcmc-steps 10"
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

# 确保 pip-installed 的 nvidia cuBLAS 优先于系统库，避免版本不兼容导致 CUBLAS_STATUS_INVALID_VALUE
export LD_LIBRARY_PATH=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

# 确保 pip-installed 的 nvidia 库优先于系统库，避免 cuBLAS 版本冲突导致 CUBLAS_STATUS_INVALID_VALUE
export LD_LIBRARY_PATH=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

# 清除分布式训练环境变量
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# 激活 conda 环境 (如果需要)
if command -v conda &> /dev/null; then
    CONDA_ENV_PATH="/mnt/shared-storage-user/puyuan/conda_envs/nanochat"
    if [ -d "$CONDA_ENV_PATH" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV_PATH" 2>/dev/null || true
    fi
fi

# 显示配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EBT 交互式对话"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint: $CKPT_PATH"
echo "温度: $TEMPERATURE | Top-P: $TOP_P | 最大 Tokens: $MAX_TOKENS"
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

# 运行 Python 脚本（显式使用 conda 环境的 Python）
PYTHON="${CONDA_ENV_PATH}/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi
$PYTHON -m openebm.elm.scripts.chat_ebt \
    --checkpoint "$CKPT_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    $OVERRIDE_FLAGS \
    $SHOW_MCMC_FLAG \
    $VERBOSE_FLAG \
    $SHOW_ENERGY_FLAG \
    $SHOW_DIST_FLAG
