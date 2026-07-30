#!/bin/bash
################################################################################
# EBT 实验目录布局工具
#
# 用法: 在训练/评估脚本顶部 source 本文件，然后调用对应的 init 函数。
#
# 所有实验产物统一存放在:
#   $EBT_RUNS_ROOT/<exp_id>/
#     ├── exp_meta.json
#     ├── base_train/{config,checkpoints,logs,status.json}
#     ├── sft_train/{config,checkpoints,logs,status.json}
#     ├── core_eval/<eval_run_id>/{config,results,eval.log,status.json}
#     └── nanochat_gsm8k_eval/<eval_run_id>/{config,master.log,nanochat_shards.log,gsm8k.log}
################################################################################

EBT_RUNS_ROOT="${EBT_RUNS_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs}"
_EXP_LAYOUT_SOURCED=1

# Ensure nanochat sub-packages (nanochat, tasks, scripts, data) are importable
_EXP_LAYOUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EXP_REPO_ROOT="$(cd "$_EXP_LAYOUT_DIR/../../../.." && pwd)"
export PYTHONPATH="${_EXP_REPO_ROOT}/nanochat:${_EXP_REPO_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

_exp_timestamp() {
    date +"%Y%m%d-%H%M"
}

_exp_timestamp_full() {
    date +"%Y-%m-%d %H:%M:%S"
}

_exp_git_info() {
    local out_file="$1"
    local repo_dir="${2:-/mnt/shared-storage-user/puyuan/code/OpenEBM}"
    {
        echo "commit: $(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo 'unknown')"
        echo "branch: $(git -C "$repo_dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
        echo "dirty:  $(git -C "$repo_dir" status --porcelain 2>/dev/null | head -20)"
        echo "date:   $(_exp_timestamp_full)"
    } > "$out_file"
}

_exp_parse_ckpt_step() {
    local ckpt_path="$1"
    local fname
    fname=$(basename "$ckpt_path" .ckpt)
    echo "$fname" | grep -oP 'step=\K[0-9]+' | head -1
}

_exp_parse_ckpt_valid_loss() {
    local ckpt_path="$1"
    local fname
    fname=$(basename "$ckpt_path" .ckpt)
    echo "$fname" | grep -oP 'valid_loss=\K[0-9]+\.[0-9]+' | head -1
}

# ---------------------------------------------------------------------------
# exp_init_base_train
#
# 生成 exp_id，创建 base_train 目录树，写入 config 快照。
#
# 需要的 shell 变量 (调用前必须设置):
#   MODEL_SIZE, CONTEXT_LENGTH, 以及训练超参数变量
#
# 可选环境变量:
#   EXP_ID          — 手动指定 exp_id (跳过自动生成)
#   EXP_TAG         — 附加到 exp_id 的自定义标签
#   DRY_RUN         — 设为 1 时只创建目录和 config，不启动训练
#
# 设置的导出变量:
#   EXP_ID, EXP_DIR, EXP_CKPT_DIR, EXP_LOG_FILE, EXP_WANDB_DIR
# ---------------------------------------------------------------------------
exp_init_base_train() {
    local caller_script="${1:-$0}"
    local optim_tag="${2:-muon_adamw}"

    if [ -z "$EXP_ID" ]; then
        local ts
        ts=$(_exp_timestamp)
        EXP_ID="${MODEL_SIZE}-ctx${CONTEXT_LENGTH}-${optim_tag}-${ts}"
        [ -n "$EXP_TAG" ] && EXP_ID="${EXP_ID}-${EXP_TAG}"
    fi

    export EXP_ID
    export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"
    local stage_dir="${EXP_DIR}/base_train"

    export EXP_CKPT_DIR="${stage_dir}/checkpoints"
    export EXP_LOG_FILE="${stage_dir}/logs/train.log"
    export EXP_WANDB_DIR="${stage_dir}/logs"

    mkdir -p "${stage_dir}/config"
    mkdir -p "${stage_dir}/checkpoints"
    mkdir -p "${stage_dir}/logs"

    cp "$caller_script" "${stage_dir}/config/run_script.sh" 2>/dev/null || true
    _exp_git_info "${stage_dir}/config/git_info.txt"

    _exp_write_meta "$EXP_DIR" "base_train"

    echo "  EXP_ID:       ${EXP_ID}"
    echo "  EXP_DIR:      ${EXP_DIR}"
    echo "  Checkpoints:  ${EXP_CKPT_DIR}"
    echo "  Log file:     ${EXP_LOG_FILE}"
}

# ---------------------------------------------------------------------------
# exp_init_resume
#
# 在 base_train/ 下创建 resume_v{N}/ 子目录。
#
# 需要的变量:
#   EXP_ID          — 必须指定 (从 base_train 继承)
#   RESUME_CKPT     — 续训的 checkpoint 路径
#
# 设置的导出变量:
#   EXP_DIR, EXP_CKPT_DIR, EXP_LOG_FILE, EXP_WANDB_DIR, RESUME_DIR
# ---------------------------------------------------------------------------
exp_init_resume() {
    local caller_script="${1:-$0}"

    if [ -z "$EXP_ID" ]; then
        echo "错误: resume 必须指定 EXP_ID 环境变量"
        exit 1
    fi

    export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"
    local base_dir="${EXP_DIR}/base_train"

    if [ ! -d "$base_dir" ]; then
        echo "错误: base_train 目录不存在: $base_dir"
        echo "请确认 EXP_ID 正确: $EXP_ID"
        exit 1
    fi

    local version=1
    while [ -d "${base_dir}/resume_v${version}" ]; do
        version=$((version + 1))
    done

    export RESUME_DIR="${base_dir}/resume_v${version}"
    export EXP_CKPT_DIR="${base_dir}/checkpoints"
    export EXP_LOG_FILE="${RESUME_DIR}/logs/train.log"
    export EXP_WANDB_DIR="${RESUME_DIR}/logs"

    mkdir -p "${RESUME_DIR}/config"
    mkdir -p "${RESUME_DIR}/logs"

    cp "$caller_script" "${RESUME_DIR}/config/run_script.sh" 2>/dev/null || true
    _exp_git_info "${RESUME_DIR}/config/git_info.txt"

    if [ -n "$RESUME_CKPT" ]; then
        local step valid_loss
        step=$(_exp_parse_ckpt_step "$RESUME_CKPT")
        valid_loss=$(_exp_parse_ckpt_valid_loss "$RESUME_CKPT")
        cat > "${RESUME_DIR}/config/resume_ckpt_ref.json" << EOREF
{
  "ckpt_path": "${RESUME_CKPT}",
  "step": ${step:-null},
  "valid_loss": ${valid_loss:-null},
  "timestamp": "$(_exp_timestamp_full)"
}
EOREF
    fi

    echo "  EXP_ID:       ${EXP_ID}"
    echo "  Resume dir:   ${RESUME_DIR}"
    echo "  Checkpoints:  ${EXP_CKPT_DIR} (shared)"
    echo "  Log file:     ${EXP_LOG_FILE}"
}

# ---------------------------------------------------------------------------
# exp_init_sft
#
# 在 exp_id 目录下创建 sft_train/ 子目录。
#
# 需要的变量:
#   EXP_ID          — 必须指定
#   PRETRAIN_CKPT   — base checkpoint 路径
#
# 设置的导出变量:
#   EXP_DIR, EXP_CKPT_DIR, EXP_LOG_FILE, EXP_WANDB_DIR
# ---------------------------------------------------------------------------
exp_init_sft() {
    local caller_script="${1:-$0}"

    if [ -z "$EXP_ID" ]; then
        echo "错误: SFT 必须指定 EXP_ID 环境变量"
        exit 1
    fi

    export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"
    local stage_dir="${EXP_DIR}/sft_train"

    if [ -d "$stage_dir" ]; then
        local version=2
        while [ -d "${EXP_DIR}/sft_train.v${version}" ]; do
            version=$((version + 1))
        done
        stage_dir="${EXP_DIR}/sft_train.v${version}"
        echo "  注意: sft_train/ 已存在，使用 $(basename $stage_dir)"
    fi

    export EXP_CKPT_DIR="${stage_dir}/checkpoints"
    export EXP_LOG_FILE="${stage_dir}/logs/train.log"
    export EXP_WANDB_DIR="${stage_dir}/logs"

    mkdir -p "${stage_dir}/config"
    mkdir -p "${stage_dir}/checkpoints"
    mkdir -p "${stage_dir}/logs"

    cp "$caller_script" "${stage_dir}/config/run_script.sh" 2>/dev/null || true
    _exp_git_info "${stage_dir}/config/git_info.txt"

    if [ -n "$PRETRAIN_CKPT" ]; then
        local step valid_loss source_exp_id
        step=$(_exp_parse_ckpt_step "$PRETRAIN_CKPT")
        valid_loss=$(_exp_parse_ckpt_valid_loss "$PRETRAIN_CKPT")
        source_exp_id=$(basename "$(dirname "$(dirname "$PRETRAIN_CKPT")")" 2>/dev/null || echo "unknown")
        cat > "${stage_dir}/config/base_ckpt_ref.json" << EOREF
{
  "ckpt_path": "${PRETRAIN_CKPT}",
  "source_exp_id": "${source_exp_id}",
  "step": ${step:-null},
  "valid_loss": ${valid_loss:-null},
  "timestamp": "$(_exp_timestamp_full)"
}
EOREF
    fi

    echo "  EXP_ID:       ${EXP_ID}"
    echo "  SFT dir:      ${stage_dir}"
    echo "  Checkpoints:  ${EXP_CKPT_DIR}"
    echo "  Log file:     ${EXP_LOG_FILE}"
}

# ---------------------------------------------------------------------------
# exp_init_eval
#
# 在 exp_id 目录下创建 core_eval/<eval_run_id>/ 子目录。
#
# 需要的变量:
#   EXP_ID          — 必须指定
#   CKPT_PATH       — 评估的 checkpoint 路径
#
# 设置的导出变量:
#   EXP_DIR, EVAL_RUN_DIR, EVAL_OUTPUT_DIR, EVAL_LOG_FILE
# ---------------------------------------------------------------------------
exp_init_eval() {
    local caller_script="${1:-$0}"

    if [ -z "$EXP_ID" ]; then
        echo "错误: eval 必须指定 EXP_ID 环境变量"
        exit 1
    fi

    export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"

    local step ts eval_run_id
    step=$(_exp_parse_ckpt_step "$CKPT_PATH")
    ts=$(_exp_timestamp)
    eval_run_id="step${step:-unknown}_${ts}"

    export EVAL_RUN_DIR="${EXP_DIR}/core_eval/${eval_run_id}"
    export EVAL_OUTPUT_DIR="${EVAL_RUN_DIR}/results"
    export EVAL_LOG_FILE="${EVAL_RUN_DIR}/eval.log"

    mkdir -p "${EVAL_RUN_DIR}/config"
    mkdir -p "${EVAL_RUN_DIR}/results"

    cp "$caller_script" "${EVAL_RUN_DIR}/config/run_script.sh" 2>/dev/null || true

    if [ -n "$CKPT_PATH" ]; then
        local valid_loss ckpt_dir_name stage
        valid_loss=$(_exp_parse_ckpt_valid_loss "$CKPT_PATH")
        ckpt_dir_name=$(basename "$(dirname "$CKPT_PATH")")
        if echo "$ckpt_dir_name" | grep -q "sft"; then
            stage="sft_train"
        else
            stage="base_train"
        fi
        cat > "${EVAL_RUN_DIR}/config/eval_ckpt_ref.json" << EOREF
{
  "ckpt_path": "${CKPT_PATH}",
  "stage": "${stage}",
  "step": ${step:-null},
  "valid_loss": ${valid_loss:-null},
  "timestamp": "$(_exp_timestamp_full)"
}
EOREF
    fi

    echo "  EXP_ID:       ${EXP_ID}"
    echo "  Eval run:     ${eval_run_id}"
    echo "  Output dir:   ${EVAL_OUTPUT_DIR}"
    echo "  Log file:     ${EVAL_LOG_FILE}"
}

# ---------------------------------------------------------------------------
# exp_init_nanochat_gsm8k_eval
#
# 在 exp_id 目录下创建 nanochat_gsm8k_eval/<eval_run_id>/ 子目录。
#
# 需要的变量:
#   EXP_ID          — 必须指定
#   CKPT_PATH       — 评估的 checkpoint 路径
#
# 设置的导出变量:
#   EXP_DIR, EVAL_RUN_DIR, EVAL_LOG_FILE, NANOCHAT_EVAL_LOG, GSM8K_EVAL_LOG
# ---------------------------------------------------------------------------
exp_init_nanochat_gsm8k_eval() {
    local caller_script="${1:-$0}"

    if [ -z "$EXP_ID" ]; then
        echo "错误: nanochat_gsm8k_eval 必须指定 EXP_ID 环境变量"
        exit 1
    fi

    export EXP_DIR="${EBT_RUNS_ROOT}/${EXP_ID}"

    local step ts eval_run_id
    step=$(_exp_parse_ckpt_step "$CKPT_PATH")
    ts=$(_exp_timestamp)
    eval_run_id="step${step:-unknown}_${ts}"

    export EVAL_RUN_DIR="${EXP_DIR}/nanochat_gsm8k_eval/${eval_run_id}"
    export EVAL_LOG_FILE="${EVAL_RUN_DIR}/master.log"
    export NANOCHAT_EVAL_LOG="${EVAL_RUN_DIR}/nanochat_shards.log"
    export GSM8K_EVAL_LOG="${EVAL_RUN_DIR}/gsm8k.log"

    mkdir -p "${EVAL_RUN_DIR}/config"

    cp "$caller_script" "${EVAL_RUN_DIR}/config/run_script.sh" 2>/dev/null || true

    if [ -n "$CKPT_PATH" ]; then
        local valid_loss ckpt_dir_name stage
        valid_loss=$(_exp_parse_ckpt_valid_loss "$CKPT_PATH")
        ckpt_dir_name=$(basename "$(dirname "$CKPT_PATH")")
        if echo "$ckpt_dir_name" | grep -q "sft"; then
            stage="sft_train"
        else
            stage="base_train"
        fi
        cat > "${EVAL_RUN_DIR}/config/eval_ckpt_ref.json" << EOREF
{
  "ckpt_path": "${CKPT_PATH}",
  "stage": "${stage}",
  "step": ${step:-null},
  "valid_loss": ${valid_loss:-null},
  "timestamp": "$(_exp_timestamp_full)"
}
EOREF
    fi

    echo "  EXP_ID:       ${EXP_ID}"
    echo "  Eval run:     ${eval_run_id}"
    echo "  Run dir:      ${EVAL_RUN_DIR}"
    echo "  Master log:   ${EVAL_LOG_FILE}"
    echo "  NanoChat log: ${NANOCHAT_EVAL_LOG}"
    echo "  GSM8K log:    ${GSM8K_EVAL_LOG}"
}

# ---------------------------------------------------------------------------
# exp_save_hparams
#
# 将关键超参数写入 hparams.json。
# 参数: $1 = 目标目录 (config/ 所在的 stage 目录)
# 后续参数: key=value 对
# ---------------------------------------------------------------------------
exp_save_hparams() {
    local config_dir="$1/config"
    shift
    local out_file="${config_dir}/hparams.json"

    echo "{" > "$out_file"
    local first=true
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$out_file"
        fi
        if echo "$val" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
            printf '  "%s": %s' "$key" "$val" >> "$out_file"
        elif [ "$val" = "true" ] || [ "$val" = "false" ]; then
            printf '  "%s": %s' "$key" "$val" >> "$out_file"
        else
            printf '  "%s": "%s"' "$key" "$val" >> "$out_file"
        fi
    done
    echo "" >> "$out_file"
    echo "}" >> "$out_file"
}

# ---------------------------------------------------------------------------
# exp_save_eval_params
#
# 将评估参数写入 eval_params.json。
# 参数: $1 = EVAL_RUN_DIR
# ---------------------------------------------------------------------------
exp_save_eval_params() {
    local eval_dir="$1"
    shift
    local out_file="${eval_dir}/config/eval_params.json"

    echo "{" > "$out_file"
    local first=true
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$out_file"
        fi
        if echo "$val" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
            printf '  "%s": %s' "$key" "$val" >> "$out_file"
        elif [ "$val" = "true" ] || [ "$val" = "false" ]; then
            printf '  "%s": %s' "$key" "$val" >> "$out_file"
        else
            printf '  "%s": "%s"' "$key" "$val" >> "$out_file"
        fi
    done
    echo "" >> "$out_file"
    echo "}" >> "$out_file"
}

# ---------------------------------------------------------------------------
# exp_save_status
#
# 写入 status.json。
# 参数: $1 = 目标目录, $2 = stage 名, $3 = exit_code
# 可选环境变量: EXP_START_TIME (由 init 函数设置)
# ---------------------------------------------------------------------------
exp_save_status() {
    local target_dir="$1"
    local stage="$2"
    local exit_code="${3:-0}"

    cat > "${target_dir}/status.json" << EOSTATUS
{
  "stage": "${stage}",
  "start_time": "${EXP_START_TIME:-unknown}",
  "end_time": "$(_exp_timestamp_full)",
  "exit_code": ${exit_code},
  "exp_id": "${EXP_ID}"
}
EOSTATUS
}

# ---------------------------------------------------------------------------
# _exp_write_meta
#
# 写入 exp_meta.json (实验根目录)。
# ---------------------------------------------------------------------------
_exp_write_meta() {
    local exp_dir="$1"
    local initial_stage="$2"

    if [ ! -f "${exp_dir}/exp_meta.json" ]; then
        cat > "${exp_dir}/exp_meta.json" << EOMETA
{
  "exp_id": "${EXP_ID}",
  "created": "$(_exp_timestamp_full)",
  "git_commit": "$(git -C /mnt/shared-storage-user/puyuan/code/OpenEBM rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "initial_stage": "${initial_stage}"
}
EOMETA
    fi
}
