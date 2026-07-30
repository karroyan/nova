#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the SATNet Sudoku EBM-vs-LLM comparison.
#
# Common overrides:
#   DRY_RUN=1 NUM_SAMPLES=16 RUN_R1=0 RUN_QWEN27=0 bash run_all_sudoku_compare.sh
#   PYTHON=/path/to/python RESULTS_ROOT=/path/to/out bash run_all_sudoku_compare.sh
#   EBM_PYTHON=/path/to/python LLM_PYTHON=/path/to/python bash run_all_sudoku_compare.sh
#   ENFORCE_EAGER=0 ATTENTION_BACKEND=FLASH_ATTN bash run_all_sudoku_compare.sh
#   R1_ATTENTION_BACKEND=FLASHMLA bash run_all_sudoku_compare.sh
#   LLM_BACKEND=transformers RUN_R1=0 RUN_QWEN27=0 NUM_SAMPLES=1 bash run_all_sudoku_compare.sh
#
# NUM_SAMPLES=-1 means full test split. Any non-negative value runs a smoke subset.
# STAGE_SIZE=0 keeps the historical one-shot behavior. STAGE_SIZE>0 evaluates
# deterministic cumulative chunks and reports after each stage.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-size|--chunk-size)
      STAGE_SIZE="$2"
      shift 2
      ;;
    --stage-size=*|--chunk-size=*)
      STAGE_SIZE="${1#*=}"
      shift
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --num-samples=*)
      NUM_SAMPLES="${1#*=}"
      shift
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --start-index=*)
      START_INDEX="${1#*=}"
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --run-id=*)
      RUN_ID="${1#*=}"
      shift
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      echo
      echo "Options:"
      echo "  --stage-size N / --chunk-size N   staged evaluation chunk size; 0 disables"
      echo "  --num-samples N                   total samples to cover; -1 means full test split"
      echo "  --start-index N                   first absolute test index"
      echo "  --run-id ID                       output run id"
      exit 0
      ;;
    *)
      echo "[run_all_sudoku_compare] ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM}"
DEFAULT_SHARED_ENV_PY="/mnt/shared-storage-user/puyuan/conda_envs/nanochat-cu128-vllm/bin/python"
DEFAULT_NANOCHAT_PY="/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python"
ALLOW_NANOCHAT_FALLBACK="${ALLOW_NANOCHAT_FALLBACK:-0}"
VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT:-cu128}"
if [[ -n "${PYTHON:-}" ]]; then
  DEFAULT_PYTHON="${PYTHON}"
elif [[ -x "${DEFAULT_NANOCHAT_PY}" ]]; then
  DEFAULT_PYTHON="${DEFAULT_NANOCHAT_PY}"
elif [[ -x "${DEFAULT_SHARED_ENV_PY}" ]]; then
  DEFAULT_PYTHON="${DEFAULT_SHARED_ENV_PY}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  DEFAULT_PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ "${ALLOW_NANOCHAT_FALLBACK}" == "1" ]]; then
  DEFAULT_PYTHON="${DEFAULT_NANOCHAT_PY}"
else
  echo "[run_all_sudoku_compare] ERROR: default Python not found: ${DEFAULT_NANOCHAT_PY}" >&2
  echo "[run_all_sudoku_compare] Or explicitly set PYTHON=/path/to/python." >&2
  exit 2
fi
PYTHON_BIN="${DEFAULT_PYTHON}"
EBM_PYTHON="${EBM_PYTHON:-${PYTHON_BIN}}"
LLM_PYTHON="${LLM_PYTHON:-${PYTHON_BIN}}"
REPORT_PYTHON="${REPORT_PYTHON:-${PYTHON_BIN}}"
for _py_name in EBM_PYTHON LLM_PYTHON REPORT_PYTHON; do
  _py_path="${!_py_name}"
  if [[ ! -x "${_py_path}" ]]; then
    echo "[run_all_sudoku_compare] ERROR: ${_py_name} is not executable: ${_py_path}" >&2
    exit 2
  fi
done

prepend_cuda_lib_path_for_python() {
  local py="$1"
  local cuda_lib
  cuda_lib="$(VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT}" "${py}" - <<'PY'
import os
from pathlib import Path
import sysconfig
purelib = Path(sysconfig.get_paths()["purelib"])
variant = os.environ.get("VLLM_CUDA_VARIANT", "cu128")
if variant == "cu13":
    candidates = [
        purelib / "nvidia" / "cu13" / "lib",
        purelib / "nvidia" / "cuda_runtime" / "lib",
    ]
else:
    candidates = [
        purelib / "nvidia" / "cuda_runtime" / "lib",
        purelib / "nvidia" / "cuda_nvrtc" / "lib",
        purelib / "nvidia" / "nvjitlink" / "lib",
    ]
print(":".join(str(p) for p in candidates if p.is_dir()))
PY
)"
  if [[ -n "${cuda_lib}" ]]; then
    export LD_LIBRARY_PATH="${cuda_lib}:${LD_LIBRARY_PATH:-}"
  fi
}

prepend_cuda_lib_path_for_python "${LLM_PYTHON}"
prepend_cuda_lib_path_for_python "${EBM_PYTHON}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/openebm/elm/data/sudoku_cache_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/openebm/elm/runs/sudoku_compare}"
STAGE_SIZE="${STAGE_SIZE:-${CHUNK_SIZE:-0}}"
if ! [[ "${STAGE_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "[run_all_sudoku_compare] ERROR: STAGE_SIZE must be a non-negative integer, got ${STAGE_SIZE}" >&2
  exit 2
fi
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID="${RUN_ID}"
elif [[ "${STAGE_SIZE}" -gt 0 ]]; then
  RUN_ID="${STAGED_RUN_ID:-staged_${STAGE_SIZE}}"
else
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
PREVIOUS_LATEST_EBM=""
if [[ -e "${RESULTS_ROOT}/latest_ebm" || -L "${RESULTS_ROOT}/latest_ebm" ]]; then
  PREVIOUS_LATEST_EBM="$(readlink -f "${RESULTS_ROOT}/latest_ebm" 2>/dev/null || true)"
fi
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${RESULTS_ROOT}/outputs/${RUN_ID}}"
EBM_OUT_DIR="${EBM_OUT_DIR:-${RUN_OUTPUT_ROOT}/ebm}"
LLM_OUT_DIR="${LLM_OUT_DIR:-${RUN_OUTPUT_ROOT}/llm}"
REPORT_OUT_DIR="${REPORT_OUT_DIR:-${RUN_OUTPUT_ROOT}/reports}"
STAGED_EVAL_PY="${STAGED_EVAL_PY:-${REPO_ROOT}/openebm/elm/scripts/sudoku_compare/staged_eval.py}"
LOG_ROOT="${LOG_ROOT:-${RESULTS_ROOT}/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_ID}}"
STATUS_LOG="${STATUS_LOG:-${LOG_DIR}/status.tsv}"
CONSOLE_LOG="${CONSOLE_LOG:-${LOG_DIR}/console.log}"
CAPTURE_CONSOLE_LOG="${CAPTURE_CONSOLE_LOG:-1}"
LOG_NAME_PREFIX="${LOG_NAME_PREFIX:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MPLCONFIGDIR

EBM_CKPT="${EBM_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-v3p1-20260529/sft_train.v4/checkpoints/s=step=4424-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss_balanced=valid_loss_balanced=0.2667.ckpt}"

RUN_EBM="${RUN_EBM:-1}"
RUN_LLMS="${RUN_LLMS:-1}"
RUN_SMALL="${RUN_SMALL:-1}"
RUN_QWEN27="${RUN_QWEN27:-1}"
RUN_R1="${RUN_R1:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_ENV="${VERIFY_ENV:-1}"
AUTO_REPAIR_SETUPTOOLS="${AUTO_REPAIR_SETUPTOOLS:-1}"
CHECK_EBM_SUMMARY="${CHECK_EBM_SUMMARY:-1}"
if [[ "${STAGE_SIZE}" -gt 0 ]]; then
  CLEAN_EBM_OUT_DIR="${CLEAN_EBM_OUT_DIR:-0}"
else
  CLEAN_EBM_OUT_DIR="${CLEAN_EBM_OUT_DIR:-1}"
fi
REUSE_EBM_RESULTS="${REUSE_EBM_RESULTS:-1}"
EBM_REUSE_DIR="${EBM_REUSE_DIR:-}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"
START_INDEX="${START_INDEX:-0}"
if ! [[ "${NUM_SAMPLES}" =~ ^-?[0-9]+$ ]]; then
  echo "[run_all_sudoku_compare] ERROR: NUM_SAMPLES must be an integer, got ${NUM_SAMPLES}" >&2
  exit 2
fi
if ! [[ "${START_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[run_all_sudoku_compare] ERROR: START_INDEX must be a non-negative integer, got ${START_INDEX}" >&2
  exit 2
fi
EVAL_START_INDEX="${START_INDEX}"
EVAL_NUM_SAMPLES="${NUM_SAMPLES}"
SEED="${SEED:-0}"
EBM_DTYPE="${EBM_DTYPE:-float32}"
EBM_GPUS="${EBM_GPUS:-0}"
GLOBAL_MAX_TOKENS="${MAX_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
RESPONSE_LOG="${RESPONSE_LOG:-truncated}"
RESPONSE_LOG_CHARS="${RESPONSE_LOG_CHARS:-12000}"
TRACE_LOG="${TRACE_LOG:-full}"
TRACE_LOG_CHARS="${TRACE_LOG_CHARS:-50000}"
CASE_EXAMPLES_PER_TYPE="${CASE_EXAMPLES_PER_TYPE:-5}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
THINKING="${THINKING:-disable}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
ANSWER_FORMAT="${ANSWER_FORMAT:-grid}"
STRUCTURED_REGEX="${STRUCTURED_REGEX:-}"
LLM_BACKEND="${LLM_BACKEND:-vllm}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"

# Match the known-good smoke-test runtime defaults. FlashInfer is enabled by
# default through vLLM's explicit attention_config path; eager mode remains
# separate and avoids the current torch.compile API mismatch in nanochat.
# DeepSeek-R1 uses MLA attention, so it must not inherit the generic
# non-MLA FLASHINFER backend. Leave it on vLLM auto-selection unless an
# R1-specific backend is explicitly requested.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${VLLM_ATTENTION_BACKEND:-FLASHINFER}}"
SMALL_ATTENTION_BACKEND="${SMALL_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
QWEN27_ATTENTION_BACKEND="${QWEN27_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
R1_ATTENTION_BACKEND="${R1_ATTENTION_BACKEND:-${DEEPSEEK_ATTENTION_BACKEND:-auto}}"
SMALL_STRUCTURED_REGEX="${SMALL_STRUCTURED_REGEX:-${STRUCTURED_REGEX}}"
QWEN27_STRUCTURED_REGEX="${QWEN27_STRUCTURED_REGEX:-${STRUCTURED_REGEX}}"
R1_STRUCTURED_REGEX="${R1_STRUCTURED_REGEX:-${STRUCTURED_REGEX}}"
SMALL_ANSWER_FORMAT="${SMALL_ANSWER_FORMAT:-${ANSWER_FORMAT}}"
QWEN27_ANSWER_FORMAT="${QWEN27_ANSWER_FORMAT:-${ANSWER_FORMAT}}"
R1_ANSWER_FORMAT="${R1_ANSWER_FORMAT:-${ANSWER_FORMAT}}"
export ATTENTION_BACKEND
unset VLLM_ATTENTION_BACKEND
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/flashinfer}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

detect_gpu_count() {
  local py="$1"
  "${py}" - <<'PY'
import subprocess
try:
    import torch
    count = int(torch.cuda.device_count())
except Exception:
    count = 0
if count <= 0:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        count = len([line for line in out.splitlines() if line.strip()])
    except Exception:
        count = 0
print(count)
PY
}

gpu_ids_from_count() {
  local count="$1"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  if [[ "${count}" -le 0 ]]; then
    echo ""
    return 0
  fi
  local ids=""
  local i
  for ((i=0; i<count; i++)); do
    if [[ -z "${ids}" ]]; then
      ids="${i}"
    else
      ids="${ids},${i}"
    fi
  done
  echo "${ids}"
}

min_positive() {
  local a="$1"
  local b="$2"
  if [[ "${a}" -le 0 ]]; then
    echo 1
  elif [[ "${a}" -lt "${b}" ]]; then
    echo "${a}"
  else
    echo "${b}"
  fi
}

GPU_COUNT="${GPU_COUNT:-$(detect_gpu_count "${LLM_PYTHON}")}"
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]]; then
  GPU_COUNT=0
fi
GPU_IDS="${GPU_IDS:-$(gpu_ids_from_count "${GPU_COUNT}")}"
AUTO_GPU_CONFIG="${AUTO_GPU_CONFIG:-1}"

SMALL_MODELS="${SMALL_MODELS:-qwen3_1p7b llama3p2_1b}"
SMALL_MAX_TOKENS="${SMALL_MAX_TOKENS:-${GLOBAL_MAX_TOKENS:-2048}}"
PARALLEL_SMALL_MODELS="${PARALLEL_SMALL_MODELS:-1}"
if [[ "${AUTO_GPU_CONFIG}" == "1" ]]; then
  SMALL_TP="${SMALL_TP:-1}"
  if [[ "${GPU_COUNT}" -gt 1 ]]; then
    # vLLM offline LLM rejects single-process data_parallel_size>1.
    # Use one process per small model on separate GPUs by default instead.
    SMALL_DP="${SMALL_DP:-1}"
    SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-16}"
    SMALL_PARALLEL_STRATEGY="${SMALL_PARALLEL_STRATEGY:-model_parallel_processes_batched}"
  else
    SMALL_DP="${SMALL_DP:-1}"
    SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-8}"
    SMALL_PARALLEL_STRATEGY="${SMALL_PARALLEL_STRATEGY:-single_process_batched}"
  fi
else
  SMALL_TP="${SMALL_TP:-1}"
  SMALL_DP="${SMALL_DP:-1}"
  SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-1}"
  SMALL_PARALLEL_STRATEGY="${SMALL_PARALLEL_STRATEGY:-manual}"
fi
SMALL_MAX_MODEL_LEN="${SMALL_MAX_MODEL_LEN:-}"
SMALL_GPU_MEMORY_UTILIZATION="${SMALL_GPU_MEMORY_UTILIZATION:-0.90}"

QWEN27_MODEL="${QWEN27_MODEL:-qwen3p6_27b}"
QWEN27_MAX_TOKENS="${QWEN27_MAX_TOKENS:-${GLOBAL_MAX_TOKENS:-4096}}"
QWEN27_BATCH_SIZE="${QWEN27_BATCH_SIZE:-1}"
if [[ "${AUTO_GPU_CONFIG}" == "1" ]]; then
  QWEN27_TP="${QWEN27_TP:-$(min_positive "${GPU_COUNT}" 4)}"
else
  QWEN27_TP="${QWEN27_TP:-4}"
fi
QWEN27_DP="${QWEN27_DP:-1}"
QWEN27_MAX_MODEL_LEN="${QWEN27_MAX_MODEL_LEN:-16384}"
QWEN27_GPU_MEMORY_UTILIZATION="${QWEN27_GPU_MEMORY_UTILIZATION:-0.90}"
QWEN27_PARALLEL_STRATEGY="${QWEN27_PARALLEL_STRATEGY:-tensor_parallel}"

R1_MODEL="${R1_MODEL:-deepseek_r1_0528}"
R1_MAX_TOKENS="${R1_MAX_TOKENS:-${GLOBAL_MAX_TOKENS:-8192}}"
R1_BATCH_SIZE="${R1_BATCH_SIZE:-1}"
if [[ "${AUTO_GPU_CONFIG}" == "1" ]]; then
  R1_TP="${R1_TP:-$(min_positive "${GPU_COUNT}" 8)}"
else
  R1_TP="${R1_TP:-8}"
fi
R1_DP="${R1_DP:-1}"
R1_MAX_MODEL_LEN="${R1_MAX_MODEL_LEN:-32768}"
# DeepSeek-R1 FP8 weights already use roughly 80 GiB/GPU on 8x H200.
# vLLM's KV-cache reservation at 0.95 leaves too little transient headroom
# for warmup/all-reduce buffers, while batch=1 only needs modest concurrency.
R1_GPU_MEMORY_UTILIZATION="${R1_GPU_MEMORY_UTILIZATION:-0.85}"
R1_PARALLEL_STRATEGY="${R1_PARALLEL_STRATEGY:-tensor_parallel}"

if [[ "${GPU_COUNT}" -gt 0 ]]; then
  if [[ $((SMALL_TP * SMALL_DP)) -gt "${GPU_COUNT}" ]]; then
    SMALL_DP=$((GPU_COUNT / SMALL_TP))
    if [[ "${SMALL_DP}" -lt 1 ]]; then
      SMALL_DP=1
    fi
  fi
  if [[ "${QWEN27_TP}" -gt "${GPU_COUNT}" ]]; then
    QWEN27_TP="${GPU_COUNT}"
  fi
  if [[ "${R1_TP}" -gt "${GPU_COUNT}" ]]; then
    R1_TP="${GPU_COUNT}"
  fi
fi

mkdir -p "${LOG_DIR}" "${EBM_OUT_DIR}" "${LLM_OUT_DIR}" "${REPORT_OUT_DIR}"
ln -sfn "${LOG_DIR}" "${LOG_ROOT}/latest"
ln -sfn "${RUN_OUTPUT_ROOT}" "${RESULTS_ROOT}/latest_outputs"
ln -sfn "${EBM_OUT_DIR}" "${RESULTS_ROOT}/latest_ebm"
ln -sfn "${LLM_OUT_DIR}" "${RESULTS_ROOT}/latest_llm"
ln -sfn "${REPORT_OUT_DIR}" "${RESULTS_ROOT}/latest_reports"
printf "timestamp\tstage\tevent\trc\tlog\tcmd\n" > "${STATUS_LOG}"
if [[ "${CAPTURE_CONSOLE_LOG}" == "1" ]]; then
  touch "${CONSOLE_LOG}"
  exec > >(tee -a "${CONSOLE_LOG}") 2>&1
  echo "[run_all_sudoku_compare] console log: ${CONSOLE_LOG}"
fi
cd "${REPO_ROOT}"
case ":${PYTHONPATH:-}:" in
  *":${REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

write_run_metadata() {
  {
    echo "RUN_ID=${RUN_ID}"
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "RESULTS_ROOT=${RESULTS_ROOT}"
    echo "RUN_OUTPUT_ROOT=${RUN_OUTPUT_ROOT}"
    echo "EBM_OUT_DIR=${EBM_OUT_DIR}"
    echo "LLM_OUT_DIR=${LLM_OUT_DIR}"
    echo "REPORT_OUT_DIR=${REPORT_OUT_DIR}"
    echo "STAGED_EVAL_PY=${STAGED_EVAL_PY}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "STATUS_LOG=${STATUS_LOG}"
    echo "CONSOLE_LOG=${CONSOLE_LOG}"
    echo "CAPTURE_CONSOLE_LOG=${CAPTURE_CONSOLE_LOG}"
    echo "PYTHONPATH=${PYTHONPATH:-}"
    echo "EBM_PYTHON=${EBM_PYTHON}"
    echo "LLM_PYTHON=${LLM_PYTHON}"
    echo "REPORT_PYTHON=${REPORT_PYTHON}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "EBM_CKPT=${EBM_CKPT}"
    echo "NUM_SAMPLES=${NUM_SAMPLES}"
    echo "START_INDEX=${START_INDEX}"
    echo "STAGE_SIZE=${STAGE_SIZE}"
    echo "STAGE_MODEL_ORDER=${STAGE_MODEL_ORDER:-${SMALL_MODELS} ${QWEN27_MODEL} ${R1_MODEL} EBM}"
    echo "EVAL_START_INDEX=${EVAL_START_INDEX}"
    echo "EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES}"
    echo "REUSE_EBM_RESULTS=${REUSE_EBM_RESULTS}"
    echo "EBM_REUSE_DIR=${EBM_REUSE_DIR}"
    echo "PREVIOUS_LATEST_EBM=${PREVIOUS_LATEST_EBM}"
    echo "GLOBAL_MAX_TOKENS=${GLOBAL_MAX_TOKENS:-}"
    echo "THINKING=${THINKING}"
    echo "EBM_DTYPE=${EBM_DTYPE}"
    echo "EBM_GPUS=${EBM_GPUS}"
    echo "CLEAN_EBM_OUT_DIR=${CLEAN_EBM_OUT_DIR}"
    echo "AUTO_REPAIR_SETUPTOOLS=${AUTO_REPAIR_SETUPTOOLS}"
    echo "GPU_COUNT=${GPU_COUNT}"
    echo "GPU_IDS=${GPU_IDS}"
    echo "AUTO_GPU_CONFIG=${AUTO_GPU_CONFIG}"
    echo "LLM_BACKEND=${LLM_BACKEND}"
    echo "TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}"
    echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
    echo "DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND}"
    echo "RUN_EBM=${RUN_EBM}"
    echo "RUN_LLMS=${RUN_LLMS}"
    echo "RUN_SMALL=${RUN_SMALL}"
    echo "RUN_QWEN27=${RUN_QWEN27}"
    echo "RUN_R1=${RUN_R1}"
    echo "RUN_REPORTS=${RUN_REPORTS}"
    echo "PARALLEL_SMALL_MODELS=${PARALLEL_SMALL_MODELS}"
    echo "SMALL_MODELS=${SMALL_MODELS}"
    echo "SMALL_MAX_TOKENS=${SMALL_MAX_TOKENS}"
    echo "SMALL_BATCH_SIZE=${SMALL_BATCH_SIZE}"
    echo "SMALL_TP=${SMALL_TP}"
    echo "SMALL_DP=${SMALL_DP}"
    echo "SMALL_PARALLEL_STRATEGY=${SMALL_PARALLEL_STRATEGY}"
    echo "QWEN27_MODEL=${QWEN27_MODEL}"
    echo "QWEN27_MAX_TOKENS=${QWEN27_MAX_TOKENS}"
    echo "QWEN27_BATCH_SIZE=${QWEN27_BATCH_SIZE}"
    echo "QWEN27_TP=${QWEN27_TP}"
    echo "QWEN27_DP=${QWEN27_DP}"
    echo "QWEN27_PARALLEL_STRATEGY=${QWEN27_PARALLEL_STRATEGY}"
    echo "R1_MODEL=${R1_MODEL}"
    echo "R1_MAX_TOKENS=${R1_MAX_TOKENS}"
    echo "R1_BATCH_SIZE=${R1_BATCH_SIZE}"
    echo "R1_TP=${R1_TP}"
    echo "R1_DP=${R1_DP}"
    echo "R1_PARALLEL_STRATEGY=${R1_PARALLEL_STRATEGY}"
    echo "SMALL_ATTENTION_BACKEND=${SMALL_ATTENTION_BACKEND}"
    echo "QWEN27_ATTENTION_BACKEND=${QWEN27_ATTENTION_BACKEND}"
    echo "R1_ATTENTION_BACKEND=${R1_ATTENTION_BACKEND}"
    echo "SYSTEM_PROMPT=${SYSTEM_PROMPT}"
    echo "ANSWER_FORMAT=${ANSWER_FORMAT}"
    echo "SMALL_ANSWER_FORMAT=${SMALL_ANSWER_FORMAT}"
    echo "QWEN27_ANSWER_FORMAT=${QWEN27_ANSWER_FORMAT}"
    echo "R1_ANSWER_FORMAT=${R1_ANSWER_FORMAT}"
    echo "STRUCTURED_REGEX=${STRUCTURED_REGEX}"
    echo "SMALL_STRUCTURED_REGEX=${SMALL_STRUCTURED_REGEX}"
    echo "QWEN27_STRUCTURED_REGEX=${QWEN27_STRUCTURED_REGEX}"
    echo "R1_STRUCTURED_REGEX=${R1_STRUCTURED_REGEX}"
    echo "TRACE_LOG=${TRACE_LOG}"
    echo "TRACE_LOG_CHARS=${TRACE_LOG_CHARS}"
    echo "CASE_EXAMPLES_PER_TYPE=${CASE_EXAMPLES_PER_TYPE}"
    echo "ATTENTION_BACKEND=${ATTENTION_BACKEND:-}"
    echo "FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-}"
    echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-}"
    echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-}"
    echo "VLLM_CUDA_VARIANT=${VLLM_CUDA_VARIANT}"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  } > "${LOG_DIR}/run_env.txt"
}

write_status_event() {
  local stage="$1"
  local event="$2"
  local rc="${3:-}"
  local log_path="${4:-}"
  local cmd_path="${5:-}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date --iso-8601=seconds)" "${stage}" "${event}" "${rc}" "${log_path}" "${cmd_path}" \
    >> "${STATUS_LOG}"
}

write_run_index() {
  {
    echo "# Sudoku Compare Run ${RUN_ID}"
    echo
    echo "## Directories"
    echo "- logs: ${LOG_DIR}"
    echo "- console: ${CONSOLE_LOG}"
    echo "- outputs: ${RUN_OUTPUT_ROOT}"
    echo "- ebm: ${EBM_OUT_DIR}"
    echo "- llm: ${LLM_OUT_DIR}"
    echo "- reports: ${REPORT_OUT_DIR}"
    echo
    echo "## Latest Links"
    echo "- ${LOG_ROOT}/latest"
    echo "- ${RESULTS_ROOT}/latest_outputs"
    echo "- ${RESULTS_ROOT}/latest_ebm"
    echo "- ${RESULTS_ROOT}/latest_llm"
    echo "- ${RESULTS_ROOT}/latest_reports"
    echo
    echo "## Trace Files"
    echo "- run_env.txt: resolved configuration"
    echo "- console.log: full launcher stdout/stderr transcript from setup onward"
    echo "- status.tsv: stage start/finish events"
    echo "- *.cmd: exact commands"
    echo "- *.log: command output"
    echo "- outputs/*/llm/<model>/progress.jsonl: batch-level progress and throughput"
    echo "- outputs/*/llm/<model>/traces/: full raw trajectories and representative cases"
    echo "- staged_manifest.json: staged-mode checkpoint/progress manifest when STAGE_SIZE>0"
    echo "- reports/stages/stage_XXXX/{current,cumulative}/: per-stage aggregate and case-study reports"
  } > "${LOG_DIR}/INDEX.md"
}

repair_setuptools_if_needed() {
  local py="$1"
  local log_path="${LOG_DIR}/repair_setuptools.log"
  if [[ "${AUTO_REPAIR_SETUPTOOLS}" != "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if "${py}" - <<'PY' >/dev/null 2>&1
import setuptools
import _distutils_hack
PY
  then
    return 0
  fi

  {
    echo "[run_all_sudoku_compare] setuptools/_distutils_hack is broken; repairing offline for ${py}"
    "${py}" - <<'PY'
from pathlib import Path
import sys
wheel = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "ensurepip" / "_bundled" / "setuptools-79.0.1-py3-none-any.whl"
print(wheel)
if not wheel.is_file():
    raise SystemExit(f"bundled setuptools wheel not found: {wheel}")
PY
    wheel_path="$("${py}" - <<'PY'
from pathlib import Path
import sys
print(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "ensurepip" / "_bundled" / "setuptools-79.0.1-py3-none-any.whl")
PY
)"
    "${py}" -m pip install --no-index --force-reinstall "${wheel_path}"
    "${py}" - <<'PY'
import setuptools
import _distutils_hack
print("setuptools", setuptools.__version__)
print("_distutils_hack ok")
PY
  } 2>&1 | tee "${log_path}"
}

verify_python_env() {
  local name="$1"
  local py="$2"
  local log_path="${LOG_DIR}/verify_${name}.log"
  {
    echo "[verify:${name}] python=${py}"
    "${py}" - <<'PY'
import os
import sys
print("python", sys.executable)
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                "device",
                idx,
                props.name,
                f"{props.total_memory / (1024**3):.2f}GiB",
            )
except Exception as e:
    print("torch_error", repr(e))
    raise
if os.environ.get("SUDOKU_COMPARE_VERIFY_VLLM") == "1":
    import inspect
    import setuptools
    import _distutils_hack
    import vllm
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    print("setuptools", setuptools.__version__)
    print("vllm", getattr(vllm, "__version__", "unknown"))
    llm_sig = inspect.signature(LLM)
    print("vllm_llm_accepts_kwargs", any(p.kind == p.VAR_KEYWORD for p in llm_sig.parameters.values()))
    print("vllm_engineargs_has_data_parallel_size", hasattr(EngineArgs, "data_parallel_size"))
    print("vllm_offline_dp_requires_external_launcher", True)
    import importlib.metadata as metadata
    for pkg in ["flashinfer-python", "flashinfer-cubin"]:
        try:
            print(pkg, metadata.version(pkg))
        except metadata.PackageNotFoundError:
            print(pkg, "not-installed")
    if os.environ.get("ATTENTION_BACKEND") == "FLASHINFER":
        import flashinfer
        print("flashinfer_import", getattr(flashinfer, "__version__", "unknown"))
PY
  } 2>&1 | tee "${log_path}"
}

verify_sudoku_compare_helpers() {
  local log_path="${LOG_DIR}/verify_sudoku_compare_helpers.log"
  {
    echo "[verify:sudoku_compare_helpers] python=${REPORT_PYTHON}"
    echo "[verify:sudoku_compare_helpers] repo_root=${REPO_ROOT}"
    echo "[verify:sudoku_compare_helpers] staged_eval=${STAGED_EVAL_PY}"
    echo "[verify:sudoku_compare_helpers] PYTHONPATH=${PYTHONPATH:-}"
    "${REPORT_PYTHON}" - <<PY
from pathlib import Path
import importlib.util
import sys

repo = Path("${REPO_ROOT}").resolve()
staged_eval = Path("${STAGED_EVAL_PY}").resolve()
print("cwd", Path.cwd())
print("sys.path[:5]", sys.path[:5])
if not staged_eval.is_file():
    raise SystemExit(f"staged_eval.py not found: {staged_eval}")
spec = importlib.util.spec_from_file_location("sudoku_compare_staged_eval_preflight", staged_eval)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load staged_eval.py spec: {staged_eval}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import openebm
print("openebm", Path(openebm.__file__).resolve())
print("staged_eval", staged_eval)
if repo not in Path(openebm.__file__).resolve().parents:
    raise SystemExit(f"openebm resolved outside REPO_ROOT: {openebm.__file__}")
PY
  } 2>&1 | tee "${log_path}"
}

run_cmd_log() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  local cmd_path="${LOG_DIR}/${name}.cmd"
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } | tee "${cmd_path}"
  write_status_event "${name}" prepared "" "${log_path}" "${cmd_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    write_status_event "${name}" dry_run 0 "${log_path}" "${cmd_path}"
    return 0
  fi
  {
    echo "[run_all_sudoku_compare] start ${name} $(date --iso-8601=seconds)"
    cat "${cmd_path}"
    write_status_event "${name}" start "" "${log_path}" "${cmd_path}"
    set +e
    "$@"
    rc=$?
    set -e
    echo "[run_all_sudoku_compare] finish ${name} rc=${rc} $(date --iso-8601=seconds)"
    write_status_event "${name}" finish "${rc}" "${log_path}" "${cmd_path}"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"
}

run_cmd_log_allow_fail() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  local cmd_path="${LOG_DIR}/${name}.cmd"
  {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  } | tee "${cmd_path}"
  write_status_event "${name}" prepared "" "${log_path}" "${cmd_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    write_status_event "${name}" dry_run 0 "${log_path}" "${cmd_path}"
    return 0
  fi
  local rc=0
  set +e
  {
    echo "[run_all_sudoku_compare] start ${name} $(date --iso-8601=seconds)"
    cat "${cmd_path}"
    write_status_event "${name}" start "" "${log_path}" "${cmd_path}"
    "$@"
    rc=$?
    echo "[run_all_sudoku_compare] finish ${name} rc=${rc} $(date --iso-8601=seconds)"
    write_status_event "${name}" finish "${rc}" "${log_path}" "${cmd_path}"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

append_num_samples_args() {
  local -n out_ref=$1
  out_ref+=(--start-index "${EVAL_START_INDEX}")
  if [[ "${EVAL_NUM_SAMPLES}" != "-1" ]]; then
    out_ref+=(--num-samples "${EVAL_NUM_SAMPLES}")
  fi
}

run_staged_eval() {
  "${REPORT_PYTHON}" "${STAGED_EVAL_PY}" "$@"
}

append_common_llm_args() {
  local -n out_ref=$1
  local attention_backend="${2:-${ATTENTION_BACKEND}}"
  out_ref+=(
    --data-dir "${DATA_DIR}"
    --out-dir "${LLM_OUT_DIR}"
    --backend "${LLM_BACKEND}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --thinking "${THINKING}"
    --resume
    --response-log "${RESPONSE_LOG}"
    --response-log-chars "${RESPONSE_LOG_CHARS}"
    --trace-log "${TRACE_LOG}"
    --trace-log-chars "${TRACE_LOG_CHARS}"
    --case-examples-per-type "${CASE_EXAMPLES_PER_TYPE}"
  )
  append_num_samples_args "$1"
  if [[ -n "${SYSTEM_PROMPT}" ]]; then
    out_ref+=(--system-prompt "${SYSTEM_PROMPT}")
  fi
  if [[ "${SAVE_PROMPTS}" == "1" ]]; then
    out_ref+=(--save-prompts)
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    out_ref+=(--trust-remote-code)
  else
    out_ref+=(--no-trust-remote-code)
  fi
  if [[ "${LLM_BACKEND}" == "vllm" && "${ENFORCE_EAGER}" == "1" ]]; then
    out_ref+=(--enforce-eager)
  fi
  if [[ "${LLM_BACKEND}" == "vllm" && -n "${attention_backend:-}" ]]; then
    out_ref+=(--attention-backend "${attention_backend}")
  fi
  if [[ "${LLM_BACKEND}" == "vllm" && -n "${DISTRIBUTED_EXECUTOR_BACKEND:-}" ]]; then
    out_ref+=(--distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}")
  fi
}

gpu_id_at() {
  local csv="$1"
  local idx="$2"
  local old_ifs="${IFS}"
  local -a _gpu_parts=()
  IFS=","
  read -r -a _gpu_parts <<< "${csv}"
  IFS="${old_ifs}"
  echo "${_gpu_parts[${idx}]:-}"
}

run_llm_group() {
  local log_name="$1"
  local models_string="$2"
  local max_tokens="$3"
  local batch_size="$4"
  local tp_size="$5"
  local dp_size="$6"
  local max_model_len="$7"
  local gpu_memory_utilization="${8:-}"
  local parallel_strategy="${9:-manual}"
  local assigned_gpus="${10:-__all__}"
  local run_summary_name="${11:-run_summary.json}"
  local attention_backend="${12:-${ATTENTION_BACKEND}}"
  local structured_regex="${13:-${STRUCTURED_REGEX}}"
  local answer_format="${14:-${ANSWER_FORMAT}}"
  local actual_assigned_gpus="${GPU_IDS}"

  read -r -a model_array <<< "${models_string}"
  local cmd=()
  if [[ -n "${assigned_gpus}" && "${assigned_gpus}" != "__all__" ]]; then
    actual_assigned_gpus="${assigned_gpus}"
    cmd=(env "CUDA_VISIBLE_DEVICES=${assigned_gpus}")
  fi
  cmd+=("${LLM_PYTHON}" -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku)
  cmd+=(--models "${model_array[@]}")
  cmd+=(--run-summary-name "${run_summary_name}")
  append_common_llm_args cmd "${attention_backend}"
  cmd+=(--answer-format "${answer_format}")
  cmd+=(
    --max-tokens "${max_tokens}"
    --batch-size "${batch_size}"
    --tensor-parallel-size "${tp_size}"
    --data-parallel-size "${dp_size}"
    --assigned-gpus "${actual_assigned_gpus}"
    --parallel-strategy "${parallel_strategy}"
  )
  if [[ -n "${max_model_len}" ]]; then
    cmd+=(--max-model-len "${max_model_len}")
  fi
  if [[ -n "${gpu_memory_utilization}" ]]; then
    cmd+=(--gpu-memory-utilization "${gpu_memory_utilization}")
  fi
  if [[ -n "${structured_regex}" ]]; then
    cmd+=(--structured-regex "${structured_regex}")
  fi
  run_cmd_log "${LOG_NAME_PREFIX}${log_name}" "${cmd[@]}"
}

run_small_models() {
  read -r -a small_model_array <<< "${SMALL_MODELS}"
  local model_count="${#small_model_array[@]}"
  if [[ "${PARALLEL_SMALL_MODELS}" == "1" \
      && "${model_count}" -gt 1 \
      && "${GPU_COUNT}" -ge "${model_count}" \
      && -n "${GPU_IDS}" \
      && "${SMALL_TP}" == "1" \
      && "${SMALL_DP}" == "1" ]]; then
    echo "[run_all_sudoku_compare] running small models concurrently across GPUs"
    local pids=()
    local idx=0
    local model
    for model in "${small_model_array[@]}"; do
      local gpu
      gpu="$(gpu_id_at "${GPU_IDS}" "${idx}")"
      if [[ -z "${gpu}" ]]; then
        echo "[run_all_sudoku_compare] WARNING: missing GPU id for ${model}; falling back to sequential small-model run" >&2
        run_llm_group llm_small "${SMALL_MODELS}" "${SMALL_MAX_TOKENS}" "${SMALL_BATCH_SIZE}" "${SMALL_TP}" "${SMALL_DP}" "${SMALL_MAX_MODEL_LEN}" "${SMALL_GPU_MEMORY_UTILIZATION}" "${SMALL_PARALLEL_STRATEGY}" "__all__" "run_summary.json" "${SMALL_ATTENTION_BACKEND}" "${SMALL_STRUCTURED_REGEX}" "${SMALL_ANSWER_FORMAT}"
        return
      fi
      (
        run_llm_group "llm_small_${model}" "${model}" "${SMALL_MAX_TOKENS}" "${SMALL_BATCH_SIZE}" "${SMALL_TP}" "${SMALL_DP}" "${SMALL_MAX_MODEL_LEN}" "${SMALL_GPU_MEMORY_UTILIZATION}" "model_parallel_process_gpu_${gpu}" "${gpu}" "run_summary_${model}.json" "${SMALL_ATTENTION_BACKEND}" "${SMALL_STRUCTURED_REGEX}" "${SMALL_ANSWER_FORMAT}"
      ) &
      pids+=("$!")
      idx=$((idx + 1))
    done
    local rc=0
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        rc=1
      fi
    done
    if [[ "${rc}" != "0" ]]; then
      echo "[run_all_sudoku_compare] ERROR: at least one parallel small-model process failed" >&2
      exit "${rc}"
    fi
  else
    run_llm_group llm_small "${SMALL_MODELS}" "${SMALL_MAX_TOKENS}" "${SMALL_BATCH_SIZE}" "${SMALL_TP}" "${SMALL_DP}" "${SMALL_MAX_MODEL_LEN}" "${SMALL_GPU_MEMORY_UTILIZATION}" "${SMALL_PARALLEL_STRATEGY}" "__all__" "run_summary.json" "${SMALL_ATTENTION_BACKEND}" "${SMALL_STRUCTURED_REGEX}" "${SMALL_ANSWER_FORMAT}"
  fi
}

check_ebm_summary() {
  if [[ "${CHECK_EBM_SUMMARY}" != "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${REPORT_PYTHON}" - <<PY
import json
from pathlib import Path
path = Path("${EBM_OUT_DIR}") / "results" / "summary.json"
if not path.is_file():
    raise SystemExit(f"EBM summary not found: {path}")
with path.open() as f:
    summary = json.load(f)
test = summary.get("test") or {}
n = int(test.get("n", 0) or 0)
parsed = int(test.get("parsed", 0) or 0)
solved = int(test.get("fully_solved", 0) or 0)
filled_acc = float(test.get("filled_cell_acc", 0.0) or 0.0)
print(f"[run_all_sudoku_compare] EBM summary n={n} parsed={parsed} solved={solved} filled_cell_acc={filled_acc:.6f}")
if n > 0 and parsed == 0:
    raise SystemExit(
        "EBM eval produced zero parsed samples. This usually means every sample "
        "hit an exception; inspect ${LOG_DIR}/ebm_test.log"
    )
PY
}

get_test_total() {
  run_staged_eval total \
    --data-dir "${DATA_DIR}" \
    --split test
}

maybe_clean_ebm_out_dir() {
  if [[ "${RUN_EBM}" != "1" || "${CLEAN_EBM_OUT_DIR}" != "1" ]]; then
    return 0
  fi
  echo "[run_all_sudoku_compare] cleaning EBM_OUT_DIR before rerun: ${EBM_OUT_DIR}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ "${EBM_OUT_DIR}" == "${RUN_OUTPUT_ROOT}" ]]; then
    echo "[run_all_sudoku_compare] ERROR: refusing to clean RUN_OUTPUT_ROOT as EBM_OUT_DIR: ${EBM_OUT_DIR}" >&2
    exit 2
  fi
  case "${EBM_OUT_DIR}/" in
    "${RUN_OUTPUT_ROOT}/"*) ;;
    *)
      echo "[run_all_sudoku_compare] ERROR: refusing to clean EBM_OUT_DIR outside RUN_OUTPUT_ROOT: ${EBM_OUT_DIR}" >&2
      exit 2
      ;;
  esac
  rm -rf "${EBM_OUT_DIR}"
  mkdir -p "${EBM_OUT_DIR}"
}

run_ebm_eval_current_range() {
  local log_name="${1:-ebm_test}"
  local keep_resume="${2:-0}"
  if [[ "${RUN_EBM}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${EBM_CKPT}" ]]; then
    echo "[run_all_sudoku_compare] ERROR: EBM checkpoint not found: ${EBM_CKPT}" >&2
    exit 2
  fi
  local ebm_sample_args=()
  if [[ "${EVAL_NUM_SAMPLES}" == "-1" ]]; then
    ebm_sample_args=(--full_test)
  else
    ebm_sample_args=(--num_samples_test "${EVAL_NUM_SAMPLES}")
  fi
  ebm_sample_args+=(--start_index_test "${EVAL_START_INDEX}")
  if [[ "${keep_resume}" == "1" ]]; then
    ebm_sample_args+=(--resume --keep_shards)
  fi
  run_cmd_log "${log_name}" \
    "${EBM_PYTHON}" -m openebm.elm.scripts.eval_sudoku_samples \
    -c "${EBM_CKPT}" \
    --data_dir "${DATA_DIR}" \
    --splits test \
    --dtype "${EBM_DTYPE}" \
    --gpus "${EBM_GPUS}" \
    --no_per_sample_print \
    --out_dir "${EBM_OUT_DIR}" \
    "${ebm_sample_args[@]}"
  check_ebm_summary
}

run_llms_current_range() {
  if [[ "${RUN_LLMS}" != "1" ]]; then
    return 0
  fi
  if [[ "${RUN_SMALL}" == "1" ]]; then
    run_small_models
  fi
  if [[ "${RUN_QWEN27}" == "1" ]]; then
    run_llm_group llm_qwen27 "${QWEN27_MODEL}" "${QWEN27_MAX_TOKENS}" "${QWEN27_BATCH_SIZE}" "${QWEN27_TP}" "${QWEN27_DP}" "${QWEN27_MAX_MODEL_LEN}" "${QWEN27_GPU_MEMORY_UTILIZATION}" "${QWEN27_PARALLEL_STRATEGY}" "__all__" "run_summary_${QWEN27_MODEL}.json" "${QWEN27_ATTENTION_BACKEND}" "${QWEN27_STRUCTURED_REGEX}" "${QWEN27_ANSWER_FORMAT}"
  fi
  if [[ "${RUN_R1}" == "1" ]]; then
    run_llm_group llm_r1 "${R1_MODEL}" "${R1_MAX_TOKENS}" "${R1_BATCH_SIZE}" "${R1_TP}" "${R1_DP}" "${R1_MAX_MODEL_LEN}" "${R1_GPU_MEMORY_UTILIZATION}" "${R1_PARALLEL_STRATEGY}" "__all__" "run_summary_${R1_MODEL}.json" "${R1_ATTENTION_BACKEND}" "${R1_STRUCTURED_REGEX}" "${R1_ANSWER_FORMAT}"
  fi
}

run_aggregate_reports() {
  if [[ "${RUN_REPORTS}" != "1" ]]; then
    return 0
  fi
  run_cmd_log "${LOG_NAME_PREFIX}aggregate" \
    "${REPORT_PYTHON}" -m openebm.elm.scripts.sudoku_compare.aggregate_results \
    --results-root "${RUN_OUTPUT_ROOT}" \
    --ebm-result-dir "${EBM_OUT_DIR}" \
    --llm-root "${LLM_OUT_DIR}" \
    --csv-out "${REPORT_OUT_DIR}/comparison.csv"

  run_cmd_log "${LOG_NAME_PREFIX}case_study" \
    "${REPORT_PYTHON}" -m openebm.elm.scripts.sudoku_compare.case_study \
    --results-root "${RUN_OUTPUT_ROOT}" \
    --ebm-result-dir "${EBM_OUT_DIR}" \
    --llm-root "${LLM_OUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --out-md "${REPORT_OUT_DIR}/case_study.md"
}

stage_manifest_event() {
  local stage_idx="$1"
  local stage_start="$2"
  local stage_end="$3"
  local event="$4"
  local phase="$5"
  local status="$6"
  local models_string="${7:-}"
  local log_path="${8:-}"
  local -a models=()
  if [[ -n "${models_string}" ]]; then
    read -r -a models <<< "${models_string}"
  fi
  local manifest_log="${LOG_DIR}/${LOG_NAME_PREFIX}stage_manifest.log"
  local rc=0
  set +e
  {
    echo "[run_all_sudoku_compare] manifest stage=${stage_idx} event=${event} phase=${phase} status=${status} models=${models_string}"
    run_staged_eval stage-event \
      --manifest "${RUN_OUTPUT_ROOT}/staged_manifest.json" \
      --run-id "${RUN_ID}" \
      --stage-size "${STAGE_SIZE}" \
      --start-index "${START_INDEX}" \
      --target-total "${STAGED_TARGET_TOTAL:-0}" \
      --total-test "${STAGED_TOTAL_TEST:-0}" \
      --stage-index "${stage_idx}" \
      --stage-start "${stage_start}" \
      --stage-end "${stage_end}" \
      --event "${event}" \
      --phase "${phase}" \
      --status "${status}" \
      --models "${models[@]}" \
      --log "${log_path}"
  } 2>&1 | tee -a "${manifest_log}"
  rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

ebm_source_candidates() {
  local -a raw=()
  if [[ -n "${EBM_REUSE_DIR}" ]]; then
    raw+=("${EBM_REUSE_DIR}")
  fi
  if [[ -n "${PREVIOUS_LATEST_EBM}" && "${PREVIOUS_LATEST_EBM}" != "${EBM_OUT_DIR}" ]]; then
    raw+=("${PREVIOUS_LATEST_EBM}")
  fi
  raw+=("${EBM_OUT_DIR}")
  local -a seen=()
  local item existing duplicate
  for item in "${raw[@]}"; do
    [[ -z "${item}" ]] && continue
    duplicate=0
    for existing in "${seen[@]}"; do
      if [[ "${existing}" == "${item}" ]]; then
        duplicate=1
        break
      fi
    done
    if [[ "${duplicate}" == "0" ]]; then
      seen+=("${item}")
    fi
  done
  printf '%s\n' "${seen[@]}"
}

try_materialize_ebm_view() {
  local stage_idx="$1"
  local end_index="$2"
  if [[ "${REUSE_EBM_RESULTS}" != "1" ]]; then
    return 1
  fi
  local -a candidates=()
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  done < <(ebm_source_candidates)
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    return 1
  fi
  run_cmd_log_allow_fail "${LOG_NAME_PREFIX}ebm_materialize" \
    "${REPORT_PYTHON}" "${STAGED_EVAL_PY}" materialize-ebm \
    --source-dirs "${candidates[@]}" \
    --out-dir "${EBM_OUT_DIR}" \
    --start-index "${START_INDEX}" \
    --end-index "${end_index}" \
    --stage-index "${stage_idx}" \
    --params "${DEFAULT_EBM_PARAMS:-~1B}"
}

run_stage_reports() {
  local stage_idx="$1"
  local stage_start="$2"
  local stage_end="$3"
  if [[ "${RUN_REPORTS}" != "1" ]]; then
    return 0
  fi
  local -a order=()
  read -r -a order <<< "${STAGE_MODEL_ORDER:-${SMALL_MODELS} ${QWEN27_MODEL} ${R1_MODEL} EBM}"
  run_cmd_log "${LOG_NAME_PREFIX}stage_report" \
    "${REPORT_PYTHON}" "${STAGED_EVAL_PY}" report \
    --data-dir "${DATA_DIR}" \
    --out-dir "${REPORT_OUT_DIR}" \
    --ebm-dir "${EBM_OUT_DIR}" \
    --llm-root "${LLM_OUT_DIR}" \
    --stage-index "${stage_idx}" \
    --stage-start "${stage_start}" \
    --stage-end "${stage_end}" \
    --cumulative-start "${START_INDEX}" \
    --cumulative-end "${stage_end}" \
    --cases-per-type "${CASE_EXAMPLES_PER_TYPE}" \
    --model-order "${order[@]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    ln -sfn "${REPORT_OUT_DIR}/stages/stage_$(printf '%04d' "${stage_idx}")/current" "${REPORT_OUT_DIR}/latest_stage_current"
    ln -sfn "${REPORT_OUT_DIR}/stages/stage_$(printf '%04d' "${stage_idx}")/cumulative" "${REPORT_OUT_DIR}/latest_stage_cumulative"
  fi
}

run_monolithic() {
  maybe_clean_ebm_out_dir
  EVAL_START_INDEX="${START_INDEX}"
  EVAL_NUM_SAMPLES="${NUM_SAMPLES}"
  run_ebm_eval_current_range ebm_test 0
  run_llms_current_range
  run_aggregate_reports
}

run_staged() {
  STAGED_TOTAL_TEST="$(get_test_total)"
  if [[ "${START_INDEX}" -ge "${STAGED_TOTAL_TEST}" ]]; then
    echo "[run_all_sudoku_compare] ERROR: START_INDEX=${START_INDEX} >= test split size ${STAGED_TOTAL_TEST}" >&2
    exit 2
  fi
  if [[ "${NUM_SAMPLES}" -lt 0 ]]; then
    STAGED_TARGET_TOTAL=$((STAGED_TOTAL_TEST - START_INDEX))
  else
    STAGED_TARGET_TOTAL="${NUM_SAMPLES}"
    local max_available=$((STAGED_TOTAL_TEST - START_INDEX))
    if [[ "${STAGED_TARGET_TOTAL}" -gt "${max_available}" ]]; then
      STAGED_TARGET_TOTAL="${max_available}"
    fi
  fi
  if [[ "${STAGED_TARGET_TOTAL}" -le 0 ]]; then
    echo "[run_all_sudoku_compare] ERROR: staged target has no samples" >&2
    exit 2
  fi
  local stage_count=$(((STAGED_TARGET_TOTAL + STAGE_SIZE - 1) / STAGE_SIZE))
  maybe_clean_ebm_out_dir
  echo "[run_all_sudoku_compare] staged mode: stage_size=${STAGE_SIZE} stages=${stage_count} start=${START_INDEX} target=${STAGED_TARGET_TOTAL} total_test=${STAGED_TOTAL_TEST}"
  local stage_idx rel_start rel_end stage_start stage_end cumulative_count
  for ((stage_idx=1; stage_idx<=stage_count; stage_idx++)); do
    rel_start=$(((stage_idx - 1) * STAGE_SIZE))
    rel_end=$((stage_idx * STAGE_SIZE))
    if [[ "${rel_end}" -gt "${STAGED_TARGET_TOTAL}" ]]; then
      rel_end="${STAGED_TARGET_TOTAL}"
    fi
    stage_start=$((START_INDEX + rel_start))
    stage_end=$((START_INDEX + rel_end))
    cumulative_count=$((stage_end - START_INDEX))
    LOG_NAME_PREFIX="stage_$(printf '%04d' "${stage_idx}")_"
    EVAL_START_INDEX="${START_INDEX}"
    EVAL_NUM_SAMPLES="${cumulative_count}"
    echo "[run_all_sudoku_compare] stage ${stage_idx}/${stage_count}: current=[${stage_start},${stage_end}) cumulative=[${START_INDEX},${stage_end}) cumulative_count=${cumulative_count}"
    stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start stage running ""

    if [[ "${RUN_LLMS}" == "1" && "${RUN_SMALL}" == "1" ]]; then
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start llm_small running "${SMALL_MODELS}"
      run_small_models
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish llm_small success "${SMALL_MODELS}"
    fi
    if [[ "${RUN_LLMS}" == "1" && "${RUN_QWEN27}" == "1" ]]; then
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start llm_qwen27 running "${QWEN27_MODEL}"
      run_llm_group llm_qwen27 "${QWEN27_MODEL}" "${QWEN27_MAX_TOKENS}" "${QWEN27_BATCH_SIZE}" "${QWEN27_TP}" "${QWEN27_DP}" "${QWEN27_MAX_MODEL_LEN}" "${QWEN27_GPU_MEMORY_UTILIZATION}" "${QWEN27_PARALLEL_STRATEGY}" "__all__" "run_summary_${QWEN27_MODEL}.json" "${QWEN27_ATTENTION_BACKEND}" "${QWEN27_STRUCTURED_REGEX}" "${QWEN27_ANSWER_FORMAT}"
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish llm_qwen27 success "${QWEN27_MODEL}"
    fi
    if [[ "${RUN_LLMS}" == "1" && "${RUN_R1}" == "1" ]]; then
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start llm_r1 running "${R1_MODEL}"
      run_llm_group llm_r1 "${R1_MODEL}" "${R1_MAX_TOKENS}" "${R1_BATCH_SIZE}" "${R1_TP}" "${R1_DP}" "${R1_MAX_MODEL_LEN}" "${R1_GPU_MEMORY_UTILIZATION}" "${R1_PARALLEL_STRATEGY}" "__all__" "run_summary_${R1_MODEL}.json" "${R1_ATTENTION_BACKEND}" "${R1_STRUCTURED_REGEX}" "${R1_ANSWER_FORMAT}"
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish llm_r1 success "${R1_MODEL}"
    fi

    if [[ "${RUN_EBM}" == "1" ]]; then
      stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start ebm running "EBM"
      if try_materialize_ebm_view "${stage_idx}" "${stage_end}"; then
        stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" skip ebm reused "EBM"
      else
        echo "[run_all_sudoku_compare] no reusable EBM rows for [${START_INDEX},${stage_end}); running EBM cumulative eval"
        run_ebm_eval_current_range ebm_test 1
        stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish ebm success "EBM"
      fi
    fi

    stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" start reports running ""
    run_stage_reports "${stage_idx}" "${stage_start}" "${stage_end}"
    stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish reports success ""
    stage_manifest_event "${stage_idx}" "${stage_start}" "${stage_end}" finish stage success ""
  done
  LOG_NAME_PREFIX=""
}

write_run_metadata
write_run_index
echo "[run_all_sudoku_compare] logs: ${LOG_DIR}"
echo "[run_all_sudoku_compare] latest logs: ${LOG_ROOT}/latest"
echo "[run_all_sudoku_compare] outputs: ${RUN_OUTPUT_ROOT}"
echo "[run_all_sudoku_compare] latest outputs: ${RESULTS_ROOT}/latest_outputs"
echo "[run_all_sudoku_compare] gpu_count=${GPU_COUNT} gpu_ids=${GPU_IDS:-<none>} auto_gpu_config=${AUTO_GPU_CONFIG}"
echo "[run_all_sudoku_compare] stage_size=${STAGE_SIZE} num_samples=${NUM_SAMPLES} start_index=${START_INDEX} reuse_ebm=${REUSE_EBM_RESULTS} ebm_reuse_dir=${EBM_REUSE_DIR:-<auto>}"
echo "[run_all_sudoku_compare] small strategy=${SMALL_PARALLEL_STRATEGY} max_tokens=${SMALL_MAX_TOKENS} batch=${SMALL_BATCH_SIZE} tp=${SMALL_TP} dp=${SMALL_DP}"
echo "[run_all_sudoku_compare] qwen27 strategy=${QWEN27_PARALLEL_STRATEGY} max_tokens=${QWEN27_MAX_TOKENS} batch=${QWEN27_BATCH_SIZE} tp=${QWEN27_TP} dp=${QWEN27_DP}"
echo "[run_all_sudoku_compare] r1 strategy=${R1_PARALLEL_STRATEGY} max_tokens=${R1_MAX_TOKENS} batch=${R1_BATCH_SIZE} tp=${R1_TP} dp=${R1_DP}"
echo "[run_all_sudoku_compare] attention backends: small=${SMALL_ATTENTION_BACKEND} qwen27=${QWEN27_ATTENTION_BACKEND} r1=${R1_ATTENTION_BACKEND}"
echo "[run_all_sudoku_compare] answer format: small=${SMALL_ANSWER_FORMAT} qwen27=${QWEN27_ANSWER_FORMAT} r1=${R1_ANSWER_FORMAT}"
echo "[run_all_sudoku_compare] structured regex: small=${SMALL_STRUCTURED_REGEX:-<none>} qwen27=${QWEN27_STRUCTURED_REGEX:-<none>} r1=${R1_STRUCTURED_REGEX:-<none>}"
verify_sudoku_compare_helpers
repair_setuptools_if_needed "${LLM_PYTHON}"
if [[ "${VERIFY_ENV}" == "1" && "${DRY_RUN}" != "1" ]]; then
  verify_python_env ebm "${EBM_PYTHON}"
  if [[ "${LLM_BACKEND}" == "vllm" ]]; then
    SUDOKU_COMPARE_VERIFY_VLLM=1 verify_python_env llm "${LLM_PYTHON}"
  else
    verify_python_env llm "${LLM_PYTHON}"
  fi
fi

if [[ "${STAGE_SIZE}" -gt 0 ]]; then
  run_staged
else
  run_monolithic
fi

echo "[run_all_sudoku_compare] done. Outputs: ${RUN_OUTPUT_ROOT}"
