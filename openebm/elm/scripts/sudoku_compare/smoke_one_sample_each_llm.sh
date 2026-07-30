#!/usr/bin/env bash
set -euo pipefail

# Load and evaluate exactly one SATNet-test Sudoku sample for each configured
# LLM. This is intended as a model-loading/vLLM smoke test before running the
# full test split.
#
# Example:
#   bash openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh
#
# Common overrides:
#   PYTHON=/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python
#   SAMPLE_INDEX=0
#   SMALL_MODELS="qwen3_1p7b llama3p2_1b"
#   RUN_R1=0 RUN_QWEN27=0
#   QWEN27_TP=8 R1_TP=8
#   MAX_TOKENS=8192
#   ENFORCE_EAGER=0
#   ATTENTION_BACKEND=FLASH_ATTN
#   FLASHINFER_DISABLE_VERSION_CHECK=1
#   FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
#   LLM_BACKEND=transformers RUN_R1=0 RUN_QWEN27=0

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM}"
DEFAULT_NANOCHAT_PY="/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${DEFAULT_NANOCHAT_PY}" ]]; then
  PYTHON_BIN="${DEFAULT_NANOCHAT_PY}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[smoke_one_sample_each_llm] ERROR: python not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
VLLM_CUDA_VARIANT="${VLLM_CUDA_VARIANT:-cu128}"

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

prepend_cuda_lib_path_for_python "${PYTHON_BIN}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/openebm/elm/data/sudoku_cache_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/openebm/elm/runs/sudoku_compare_smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/llm_one_sample/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${RESULTS_ROOT}/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_ID}}"
STATUS_LOG="${STATUS_LOG:-${LOG_DIR}/status.tsv}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MPLCONFIGDIR

SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
GLOBAL_MAX_TOKENS="${MAX_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"
RESPONSE_LOG="${RESPONSE_LOG:-truncated}"
RESPONSE_LOG_CHARS="${RESPONSE_LOG_CHARS:-12000}"
TRACE_LOG="${TRACE_LOG:-full}"
TRACE_LOG_CHARS="${TRACE_LOG_CHARS:-50000}"
CASE_EXAMPLES_PER_TYPE="${CASE_EXAMPLES_PER_TYPE:-5}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
THINKING="${THINKING:-disable}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LLM_BACKEND="${LLM_BACKEND:-vllm}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_ENV="${VERIFY_ENV:-1}"

# Use FlashInfer by default for non-MLA models. DeepSeek-R1 uses MLA attention,
# so it must not inherit the generic non-MLA FLASHINFER backend.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${VLLM_ATTENTION_BACKEND:-FLASHINFER}}"
SMALL_ATTENTION_BACKEND="${SMALL_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
QWEN27_ATTENTION_BACKEND="${QWEN27_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
R1_ATTENTION_BACKEND="${R1_ATTENTION_BACKEND:-${DEEPSEEK_ATTENTION_BACKEND:-auto}}"
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

GPU_COUNT="${GPU_COUNT:-$(detect_gpu_count "${PYTHON_BIN}")}"
if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]]; then
  GPU_COUNT=0
fi
GPU_IDS="${GPU_IDS:-$(gpu_ids_from_count "${GPU_COUNT}")}"
AUTO_GPU_CONFIG="${AUTO_GPU_CONFIG:-1}"

RUN_SMALL="${RUN_SMALL:-1}"
RUN_QWEN27="${RUN_QWEN27:-1}"
RUN_R1="${RUN_R1:-1}"
SMALL_MODELS="${SMALL_MODELS:-qwen3_1p7b llama3p2_1b}"
QWEN27_MODEL="${QWEN27_MODEL:-qwen3p6_27b}"
R1_MODEL="${R1_MODEL:-deepseek_r1_0528}"

SMALL_MAX_TOKENS="${SMALL_MAX_TOKENS:-${GLOBAL_MAX_TOKENS:-2048}}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-1}"
SMALL_TP="${SMALL_TP:-1}"
SMALL_DP="${SMALL_DP:-1}"
SMALL_MAX_MODEL_LEN="${SMALL_MAX_MODEL_LEN:-}"
SMALL_GPU_MEMORY_UTILIZATION="${SMALL_GPU_MEMORY_UTILIZATION:-0.90}"
SMALL_PARALLEL_STRATEGY="${SMALL_PARALLEL_STRATEGY:-single_sample_single_gpu}"

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
QWEN27_PARALLEL_STRATEGY="${QWEN27_PARALLEL_STRATEGY:-tensor_parallel_smoke}"

R1_MAX_TOKENS="${R1_MAX_TOKENS:-${GLOBAL_MAX_TOKENS:-8192}}"
R1_BATCH_SIZE="${R1_BATCH_SIZE:-1}"
if [[ "${AUTO_GPU_CONFIG}" == "1" ]]; then
  R1_TP="${R1_TP:-$(min_positive "${GPU_COUNT}" 8)}"
else
  R1_TP="${R1_TP:-8}"
fi
R1_DP="${R1_DP:-1}"
R1_MAX_MODEL_LEN="${R1_MAX_MODEL_LEN:-32768}"
# Keep headroom for DeepSeek-R1 warmup/all-reduce buffers after FP8 weights
# and KV cache are allocated.
R1_GPU_MEMORY_UTILIZATION="${R1_GPU_MEMORY_UTILIZATION:-0.85}"
R1_PARALLEL_STRATEGY="${R1_PARALLEL_STRATEGY:-tensor_parallel_smoke}"

if [[ "${GPU_COUNT}" -gt 0 ]]; then
  if [[ "${QWEN27_TP}" -gt "${GPU_COUNT}" ]]; then
    QWEN27_TP="${GPU_COUNT}"
  fi
  if [[ "${R1_TP}" -gt "${GPU_COUNT}" ]]; then
    R1_TP="${GPU_COUNT}"
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
ln -sfn "${LOG_DIR}" "${LOG_ROOT}/latest_one_sample"
ln -sfn "${OUT_DIR}" "${RESULTS_ROOT}/latest_one_sample_outputs"
printf "timestamp\tstage\tevent\trc\tlog\tcmd\n" > "${STATUS_LOG}"
cd "${REPO_ROOT}"

write_run_metadata() {
  {
    echo "RUN_ID=${RUN_ID}"
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "OUT_DIR=${OUT_DIR}"
    echo "LOG_DIR=${LOG_DIR}"
    echo "STATUS_LOG=${STATUS_LOG}"
    echo "SAMPLE_INDEX=${SAMPLE_INDEX}"
    echo "GLOBAL_MAX_TOKENS=${GLOBAL_MAX_TOKENS:-}"
    echo "THINKING=${THINKING}"
    echo "RUN_SMALL=${RUN_SMALL}"
    echo "RUN_QWEN27=${RUN_QWEN27}"
    echo "RUN_R1=${RUN_R1}"
    echo "GPU_COUNT=${GPU_COUNT}"
    echo "GPU_IDS=${GPU_IDS}"
    echo "AUTO_GPU_CONFIG=${AUTO_GPU_CONFIG}"
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
    echo "LLM_BACKEND=${LLM_BACKEND}"
    echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
    echo "DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND}"
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
    echo "# Sudoku Compare Smoke Run ${RUN_ID}"
    echo
    echo "## Directories"
    echo "- logs: ${LOG_DIR}"
    echo "- outputs: ${OUT_DIR}"
    echo
    echo "## Latest Links"
    echo "- ${LOG_ROOT}/latest_one_sample"
    echo "- ${RESULTS_ROOT}/latest_one_sample_outputs"
    echo
    echo "## Trace Files"
    echo "- run_env.txt: resolved configuration"
    echo "- status.tsv: model start/finish events"
    echo "- *.cmd: exact commands"
    echo "- *.log: command output"
    echo "- outputs/<model>/progress.jsonl: batch-level progress and throughput"
    echo "- outputs/<model>/traces/: full raw trajectories and representative cases"
    echo "- summary.txt: compact metric summary when all selected models finish"
  } > "${LOG_DIR}/INDEX.md"
}

verify_env() {
  if [[ "${VERIFY_ENV}" != "1" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  local verify_vllm=0
  if [[ "${LLM_BACKEND}" == "vllm" ]]; then
    verify_vllm=1
  fi
  SUDOKU_COMPARE_VERIFY_VLLM="${verify_vllm}" "${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/verify_env.log"
import os
import sys
print("python", sys.executable)
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print("device", idx, props.name, f"{props.total_memory / (1024**3):.2f}GiB")
import setuptools
import _distutils_hack
print("setuptools", setuptools.__version__)
if os.environ.get("SUDOKU_COMPARE_VERIFY_VLLM") == "1":
    import inspect
    import vllm
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
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
    echo "[smoke_one_sample_each_llm] start ${name} $(date --iso-8601=seconds)"
    cat "${cmd_path}"
    write_status_event "${name}" start "" "${log_path}" "${cmd_path}"
    set +e
    "$@"
    rc=$?
    set -e
    echo "[smoke_one_sample_each_llm] finish ${name} rc=${rc} $(date --iso-8601=seconds)"
    write_status_event "${name}" finish "${rc}" "${log_path}" "${cmd_path}"
    exit "${rc}"
  } 2>&1 | tee "${log_path}"
}

append_common_args() {
  local -n out_ref=$1
  local attention_backend="${2:-${ATTENTION_BACKEND}}"
  out_ref+=(
    --data-dir "${DATA_DIR}"
    --out-dir "${OUT_DIR}"
    --backend "${LLM_BACKEND}"
    --start-index "${SAMPLE_INDEX}"
    --num-samples 1
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --thinking "${THINKING}"
    --overwrite
    --response-log "${RESPONSE_LOG}"
    --response-log-chars "${RESPONSE_LOG_CHARS}"
    --trace-log "${TRACE_LOG}"
    --trace-log-chars "${TRACE_LOG_CHARS}"
    --case-examples-per-type "${CASE_EXAMPLES_PER_TYPE}"
  )
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

run_model_group() {
  local log_name="$1"
  local models_string="$2"
  local max_tokens="$3"
  local batch_size="$4"
  local tp_size="$5"
  local dp_size="$6"
  local max_model_len="$7"
  local gpu_memory_utilization="$8"
  local parallel_strategy="$9"
  local attention_backend="${10:-${ATTENTION_BACKEND}}"

  read -r -a model_array <<< "${models_string}"
  local cmd=("${PYTHON_BIN}" -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku)
  cmd+=(--models "${model_array[@]}")
  cmd+=(--run-summary-name "run_summary_${log_name}.json")
  append_common_args cmd "${attention_backend}"
  cmd+=(
    --max-tokens "${max_tokens}"
    --batch-size "${batch_size}"
    --tensor-parallel-size "${tp_size}"
    --data-parallel-size "${dp_size}"
    --assigned-gpus "${GPU_IDS}"
    --parallel-strategy "${parallel_strategy}"
  )
  cmd+=(--gpu-memory-utilization "${gpu_memory_utilization}")
  if [[ -n "${max_model_len}" ]]; then
    cmd+=(--max-model-len "${max_model_len}")
  fi
  run_cmd_log "${log_name}" "${cmd[@]}"
}

write_run_metadata
write_run_index
echo "[smoke_one_sample_each_llm] logs: ${LOG_DIR}"
echo "[smoke_one_sample_each_llm] latest logs: ${LOG_ROOT}/latest_one_sample"
echo "[smoke_one_sample_each_llm] outputs: ${OUT_DIR}"
echo "[smoke_one_sample_each_llm] latest outputs: ${RESULTS_ROOT}/latest_one_sample_outputs"
echo "[smoke_one_sample_each_llm] gpu_count=${GPU_COUNT} gpu_ids=${GPU_IDS:-<none>} auto_gpu_config=${AUTO_GPU_CONFIG}"
echo "[smoke_one_sample_each_llm] small strategy=${SMALL_PARALLEL_STRATEGY} max_tokens=${SMALL_MAX_TOKENS} batch=${SMALL_BATCH_SIZE} tp=${SMALL_TP} dp=${SMALL_DP}"
echo "[smoke_one_sample_each_llm] qwen27 strategy=${QWEN27_PARALLEL_STRATEGY} max_tokens=${QWEN27_MAX_TOKENS} batch=${QWEN27_BATCH_SIZE} tp=${QWEN27_TP} dp=${QWEN27_DP}"
echo "[smoke_one_sample_each_llm] r1 strategy=${R1_PARALLEL_STRATEGY} max_tokens=${R1_MAX_TOKENS} batch=${R1_BATCH_SIZE} tp=${R1_TP} dp=${R1_DP}"
echo "[smoke_one_sample_each_llm] attention backends: small=${SMALL_ATTENTION_BACKEND} qwen27=${QWEN27_ATTENTION_BACKEND} r1=${R1_ATTENTION_BACKEND}"
verify_env

if [[ "${RUN_SMALL}" == "1" ]]; then
  read -r -a small_model_array <<< "${SMALL_MODELS}"
  for model in "${small_model_array[@]}"; do
    run_model_group "${model}" "${model}" "${SMALL_MAX_TOKENS}" "${SMALL_BATCH_SIZE}" "${SMALL_TP}" "${SMALL_DP}" "${SMALL_MAX_MODEL_LEN}" "${SMALL_GPU_MEMORY_UTILIZATION}" "${SMALL_PARALLEL_STRATEGY}" "${SMALL_ATTENTION_BACKEND}"
  done
fi
if [[ "${RUN_QWEN27}" == "1" ]]; then
  run_model_group "${QWEN27_MODEL}" "${QWEN27_MODEL}" "${QWEN27_MAX_TOKENS}" "${QWEN27_BATCH_SIZE}" "${QWEN27_TP}" "${QWEN27_DP}" "${QWEN27_MAX_MODEL_LEN}" "${QWEN27_GPU_MEMORY_UTILIZATION}" "${QWEN27_PARALLEL_STRATEGY}" "${QWEN27_ATTENTION_BACKEND}"
fi
if [[ "${RUN_R1}" == "1" ]]; then
  run_model_group "${R1_MODEL}" "${R1_MODEL}" "${R1_MAX_TOKENS}" "${R1_BATCH_SIZE}" "${R1_TP}" "${R1_DP}" "${R1_MAX_MODEL_LEN}" "${R1_GPU_MEMORY_UTILIZATION}" "${R1_PARALLEL_STRATEGY}" "${R1_ATTENTION_BACKEND}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<PY | tee "${LOG_DIR}/summary.txt"
import json
from pathlib import Path
root = Path("${OUT_DIR}")
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text())
    print(
        f"{s.get('model')}: n={s.get('n')} "
        f"valid={s.get('valid_rate'):.3f} board={s.get('board_acc'):.3f} "
        f"cell={s.get('cell_acc'):.3f} avg_time={s.get('avg_time'):.3f}s"
    )
PY
fi

echo "[smoke_one_sample_each_llm] done"
