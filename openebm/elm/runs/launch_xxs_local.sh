#!/usr/bin/env bash
# Local launcher that wraps runs/run_ebt.sh for a 2-GPU, no-SLURM, WandB-offline run.
# Does not modify run_ebt.sh. Handles five things the original script assumes:
#   1) conda env `ebt` is active (replaces the hardcoded venv path that doesn't exist)
#   2) cwd is elm/ (so `python train.py` and the source-`../../.venv` line resolve)
#   3) SLURM_ARRAY_TASK_ID is set (so `${lr[$SLURM_ARRAY_TASK_ID]}` indexes correctly)
#   4) The conditional `--is_slurm_run` flag is stripped (its assertion requires
#      limit_val_batches==1, but the script passes 100 -> would AssertionError)
#   5) NANOCHAT_BASE_DIR points at the tokenizer + fineweb shards on disk
#
# Usage:
#   bash runs/launch_xxs_local.sh                 # default: 2 GPUs (0,1)
#   CUDA_VISIBLE_DEVICES=0,1 bash runs/launch_xxs_local.sh
#   RUN_NAME=ebt-xxs-myexp- bash runs/launch_xxs_local.sh

set -euo pipefail

# ---- Conda env ----
# Use luyudong's ebt env (same one rjob uses) — local lixueyan/ebt is missing `lightning`.
# Activate by absolute path; works whether the user has their own miniconda or not.
# shellcheck disable=SC1091
source /mnt/shared-storage-user/lixueyan/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH:-}"

# ---- WandB offline ----
export WANDB_MODE=offline

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EBT="${SCRIPT_DIR}/run_ebt.sh"

# Where tokenizer/tokenizer.pkl and base_data/shard_*.parquet actually live.
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"

# 2 GPUs by default. --gpus -1 in run_ebt.sh asks PL to use all visible CUDA devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Makes `${lr[$SLURM_ARRAY_TASK_ID]}` index = 0; --is_slurm_run is stripped below.
export SLURM_ARRAY_TASK_ID=0

cd "$ELM_DIR"

# Strip flags that current train.py doesn't accept / can't run locally.
# IMPORTANT: keep the temp script in $SCRIPT_DIR (= runs/), because run_ebt.sh derives
# REPO_ROOT from its own BASH_SOURCE location; if we dropped it in /tmp, REPO_ROOT
# would resolve to `/` and PYTHONPATH would be wrong (openebm import fails).
#   - `--is_slurm_run`         : its assertion requires limit_val_batches==1 (we pass 100)
#   - `--num_workers 12`       : removed from train.py (nanochat DataLoader hardcodes it)
TMP_SCRIPT="$(mktemp "${SCRIPT_DIR}/.run_ebt_local.XXXXXX.sh")"
trap 'rm -f "$TMP_SCRIPT"' EXIT
sed \
    -e 's|\${SLURM_ARRAY_TASK_ID:+--is_slurm_run}||' \
    -e '/^--num_workers[[:space:]]/d' \
    "$RUN_EBT" > "$TMP_SCRIPT"

echo "[launcher] elm dir            : $ELM_DIR"
echo "[launcher] python             : $(command -v python)"
echo "[launcher] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[launcher] visible GPU count  : $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "[launcher] WANDB_MODE         : $WANDB_MODE"
echo "[launcher] NANOCHAT_BASE_DIR  : $NANOCHAT_BASE_DIR"
echo "[launcher] running            : $RUN_EBT (--is_slurm_run stripped)"
echo

bash "$TMP_SCRIPT"
