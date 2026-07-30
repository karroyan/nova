#!/usr/bin/env bash
# Local launcher for the TF-head xxs run.
# Same conda/path setup as launch_xxs_local.sh. Hardcodes training hparams from
# runs/run_ebt.sh (xxs / fineweb / ctx256 / bs2x2 / lr 2e-4 / 10k steps) and appends
# the TF-head flags. Keeps the baseline (launch_xxs_local.sh) untouched so A/B is
# just running one script vs the other.

set -euo pipefail

# ---- Conda env (same as launch_xxs_local.sh) ----
# shellcheck disable=SC1091
source /mnt/shared-storage-user/lixueyan/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH:-}"

# ---- WandB offline ----
export WANDB_MODE=offline

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ELM_DIR/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ---- Run name (distinguishes from baseline in WandB / ckpt dirs) ----
RUN_NAME="${RUN_NAME:-ebt-xxs-tfhead-bs_256_s1_lr_0.0002}"

# ---- TF-head knobs (override via env at launch time) ----
TF_HEAD_TYPE="${TF_HEAD_TYPE:-transformer}"
TF_HEAD_LAYERS="${TF_HEAD_LAYERS:-1}"
TF_HEAD_N_HEADS="${TF_HEAD_N_HEADS:-0}"   # 0 = inherit trunk's multiheaded_attention_heads (6 for xxs)
TF_HEAD_FFN_MULT="${TF_HEAD_FFN_MULT:-4.0}"

# ---- Training hparams (mirror runs/run_ebt.sh xxs) ----
MODEL_NAME=ebt
MODEL_SIZE=xxs
LR=0.0002
ALPHA=500
ALPHA_LR=1500
MAX_STEP=50000
WARMUP_STEP=1000

cd "$ELM_DIR"

echo "[launcher-tf] elm dir              : $ELM_DIR"
echo "[launcher-tf] python               : $(command -v python)"
echo "[launcher-tf] CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "[launcher-tf] visible GPU count    : $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "[launcher-tf] WANDB_MODE           : $WANDB_MODE"
echo "[launcher-tf] NANOCHAT_BASE_DIR    : $NANOCHAT_BASE_DIR"
echo "[launcher-tf] RUN_NAME             : $RUN_NAME"
echo "[launcher-tf] TF head              : type=$TF_HEAD_TYPE layers=$TF_HEAD_LAYERS n_heads=$TF_HEAD_N_HEADS ffn_mult=$TF_HEAD_FFN_MULT"
echo

python train.py \
    --run_name "${RUN_NAME}" \
    --modality NLP \
    --model_name "${MODEL_NAME}" \
    --model_size "${MODEL_SIZE}" \
    --pretokenize_dataset \
    --normalize_initial_condition \
    --ebt_type time_embed \
    --denoising_initial_condition random_noise \
    --mcmc_step_size_learnable \
    --mcmc_step_size "${ALPHA}" \
    --mcmc_step_size_lr_multiplier "${ALPHA_LR}" \
    --mcmc_num_steps 2 \
    --context_length 256 \
    --gpus -1 \
    --peak_learning_rate "${LR}" \
    --batch_size_per_device 2 \
    --accumulate_grad_batches 2 \
    --gradient_clip_val 1.0 \
    --weight_decay 0.01 \
    --min_lr_scale 10 \
    --max_steps "${MAX_STEP}" \
    --max_scheduling_steps "${MAX_STEP}" \
    --warm_up_steps "${WARMUP_STEP}" \
    --dataset_name fineweb \
    --val_check_interval 1000 \
    --limit_val_batches 100 \
    --val_sanity 1 \
    --wandb_project nlp_pretrain \
    --log_model_archi \
    --log_every_n_steps 100 \
    --set_matmul_precision medium \
    --wandb_watch \
    --use_tf_head \
    --tf_head_type "${TF_HEAD_TYPE}" \
    --tf_head_layers "${TF_HEAD_LAYERS}" \
    --tf_head_n_heads "${TF_HEAD_N_HEADS}" \
    --tf_head_ffn_mult "${TF_HEAD_FFN_MULT}"
