#!/usr/bin/env bash
# Local launcher for the direct-unembed TF-head + free-embedding-MCMC xxs run.
# Mirrors launch_xxs_local_free_embed.sh, but sets --tf_head_type direct_unembed
# so the trunk post-norm candidate hidden is projected directly to vocab logits
# without using first-layer prev_embed or extra TF-head blocks.

set -euo pipefail

# ---- Conda env ----
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
    export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH:-}"
else
    echo "[launcher-direct] missing conda init: /root/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi

# ---- WandB offline ----
export WANDB_MODE=offline

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ELM_DIR/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/luyudong/nanochat/.cache/nanochat}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ---- Run name ----
RUN_NAME="${RUN_NAME:-ebt-xxs-directunembed-freeembed-4gpu-tokmatch-bs_256_s1_lr_0.0002}"

# ---- TF-head knobs ----
TF_HEAD_TYPE="${TF_HEAD_TYPE:-direct_unembed}"
TF_HEAD_LAYERS="${TF_HEAD_LAYERS:-1}"
TF_HEAD_N_HEADS="${TF_HEAD_N_HEADS:-0}"
TF_HEAD_FFN_MULT="${TF_HEAD_FFN_MULT:-4.0}"

# ---- Free-embedding-MCMC knobs ----
# Doc §4 noise rescaling: extra multiplier on initial D-dim corruption magnitude.
# F=1.0 is baseline; doc suggests trying 1.5~3.0 to force the model to use context
# instead of doing local nearest-neighbour repair. Defaults conservative (1.0).
FREE_EMBED_NOISE_SCALE="${FREE_EMBED_NOISE_SCALE:-1.0}"

# ---- Training hparams (mirror runs/run_ebt.sh xxs) ----
MODEL_NAME=ebt
MODEL_SIZE=xxs
LR=0.0002
ALPHA=500
ALPHA_LR=1500
# 4 GPUs with 25k steps keeps the same total-token budget as the original
# 2-GPU, 50k-step xxs launcher:
#   2 * 2 batch/device * 2 grad_accum * 256 ctx * 50000 = 102.4M tokens
#   4 * 2 batch/device * 2 grad_accum * 256 ctx * 25000 = 102.4M tokens
# Keep the warmup token budget aligned too:
#   old 2-GPU warmup = 1000 optimizer steps * 2048 tokens/step = 2.048M tokens
#   new 4-GPU warmup = 500 optimizer steps * 4096 tokens/step = 2.048M tokens
MAX_STEP=25000
WARMUP_STEP=500

cd "$ELM_DIR"

echo "[launcher-direct] elm dir                : $ELM_DIR"
echo "[launcher-direct] python                 : $(command -v python)"
echo "[launcher-direct] CUDA_VISIBLE_DEVICES   : $CUDA_VISIBLE_DEVICES"
echo "[launcher-direct] visible GPU count      : $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
echo "[launcher-direct] WANDB_MODE             : $WANDB_MODE"
echo "[launcher-direct] NANOCHAT_BASE_DIR      : $NANOCHAT_BASE_DIR"
echo "[launcher-direct] RUN_NAME               : $RUN_NAME"
echo "[launcher-direct] TF head                : type=$TF_HEAD_TYPE layers=$TF_HEAD_LAYERS n_heads=$TF_HEAD_N_HEADS ffn_mult=$TF_HEAD_FFN_MULT"
echo "[launcher-direct] free_embed_noise_scale : $FREE_EMBED_NOISE_SCALE"
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
    --tf_head_ffn_mult "${TF_HEAD_FFN_MULT}" \
    --free_embedding_mcmc \
    --free_embed_noise_scale "${FREE_EMBED_NOISE_SCALE}"
