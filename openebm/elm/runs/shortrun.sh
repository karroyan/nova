### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
source ../../.venv/ebt/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/nanochat:${REPO_ROOT}:${PYTHONPATH:-}"

### LOG INFO ###
# SBATCH --job-name=ebt-xxs-bs_256_s1_lr_
# SBATCH --output=logs/slurm/nlp/ebt-xxs-bs_256_s1_lr_%A-%a.log
# export RUN_NAME="ebt-d26-bs_256_s1_lr_"
export RUN_NAME="ebt-xxs-bs_256_s1_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"

export WANDB_API_KEY="Your WandB API Key"
mkdir -p logs/slurm/nlp/
module purge

lr=(0.0002)
alpha=(500)
alpha_lr=(1500)
MAX_STEP=100
WARMUP_STEP=10

python train.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 256 \
\
--gpus -1 \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 2 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps ${MAX_STEP} \
--max_scheduling_steps ${MAX_STEP} \
--warm_up_steps ${WARMUP_STEP} \
\
--dataset_name "fineweb" \
--num_workers 12 \
--val_check_interval 50 \
--limit_val_batches 10 \
--val_sanity 1 \
\
--wandb_project 'nlp_pretrain' \
\
--log_model_archi \
--log_every_n_steps 100 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}
