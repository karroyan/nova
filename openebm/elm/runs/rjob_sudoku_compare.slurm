#!/bin/bash
#SBATCH --job-name=sudoku-compare
#SBATCH --partition=TDS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH --output=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.out
#SBATCH --error=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.err

export PORT=${PORT:-10051}
CKPT="/mnt/petrelfs/lixueyan/nar/EBT_ckpt/sudoku_ckpt/s=step=4424-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss_balanced=valid_loss_balanced=0.2667.ckpt"

mkdir -p /mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs

echo "Node: $(hostname)"
echo "URL:  http://$(hostname -i | awk '{print $1}'):${PORT}"

export NANOCHAT_OFFLINE_MODE=1
export HF_HUB_OFFLINE=1
export NANOCHAT_BASE_DIR="/mnt/petrelfs/lixueyan/nar"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONDA_ENV_PATH="/mnt/petrelfs/lixueyan/nar/nanochat_env"
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}

cd /mnt/petrelfs/lixueyan/nar/nova/openebm/elm
apptainer exec --nv --bind /mnt:/mnt ~/containers/python311.sif \
    /mnt/petrelfs/lixueyan/nar/nanochat_env/bin/python \
    scripts/sudoku_compare.py \
    --port "$PORT" --host 0.0.0.0 \
    --checkpoint "$CKPT" \
    --tokenizer /mnt/petrelfs/lixueyan/nar/tokenizer \
    --device cuda
