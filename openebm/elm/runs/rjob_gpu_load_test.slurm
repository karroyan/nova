#!/bin/bash
#SBATCH --job-name=gpu-load-test
#SBATCH --partition=TDS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0
#SBATCH --output=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.out
#SBATCH --error=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.err

export URL="${URL:-http://127.0.0.1:10050}"
export WORKERS="${WORKERS:-16}"

mkdir -p /mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs

echo "Node: $(hostname)"
echo "Target: $URL"
echo "Workers: $WORKERS"

apptainer exec --bind /mnt:/mnt ~/containers/python311.sif \
    /mnt/petrelfs/lixueyan/nar/nanochat_env/bin/python \
    /mnt/petrelfs/lixueyan/nar/nova/openebm/elm/scripts/gpu_load_test.py \
    --url "$URL" \
    --workers "$WORKERS"
