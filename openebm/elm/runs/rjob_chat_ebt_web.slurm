#!/bin/bash
#SBATCH --job-name=ebt-web-chat
#SBATCH --partition=TDS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0
#SBATCH --output=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.out
#SBATCH --error=/mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs/%x-%j.err

set -e
mkdir -p /mnt/petrelfs/lixueyan/nar/nova/openebm/elm/logs

export PORT=${PORT:-10050}

echo "Node: $(hostname)"
echo "Port: $PORT"

cd /mnt/petrelfs/lixueyan/nar/nova/openebm/elm
apptainer exec --nv --bind /mnt:/mnt ~/containers/python311.sif \
    bash runs/chat_ebt_web.sh --num-gpus 7 --load-workers 112 --batch-size 8 --batch-wait-ms 10 "$@"
