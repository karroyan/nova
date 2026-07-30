#!/bin/bash

################################################################################
# Submit d26 free-embedding direct-unembed Sudoku SFT on 1 node x 8 GPUs.
################################################################################

set -e

RUN_SCRIPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/rjob/run_ebt_freeembed_sudoku_sft_1node_8gpu.sh"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-/mnt/shared-storage-user/luyudong/nova-sft/nova/logs/ebt_runs/d26-ctx2048-20260605-s2-step1-learnablealpha-init300-freeembed-direct/sft_train.v4/checkpoints/s=step=1687-d26-ctx2048-lr0.00024-bs1x32-muon_adamw-valid_loss=valid_loss=0.6590.ckpt}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat}"
RUN_PREFIX="${RUN_PREFIX:-d26-ctx2048-sudoku-sft-freeembed-direct-1node-8gpu}"
EXP_ID="${EXP_ID:-${RUN_PREFIX}-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

rjob submit \
  --name=ebt-d26-freeembed-sudoku-sft-1n8g \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  -P 1 \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  -e DISTRIBUTED_JOB=true \
  -e PRETRAIN_CKPT="${PRETRAIN_CKPT}" \
  -e CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
  -e RUN_PREFIX="${RUN_PREFIX}" \
  -e EXP_ID="${EXP_ID}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "${RUN_SCRIPT}"
