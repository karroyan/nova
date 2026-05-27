#!/bin/bash

################################################################################
# EBT d26 blockwise + TF-MTP-head (tfmtphead L1) 提交脚本 - 1节点 × 8卡
#
# 用法:
#   bash rjob_ebt_blockwise_tfmtphead_1node_8gpu.sh
#
# 可选环境变量 (将透传到容器内的 run script):
#   BLOCK_MODE=blockwise|mtp_mcmc|future_latent_non_causal|future_latent_bidirectional
#   TRAIN_BLOCK_SIZE=2
#   BLOCK_LATENT_HEAD_TYPE=per_offset_tf_transformer    (默认; 也可 per_offset_tf_linear / per_offset_transformer / ...)
#   BLOCK_LATENT_HEAD_LAYERS=1
#   CONTEXT_LENGTH=1024
#   RUN_PREFIX=...   (自定义实验名前缀)
#
# 平台通过 -e DISTRIBUTED_JOB=true 自动注入:
#   NODE_RANK / NODE_COUNT / MASTER_ADDR / PROC_PER_NODE / JOB_ID
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX / CUDA_VISIBLE_DEVICES
################################################################################

BLOCK_MODE="${BLOCK_MODE:-blockwise}"
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"
BLOCK_LATENT_HEAD_TYPE="${BLOCK_LATENT_HEAD_TYPE:-per_offset_tf_transformer}"
BLOCK_LATENT_HEAD_LAYERS="${BLOCK_LATENT_HEAD_LAYERS:-1}"
BLOCK_LATENT_HEAD_FFN_MULT="${BLOCK_LATENT_HEAD_FFN_MULT:-4.0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
# Defaults match the verified-working config from c1024 testing.
# Override OPTIMIZER=muon_adamw only after head param-group routing is added.
OPTIMIZER="${OPTIMIZER:-adamw}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FLOAT_PRECISION="${FLOAT_PRECISION:-bf16-mixed}"
PEAK_LR="${PEAK_LR:-}"
WARM_UP_STEPS="${WARM_UP_STEPS:-}"
RUN_PREFIX="${RUN_PREFIX:-1node-8gpu-bf16mixed-blockwise-tfmtphead-L${BLOCK_LATENT_HEAD_LAYERS}}"

rjob submit \
  --name=ebt-d26-blockwise-tfmtphead-1node-8gpu \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  --host-network=false \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/lixueyan:/mnt/shared-storage-user/lixueyan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  -e DISTRIBUTED_JOB=true \
  -e BLOCK_MODE="${BLOCK_MODE}" \
  -e TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE}" \
  -e BLOCK_LATENT_HEAD_TYPE="${BLOCK_LATENT_HEAD_TYPE}" \
  -e BLOCK_LATENT_HEAD_LAYERS="${BLOCK_LATENT_HEAD_LAYERS}" \
  -e BLOCK_LATENT_HEAD_FFN_MULT="${BLOCK_LATENT_HEAD_FFN_MULT}" \
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  -e GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
  -e FLOAT_PRECISION="${FLOAT_PRECISION}" \
  -e OPTIMIZER="${OPTIMIZER}" \
  -e PEAK_LR="${PEAK_LR}" \
  -e WARM_UP_STEPS="${WARM_UP_STEPS}" \
  -e RUN_PREFIX="${RUN_PREFIX}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "bash /mnt/shared-storage-user/lixueyan/nar/nova_dev/nova/ebt/runs/rjob/run_ebt_blockwise_tfmtphead_1node_8gpu.sh"
