#!/bin/bash

################################################################################
# EBT d26 单机训练 rjob 提交脚本 - 1节点 × 8卡
#
# 用法:
#   bash rjob_ebt_1node_8gpu.sh
#
# 平台通过 -e DISTRIBUTED_JOB=true 自动注入:
#   NODE_RANK / NODE_COUNT / MASTER_ADDR / PROC_PER_NODE / JOB_ID
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX / CUDA_VISIBLE_DEVICES
################################################################################

rjob submit \
  --name=ebt-d26-1node-8gpu \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  --host-network=false \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  -e DISTRIBUTED_JOB=true \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "/mnt/shared-storage-user/luyudong/nova/nova/ebt/runs/rjob/run_ebt_1node_8gpu.sh"
