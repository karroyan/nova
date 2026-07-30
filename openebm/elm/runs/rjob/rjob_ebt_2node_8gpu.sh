#!/bin/bash

################################################################################
# EBT d26 多机训练 rjob 提交脚本 - 2节点 × 8卡
#
# 用法:
#   bash rjob_ebt_2node_8gpu.sh
#
# 平台通过 -e DISTRIBUTED_JOB=true 自动注入:
#   NODE_RANK / NODE_COUNT / MASTER_ADDR / PROC_PER_NODE / JOB_ID
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX / CUDA_VISIBLE_DEVICES
################################################################################

BASE_TRAIN_DATASET="${base_train_dataset:-${BASE_TRAIN_DATASET:-fineweb}}"
case "${BASE_TRAIN_DATASET}" in
  fineweb|climbmix|dclm) ;;
  *)
    echo "Unsupported base_train_dataset=${BASE_TRAIN_DATASET}; expected fineweb, climbmix, or dclm" >&2
    exit 2
    ;;
esac

rjob submit \
  --name=ebt-d26-2node-8gpu-${BASE_TRAIN_DATASET} \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  -P 2 \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  -e DISTRIBUTED_JOB=true \
  -e BASE_TRAIN_DATASET="${BASE_TRAIN_DATASET}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/rjob/run_ebt_freeembed_2node_8gpu.sh"
