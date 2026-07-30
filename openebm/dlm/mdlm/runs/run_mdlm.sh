#!/bin/bash
# Run from nova/mdlm/ directory: bash runs/run_mdlm.sh

source ../../.venv/mdlm/bin/activate

export WANDB_API_KEY=<Your API Key>

python train.py \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=False \
  sampling.steps=1000
