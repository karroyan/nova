# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

source .venv/nanochat/bin/activate
export PYTHONPATH="${PWD}/nanochat:${PWD}:${PYTHONPATH:-}"

python -m scripts.base_train --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run=$WANDB_RUN

# torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run=$WANDB_RUN
