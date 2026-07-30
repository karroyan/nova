# Masked Diffusion Language Models

This is a reproduction of [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.07524) (MDLM, NeurIPS 2024), extended to train on the NanoChat (FineWeb-Edu) dataset. All commands below must be run from the `nova/mdlm/` directory.

### Setup environment

```bash
bash runs/install.sh
```

### Dataset preparation

```bash
bash runs/dataset_prepare.sh
```

The trained tokenizer is cached at `~/.cache/nanochat/tokenizer/`. MDLM always uses the nanochat tokenizer — the `tokenizer_name_or_path` field in `configs/data/*.yaml` is ignored.

### Setting wandb logger

Sign up a free account on [wandb.ai](https://wandb.ai/site), generate an API key, and set it in the '../../runs/run_ebt.sh' as follows:

```bash
export WANDB_API_KEY=<Your API Key>
```

### Training

```bash
bash runs/run_mdlm.sh
```

### Command Explanation

```
python train.py \
  model=small \                            # Model size: tiny / small / medium
  data=openwebtext-split \                 # Dataset; openwebtext / wikitext103 / lm1b / text8 / ag_news
  parameterization=subs \                  # Loss parameterization: subs (MDLM) / d3pm / sedd
  noise=loglinear \                        # Noise schedule: loglinear / linear / polynomial
  backbone=dit \                           # Backbone: dit (default) / hf_dit (HuggingFace ckpt)
  T=0 \                                    # 0 = continuous time; 1000 = discrete T-step
  \
  model.length=1024 \                      # Context length
  loader.batch_size=16 \                   # Per-machine batch size (global=512, auto-distributed)
  loader.eval_batch_size=16 \
  \
  training.ema=0.9999 \                    # EMA decay (EMA weights used for validation/sampling)
  training.antithetic_sampling=True \      # Antithetic timestep sampling
  training.importance_sampling=False \     # Importance sampling for loglinear noise
  \
  optim.lr=3e-4 \                          # Peak learning rate (AdamW, β₁=0.9, β₂=0.999)
  trainer.gradient_clip_val=1.0 \          # Gradient clipping
  trainer.max_steps=1000000 \              # Total training steps
  trainer.val_check_interval=1000 \        # Validation every 1000 steps
  \
  sampling.predictor=ddpm_cache \          # Sampler: analytic / ddpm / ddpm_cache
  sampling.steps=128 \                     # Denoising steps at inference
  \
  wandb.name=mdlm-owt \                    # WandB run name (wandb.project=text-diffusion)
  \
  checkpointing.resume_from_ckpt=true      # Auto-resume from last.ckpt if it exists
```

**Model sizes** (hidden_size / n_blocks / n_heads):

| Config   | Hidden | Blocks | Heads |
|----------|--------|--------|-------|
| `tiny`   | 256    | 4      | 4     |
| `small`  | 768    | 12     | 12    |
| `medium` | 1024   | 16     | 16    |

**Parameterization constraints**:
- `d3pm` requires `T > 0`
- `sedd` disallows `importance_sampling` and `change_of_variables`
- Discrete time (`T > 0`) only works with `d3pm` or `subs`

Hydra outputs go to `./outputs/{data.train}/{date}/{time}/`. Lightning checkpoints are saved under `checkpoints/` in that directory; `last.ckpt` is used for auto-resume.

### Run modes

```bash
# Default: full training
python train.py

# Evaluate validation perplexity from a checkpoint
python train.py mode=ppl_eval eval.checkpoint_path=/path/to/last.ckpt

# Generate text samples (+ optional GPT-2-large generative perplexity)
python train.py mode=sample_eval eval.checkpoint_path=/path/to/last.ckpt \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
```

## File structure

```
.
├── README.md
├── CLAUDE.md
├── configs/
│   ├── config.yaml             # Root Hydra config (all defaults)
│   ├── model/                  # tiny, small, medium, small-ar, tiny-ar, tiny-dimamba
│   ├── data/                   # openwebtext, wikitext103, lm1b, text8, ag_news, ...
│   ├── noise/                  # loglinear, linear, polynomial, ar
│   ├── strategy/               # ddp, fsdp
│   ├── lr_scheduler/           # constant_warmup, cosine_decay_warmup
│   └── callbacks/              # checkpoint_every_n_steps, checkpoint_monitor, ...
├── runs/
│   ├── install.sh              # Environment setup
│   ├── dataset_prepare.sh      # Data download + tokenizer training
│   └── run_mdlm.sh             # Reference training launch script
├── train.py                    # Entry point: @hydra.main, dispatches train/ppl_eval/sample_eval
├── diffusion.py                # Diffusion(L.LightningModule): masking, loss, EMA, sampling
├── dit.py                      # DIT backbone: bidirectional Transformer for Lightning stack
├── mdlm.py                     # Standalone MDLM(nn.Module): nanochat-style rewrite
├── noise_schedule.py           # Noise schedules: LogLinear, Geometric, Cosine, Linear
├── nanochat_dataloader.py      # Active data pipeline: FineWeb-Edu parquet via nanochat
├── dataset.py                  # Older nanochat pipeline (reference only, not used by train.py)
├── dataloader.py               # Original HuggingFace data pipeline (samplers used by diffusion.py)
├── tokenizer.py                # _RustBPETokenizer: HF-compatible wrapper around nanochat tokenizer
├── ema.py                      # ExponentialMovingAverage: store/copy_to/restore pattern
├── utils.py                    # LR scheduler, fsspec helpers, sampling utilities, logger
└── pyproject.toml

```
