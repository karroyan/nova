<!-- # Energy-based Language Models

This is the reproduction of the ICLR ORAL paper [Energy-based Language Models](https://arxiv.org/abs/2203.02155). This is the first version of the EBT extended to NanoChat dataset.

### Setup environment

```bash
bash runs/install.sh
```

### Dataset preparation

```bash
bash runs/dataset_prepare.sh
```

### Setting wandb logger

Sign up a free account on [wandb.ai](https://wandb.ai/site), generate an API key, and set it in the '../../runs/run_ebt.sh' as follows:

```bash
export WANDB_API_KEY=<Your API Key>
```

### Training

```bash
bash runs/run_ebt.sh
```

You can check the training logs from the wandb dashboard at `https://wandb.ai/<Your account>/nlp_pretrain`.

### Command Explanation

```
python train.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
--tokenizer "EleutherAI/gpt-neox-20b" \                         # internally use nanochat tokenizer
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 256 \                                                  # Context length 256
\
--gpus "-1" \                                                           # Use all available GPUs  
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \                                    # Batch size per device 32
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "nanochat" \                                     # internally use nanochat dataset
--num_workers 12 \
--val_check_interval 1000 \                                     # Validation every 1000 steps
--limit_val_batches 100 \                                       # Validation has 100 steps
--val_sanity 1 \                                                # Validation before training
\
--wandb_project 'nlp_pretrain' \
\
--log_model_archi \
--log_every_n_steps: 100 \                                      # Log every 100 steps
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}
```

## File structure
```
.
├── README.md
├── nanolightning
│   ├── __init__.py
│   ├── iteratabledataset.py
│   ├── iteratabletrainer.py
│   ├── torchlightning_function.py
│   ├── torchlightning_module.py
│   └── torchlightning_trainer.py
├── ar_ebt_adaln.py
├── ar_ebt_default.py
├── ar_ebt_time_embed.py
├── ar_transformer.py
├── collector.py
├── dataset.py
├── eval.py
├── generate.py
├── logger.py
├── metrics.py
├── modeling_ebt.py
├── optimization.py
├── replay_buffer.py
├── tokenizer.py
├── train.py
├── trainer.py
└── utils.py          
``` -->





