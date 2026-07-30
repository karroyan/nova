<!-- # Non-autoregressive Optimization & Vast Architecture (NOVA)

![nanochat logo](dev/nova.png)

This project aims to explore non-autoregressive (NAR) large language models and investigate their potential to scale effectively with increasing training data and model size.
While autoregressive (AR) models have demonstrated strong scaling behavior, non-autoregressive large models remain underexplored, especially in the context of foundation-scale training.
We welcome contributions from the community to push this boundary, and make non-autoregressive models truly powerful and practical.


## Why non-autoregressive models?

AR models are fundamentally limited by their sequential generation process, which restricts parallelism and token-by-token error accumulation.
NAR models offer structural advantages to enhance parallelism, safety and controllability
This year, it is exciting to see more substantial progress in NAR modeling this year, which includes:

- **EBT:** #Energy-Based Transformers are Scalable Learners and Thinkers# [[ICLR Oral]](https://openreview.net/forum?id=ZBj3Qp1bYg)

- **EBM:** Logical Intelligence, a startup focused on developing energy-based foundation models. See [Logical Inteligence](https://logicalintelligence.com/).

- **MDLM** The first product-level diffusion LM has been released online. See [Mercury-2](https://www.inceptionlabs.ai/blog/introducing-mercury-2).

We look forward to continued deployment and breakthroughs of non-autoregressive models.


## What do we provide?

With this repository, you are ready to train NAR models on a 100B pretraining dataset, and explore the scaling laws of NAR models. The dataset has been carefully curated and cleaned to support training at the GPT-2 scale.
The training data originates from the open-source project [nanochat](https://github.com/karpathy/nanochat). We gratefully acknowledge the contributions of Dr. Andrej Karpathy and the nanoGPT community.



## Guides

### Energy-based Language Models

A step-by-step turtorial is given in the ['nova/ebt'](https://github.com/openway-ai/nova/tree/main/nova/ebt) directory.

### Masked Diffusion Language Models

On the way.


## File structure
```
.
├── LICENSE
├── README.md
├── dev
├── nanochat
├── nova
│   ├── ebt                         # Energy-based LM             
│   └── mdlm                        # Masked Diffusion LM
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   ├── speedrun.sh                 # Train the ~$100 nanochat d20
│   ├── dataset_prepare.sh          # Prepare the dataset
│   ├── install.sh                  # Install dependencies
│   └── run_ebt.sh                  # Train the EBT model
├── scripts
├── tasks
├── tests
└── uv.lock
```

## Research

If you are a researcher and wish to help merge more non-autoregressive models to this repository, we welcome your contributions. We encourage clean, modular, and well-documented implementations, and recommend opening an issue to briefly describe your implemented models, code structures, and experimental results before submitting a pull request. We welcome contributions from the community to push this boundary, and make non-autoregressive models truly powerful and practical.

## Acknowledgements

This project is primiarily forked from [nanochat](https://github.com/karpathy/nanochat), and we extend our most deep appreciation to Dr. Andrej Karpathy for releasing such a powerful and accessible foundation. Without this groundwork, follow-up research and development would be more challenging. We remain grateful for the continued support and collaboration from the open-source research community.

## License

MIT -->
