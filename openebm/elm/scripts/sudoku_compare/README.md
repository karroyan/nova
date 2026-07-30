# Sudoku Comparison 模块说明

## 功能简介

`openebm/elm/scripts/sudoku_compare/` 用于在 SATNet Sudoku `test` split 上对比 EBM 模型与多个 LLM 的解题效果。当前默认对比对象为：

- 小模型组：`qwen3_1p7b`、`llama3p2_1b`
- 大模型组：`qwen3p6_27b`、`deepseek_r1_0528`
- EBM：由 `openebm.elm.scripts.eval_sudoku_samples` 加载 checkpoint 评估

模块会完成以下工作：

- 加载 Sudoku test split。
- 对 LLM 生成解题输出，解析最终 9x9 答案。
- 统计 `board_acc`、`cell_acc`、`valid_rate`、耗时等指标。
- 保存每条样本的结构化结果、完整轨迹和典型 case。
- 聚合 EBM 与 LLM 结果，导出 CSV/Markdown。
- 支持一次性全量评估、单样本 smoke test、分阶段 staged evaluation。

## 目录/文件结构说明

```text
openebm/elm/scripts/sudoku_compare/
  __init__.py
  README.md
  common.py
  eval_llm_sudoku.py
  aggregate_results.py
  case_study.py
  staged_eval.py
  run_all_sudoku_compare.sh
  smoke_one_sample_each_llm.sh
```

- `common.py`
  - 共享数据加载、模型别名、Sudoku board 格式化、LLM 输出解析、指标统计、JSON/JSONL 读写。
  - 默认数据目录：`openebm/elm/data/sudoku_cache_v2`。
  - 默认模型别名：
    - `qwen3_1p7b`
    - `llama3p2_1b`
    - `qwen3p6_27b`
    - `deepseek_r1_0528`

- `eval_llm_sudoku.py`
  - LLM 评估入口。
  - 默认 backend 为 `vllm`，也支持 `transformers`。
  - 支持 `--resume`、`--overwrite`、tensor parallel、FlashInfer attention backend、trace 导出和 batch 级进度日志。

- `aggregate_results.py`
  - 读取 EBM 与 LLM 的 `summary.json`，输出对比表。
  - 输出字段包括：`model`、`params`、`board_acc`、`cell_acc`、`valid_rate`、`avg_time`、`n`、`source`。

- `case_study.py`
  - 读取 EBM/LLM 的 `test.jsonl`，导出代表性 Markdown case。
  - 默认选择：
    - EBM 正确、同量级 LLM 错误。
    - EBM 正确、大模型错误。
    - 低 givens 的困难样本。

- `staged_eval.py`
  - 分阶段执行的辅助工具。
  - 子命令：
    - `total`：读取 split 总样本数。
    - `materialize-ebm`：从已有 EBM 结果物化阶段累计视图。
    - `report`：生成本轮和累计报告。
    - `stage-event`：更新 `staged_manifest.json`。

- `run_all_sudoku_compare.sh`
  - 主运行脚本。
  - 负责 EBM、LLM、小模型并发、大模型 tensor parallel、聚合、case study 和 staged evaluation 编排。

- `smoke_one_sample_each_llm.sh`
  - LLM 单样本冒烟测试脚本。
  - 每个启用模型评估 1 条 Sudoku test 样本，用于检查模型加载、vLLM、FlashInfer、解析和轨迹导出是否正常。

## 环境依赖

建议使用项目当前默认环境：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python
```

主要依赖：

- Python 3.10+。
- `torch`：加载 Sudoku split、EBM 评估、GPU 检测。
- `transformers`：加载 tokenizer 和可选 transformers backend。
- `vllm`：默认 LLM 推理 backend。
- `tqdm`：进度条。
- `flashinfer-python`、`flashinfer-cubin`：非 MLA 模型默认 FlashInfer attention backend。
- `nvidia-smi`：可选，用于 GPU 数量检测 fallback。
- OpenEBM 项目自身模块，尤其是 `openebm.elm.scripts.eval_sudoku_samples`。

脚本默认启用：

```bash
LLM_BACKEND=vllm
ATTENTION_BACKEND=FLASHINFER
R1_ATTENTION_BACKEND=auto
FLASHINFER_DISABLE_VERSION_CHECK=1
FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer
VLLM_WORKER_MULTIPROC_METHOD=spawn
ENFORCE_EAGER=1
THINKING=disable
```

注意：`eval_llm_sudoku.py` 会拒绝 single-process vLLM `--data-parallel-size > 1`，除非显式使用 `--distributed-executor-backend external_launcher`。主脚本默认不使用 vLLM offline DP；小模型多卡时采用“每个小模型一个进程 + 大 batch”的方式。

## 运行命令

以下命令均从仓库根目录运行：

```bash
cd /mnt/shared-storage-user/puyuan/code/OpenEBM
```

### 1. 查看完整命令计划

不会加载模型，适合先确认输出目录、GPU 分配和实际命令。

```bash
DRY_RUN=1 VERIFY_ENV=0 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

### 2. 单样本 LLM smoke test

默认对所有启用的 LLM 各评估 `SAMPLE_INDEX=0` 的 1 条样本。

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh
```

只测小模型：

```bash
RUN_QWEN27=0 RUN_R1=0 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh
```

指定样本：

```bash
SAMPLE_INDEX=10 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh
```

### 3. 一次性完整评估

默认 `NUM_SAMPLES=-1`，即评估完整 test split。

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

只跑前 32 条并跳过大模型：

```bash
NUM_SAMPLES=32 RUN_QWEN27=0 RUN_R1=0 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

跳过 EBM，只跑 LLM 并生成报告：

```bash
RUN_EBM=0 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

### 4. 分阶段 staged evaluation

默认 `STAGE_SIZE=0`，表示关闭 staged 模式。设置 `STAGE_SIZE>0` 后，每轮按如下顺序执行：

```text
LLM small -> Qwen3.6-27B -> DeepSeek-R1 -> EBM -> 当前轮/累计报告
```

每轮 100 条，直到完整 test split：

```bash
STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

等价命令行参数：

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh --stage-size 100
```

只评估前 1250 条，每轮 100 条：

```bash
NUM_SAMPLES=1250 STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

复用已有 EBM 结果，避免重复推理：

```bash
EBM_REUSE_DIR=/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_ebm \
STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

staged 模式默认 `RUN_ID=staged_<STAGE_SIZE>`，例如 `staged_100`。中断后使用同一命令会回到同一目录续跑。如需全新输出目录，显式设置：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S) STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

### 5. 直接运行 LLM 评估

小模型：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models qwen3_1p7b llama3p2_1b \
  --data-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2 \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/manual_llm \
  --backend vllm \
  --batch-size 8 \
  --max-tokens 2048 \
  --temperature 0 \
  --top-p 1.0 \
  --thinking disable \
  --resume \
  --enforce-eager \
  --attention-backend FLASHINFER
```

Qwen3.6-27B tensor parallel 示例：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models qwen3p6_27b \
  --data-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2 \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/manual_llm \
  --backend vllm \
  --tensor-parallel-size 4 \
  --batch-size 1 \
  --max-tokens 4096 \
  --max-model-len 16384 \
  --temperature 0 \
  --top-p 1.0 \
  --thinking disable \
  --resume \
  --enforce-eager \
  --attention-backend FLASHINFER
```

DeepSeek-R1 示例：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models deepseek_r1_0528 \
  --data-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2 \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/manual_llm \
  --backend vllm \
  --tensor-parallel-size 8 \
  --batch-size 1 \
  --max-tokens 8192 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --temperature 0 \
  --top-p 1.0 \
  --thinking disable \
  --resume \
  --enforce-eager \
  --attention-backend auto
```

DeepSeek-R1 使用 MLA attention，不要强制使用非 MLA 的 `FLASHINFER`。保持
`auto` 时 vLLM 会在当前 CUDA 平台上选择可用的 MLA backend。

使用本地自定义 Hugging Face 模型路径：

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models my_model=/path/to/local/hf/model \
  --backend transformers \
  --batch-size 1 \
  --num-samples 8
```

### 6. 直接聚合结果

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.aggregate_results \
  --results-root /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_outputs \
  --ebm-result-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_ebm \
  --llm-root /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_llm \
  --csv-out /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_reports/comparison.csv
```

### 7. 直接生成 case study

```bash
/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python \
  -m openebm.elm.scripts.sudoku_compare.case_study \
  --data-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2 \
  --ebm-result-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_ebm \
  --llm-root /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_llm \
  --out-md /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_reports/case_study.md
```

## 输入与输出说明

### 输入数据

默认数据目录：

```text
/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2
```

`common.py` 复用 `eval_sudoku_samples.py` 的 split 定义，默认文件包括：

```text
rrn_train.pt
rrn_val.pt
satnet_test.pt
```

每条样本需要包含：

- `puzzle`：81 个 cell，`0` 表示空格。
- `solution`：81 个 cell 的标准答案。

### 模型路径

默认 LLM alias 到 Hugging Face cache 根目录：

```text
qwen3_1p7b        /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-1.7B
llama3p2_1b      /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
qwen3p6_27b      /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.6-27B
deepseek_r1_0528 /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528
```

默认 EBM checkpoint：

```text
/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-v3p1-20260529/sft_train.v4/checkpoints/s=step=4424-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss_balanced=valid_loss_balanced=0.2667.ckpt
```

### 主运行输出

默认输出根目录：

```text
/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare
```

一次运行的结构：

```text
runs/sudoku_compare/
  logs/<RUN_ID>/
    INDEX.md
    console.log
    run_env.txt
    status.tsv
    *.cmd
    *.log
  outputs/<RUN_ID>/
    ebm/
      results/test.jsonl
      results/summary.json
      results/summary.csv
      results/timing.json
    llm/<model>/
      config.json
      progress.jsonl
      test.jsonl
      summary.json
      traces/
        all.jsonl
        correct_examples.jsonl
        wrong_examples.jsonl
        correct_5.jsonl
        wrong_5.jsonl
        parse_failed.jsonl
        near_miss.jsonl
        invalid_sudoku.jsonl
        typical_cases.jsonl
        trace_summary.json
    reports/
      comparison.csv
      case_study.md
```

主脚本还会更新 symlink：

```text
runs/sudoku_compare/logs/latest
runs/sudoku_compare/latest_outputs
runs/sudoku_compare/latest_ebm
runs/sudoku_compare/latest_llm
runs/sudoku_compare/latest_reports
```

### staged 输出

设置 `STAGE_SIZE>0` 后，额外输出：

```text
outputs/<RUN_ID>/staged_manifest.json
outputs/<RUN_ID>/reports/stages/stage_0001/current/
outputs/<RUN_ID>/reports/stages/stage_0001/cumulative/
outputs/<RUN_ID>/reports/stages/stage_0002/current/
outputs/<RUN_ID>/reports/stages/stage_0002/cumulative/
outputs/<RUN_ID>/reports/latest_stage_current
outputs/<RUN_ID>/reports/latest_stage_cumulative
```

`current` 表示本轮 `[stage_start, stage_end)`；`cumulative` 表示截至当前的 `[START_INDEX, stage_end)`。

### smoke 输出

默认输出根目录：

```text
/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare_smoke
```

结构：

```text
runs/sudoku_compare_smoke/
  logs/<RUN_ID>/
  llm_one_sample/<RUN_ID>/<model>/
    config.json
    progress.jsonl
    test.jsonl
    summary.json
    traces/
```

## 指标与答案解析

主要指标：

- `board_acc`：完整 9x9 答案全对才算正确。
- `cell_acc` / `filled_cell_acc`：只统计原始题目中为 `0` 的空格位置。
- `valid_rate` / `parsed_pct`：是否能解析出完整 81 cell 的 1-9 答案。
- `valid_sudoku_rate`：解析出的答案是否满足 Sudoku 规则，保存在 JSON summary 中。
- `avg_time` / `avg_s_per_sample`：平均每样本耗时。

LLM 答案解析逻辑：

- 优先解析 `Final answer:`、`Final grid:`、`Completed grid:`、`Solution:`、`Answer:` 后的内容。
- 支持 `</think>` 后答案、去除 `<think>...</think>` 后答案、9 行网格、Markdown 表格和连续 81 位数字。
- 只有完整 81 个 cell 且全部为 `1-9` 才算 parsed。
- 复述题面或包含 `0`/`.` 的未完成网格不会算作有效答案。
- 解析失败样本会写入 `traces/parse_failed.jsonl`，并计入 `n` 分母，视为错误。

## 常用参数

主脚本常用环境变量：

```text
NUM_SAMPLES=-1             # -1 表示完整 test split；非负数表示评估子集
START_INDEX=0              # 起始 test idx
STAGE_SIZE=0               # 0 表示一次性评估；>0 表示 staged evaluation
RUN_EBM=1
RUN_LLMS=1
RUN_SMALL=1
RUN_QWEN27=1
RUN_R1=1
RUN_REPORTS=1
DRY_RUN=0
VERIFY_ENV=1
RESULTS_ROOT=<输出根目录>
PYTHON=<统一 Python>
EBM_PYTHON=<EBM Python>
LLM_PYTHON=<LLM Python>
REPORT_PYTHON=<报告 Python>
CAPTURE_CONSOLE_LOG=1      # 写入 logs/<RUN_ID>/console.log；设 0 可关闭
```

LLM 推理相关：

```text
LLM_BACKEND=vllm
ATTENTION_BACKEND=FLASHINFER
R1_ATTENTION_BACKEND=auto
ENFORCE_EAGER=1
THINKING=disable
SYSTEM_PROMPT=             # 可选；覆盖 eval_llm_sudoku.py 默认 system prompt
ANSWER_FORMAT=grid         # grid 或 flat81；flat81 要求 81 位行优先数字
R1_ANSWER_FORMAT=          # 可选；只覆盖 DeepSeek-R1，例如 flat81
STRUCTURED_REGEX=          # 可选；传给 vLLM structured_outputs.regex
R1_STRUCTURED_REGEX=       # 可选；只约束 DeepSeek-R1，例如 '[1-9]{81}'
TEMPERATURE=0
TOP_P=1.0
MAX_TOKENS=<覆盖所有模型组>
SMALL_MAX_TOKENS=2048
QWEN27_MAX_TOKENS=4096
R1_MAX_TOKENS=8192
R1_GPU_MEMORY_UTILIZATION=0.85
TRACE_LOG=full
TRACE_LOG_CHARS=50000
RESPONSE_LOG=truncated
RESPONSE_LOG_CHARS=12000
SAVE_PROMPTS=0
```

并行相关：

```text
GPU_COUNT=<自动检测或手动指定>
GPU_IDS=0,1,2,3
PARALLEL_SMALL_MODELS=1
SMALL_BATCH_SIZE=16        # 多 GPU 默认；单 GPU 默认 8
QWEN27_TP=min(GPU_COUNT,4)
R1_TP=min(GPU_COUNT,8)
```

## 备注/注意事项

- `eval_llm_sudoku.py` 默认只从本地模型目录加载：`local_files_only=True`。
- `--resume` 会根据已有 `test.jsonl` 跳过已完成 idx；`--overwrite` 会清理旧 LLM trace 导出。
- `test.jsonl` 默认保存截断后的 response；完整输出默认保存在 `traces/all.jsonl`。
- 分阶段模式中，LLM 每轮传入累计范围并依赖 `--resume` 只补新样本；EBM 优先复用已有结果，缺失时才累计运行。
- EBM 多卡评估日志中的 tqdm 是 rank0 本地 shard 进度，不等于全局 test 样本总数。
- 若 FlashInfer 版本检查导致 import 报错，可保持默认 `FLASHINFER_DISABLE_VERSION_CHECK=1`。
- DeepSeek-R1 默认 `R1_GPU_MEMORY_UTILIZATION=0.85`，用于给 warmup/all-reduce 临时显存留出余量。若机器空闲且确认稳定，可手动调高。
- 若需要降低长推理模型耗时，可以调低 `QWEN27_MAX_TOKENS` 或 `R1_MAX_TOKENS`，但这可能影响最终答案输出。
