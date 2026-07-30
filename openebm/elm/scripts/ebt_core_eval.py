"""
EBT CORE 评估脚本
基于 nanochat 的 base_eval.py，适配 EBT 模型
"""
import os
import sys
import csv
import time
import json
import yaml
import heapq
import random
import argparse
import glob as glob_module
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from openebm.elm.trainer import ModelTrainer
from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
from nanochat.core_eval import (
    evaluate_example,
    evaluate_task as _nanochat_evaluate_task,
    forward_model,
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    batch_sequences_lm,
    stack_sequences,
)


class EBTModelWrapper:
    """将 EBT 模型包装为 nanochat 兼容的接口"""

    def __init__(self, model, tokenizer, device, max_seq_len=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        """
        前向传播，兼容 nanochat 的接口

        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] 或 None
            loss_reduction: 'mean' 或 'none'

        Returns:
            如果 targets is None: 返回 logits [batch_size, seq_len, vocab_size]
            否则: 返回 loss (scalar 或 [batch_size])
        """
        with torch.no_grad():
            # EBT forward 返回 (logits_list, energies)
            outputs = self.model.forward(input_ids, start_pos=0, learning=False, return_raw_logits=True)

            # 取最后一个 MCMC 步骤的 logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if isinstance(logits, list):
                    logits = logits[-1]  # 最后一个 MCMC 步骤
            else:
                logits = outputs

            if targets is None:
                return logits

            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss

    def get_device(self):
        return self.device


def load_ebt_model(ckpt_path, tokenizer_path, device, dtype=torch.bfloat16):
    """加载 EBT checkpoint"""
    print(f"Loading EBT checkpoint from: {ckpt_path}")

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # 设置推理模式参数
    hparams['execution_mode'] = 'inference'
    hparams['no_wandb'] = True
    # 禁用 torch.compile — eval 不需要编译，且 compile + MCMC grad + mp.spawn 会导致 CUBLAS 错误
    hparams['compile_model'] = False

    # 加载模型
    model_trainer = ModelTrainer(hparams)

    # 复用 on_load_checkpoint 的 _orig_mod 前缀修复逻辑
    state_dict = checkpoint['state_dict']
    has_orig_mod_keys = any('_orig_mod.' in k for k in state_dict)
    model_has_orig_mod = any('_orig_mod.' in k for k in model_trainer.state_dict())

    if has_orig_mod_keys and not model_has_orig_mod:
        state_dict = {k.replace('._orig_mod.', '.'): v for k, v in state_dict.items()}
        print(f"[load_ebt_model] Stripped '_orig_mod' prefix from {len(checkpoint['state_dict'])} keys")
    elif not has_orig_mod_keys and model_has_orig_mod:
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_sd['model._orig_mod.' + k[len('model.'):]] = v
            else:
                new_sd[k] = v
        state_dict = new_sd
        print(f"[load_ebt_model] Added '_orig_mod' prefix to {len(checkpoint['state_dict'])} keys")

    model_trainer.load_state_dict(state_dict)
    model_trainer.eval()
    model = model_trainer.model
    model.to(device=device, dtype=dtype)
    model.eval()

    # 加载 tokenizer
    from nanochat.tokenizer import get_tokenizer
    tokenizer_obj = get_tokenizer()
    tokenizer_wrapper = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer_obj)

    # 获取最大序列长度
    max_seq_len = hparams.get('context_length', 256)

    # 包装模型
    wrapped_model = EBTModelWrapper(model, tokenizer_wrapper, device, max_seq_len=max_seq_len)

    print(f"✓ Model loaded: {hparams['model_name']} (size: {hparams.get('model_size', 'unknown')})")
    print(f"✓ Context length: {max_seq_len}")

    # 返回原始 RustBPETokenizer 给 eval 函数（nanochat core_eval 需要 .get_bos_token_id() 和 __call__(prompts, prepend=...) 接口）
    return wrapped_model, tokenizer_obj, hparams


def evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len=None):
    """适配 nanochat 的 evaluate_task，忽略 max_seq_len 参数（已通过 model.max_seq_len 传递）"""
    return _nanochat_evaluate_task(model, tokenizer, data, device, task_meta)


@torch.no_grad()
def evaluate_example_with_trajectory(idx, model, tokenizer, data, device, task_meta):
    """
    与 nanochat evaluate_example 逻辑一致，额外返回 trajectory dict。
    不修改 nanochat 代码，复用其 render/batch/stack/forward 工具函数。

    Returns:
        (is_correct: bool, trajectory: dict)
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item) — 与 nanochat 一致
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # max_seq_len truncation — 与 nanochat 一致
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0
                assert e - num_to_crop >= 0
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model
    losses, predictions = forward_model(model, input_ids)

    # Build trajectory dict
    trajectory = {
        "sample_id": idx,
        "is_correct": False,
        "metadata": {
            "task_type": task_type,
            "num_fewshot": num_fewshot,
        },
    }

    if task_type == 'language_modeling':
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()

        # Trajectory: decode ground truth continuation and model prediction
        try:
            gt_text = tokenizer.decode(actual_tokens.tolist())
        except Exception:
            gt_text = str(actual_tokens.tolist())
        try:
            pred_text = tokenizer.decode(predicted_tokens.tolist())
        except Exception:
            pred_text = str(predicted_tokens.tolist())

        trajectory["input"] = prompts[0]  # prompt_without
        trajectory["ground_truth"] = gt_text
        trajectory["prediction"] = pred_text
        trajectory["parsed_answer"] = pred_text

    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        gold_idx = item['gold']
        is_correct = pred_idx == gold_idx

        # Trajectory: record choices and losses
        if task_type == 'multiple_choice':
            choices = item.get('choices', [])
            trajectory["input"] = prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            trajectory["ground_truth"] = choices[gold_idx] if gold_idx < len(choices) else str(gold_idx)
            trajectory["prediction"] = choices[pred_idx] if pred_idx < len(choices) else str(pred_idx)
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(choices)
        else:  # schema
            context_options = item.get('context_options', [])
            continuation = item.get('continuation', '')
            trajectory["input"] = prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            trajectory["ground_truth"] = context_options[gold_idx] if gold_idx < len(context_options) else str(gold_idx)
            trajectory["prediction"] = context_options[pred_idx] if pred_idx < len(context_options) else str(pred_idx)
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(context_options)
            trajectory["metadata"]["continuation"] = continuation

        trajectory["metadata"]["mean_losses"] = mean_losses
        trajectory["metadata"]["gold_idx"] = gold_idx
        trajectory["metadata"]["pred_idx"] = pred_idx
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    trajectory["is_correct"] = bool(is_correct)
    return is_correct, trajectory


def evaluate_task_with_trajectory(model, tokenizer, data, device, task_meta,
                                  output_dir=None, task_label=None,
                                  num_trajectory_samples=10,
                                  show_progress=True, progress_position=1):
    """
    替代 _nanochat_evaluate_task，循环中收集 trajectory 并写入 JSONL。
    前 num_trajectory_samples 个样本调用 evaluate_example_with_trajectory，
    剩余的调用原 evaluate_example（性能考虑）。

    Returns:
        (mean_correct: float, trajectories: list[dict])
    """
    n = len(data)
    correct = 0

    # Prepare trajectory output
    trajectories = []
    traj_file = None
    if num_trajectory_samples > 0 and output_dir and task_label:
        traj_dir = os.path.join(output_dir, "trajectories", task_label)
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, "samples.jsonl")
        traj_file = open(traj_path, 'w', encoding='utf-8')

    # Sample-level progress bar (only for single-GPU path)
    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  samples", position=progress_position,
                        leave=False, unit="ex")

    try:
        for idx in iterator:
            if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                is_correct, traj = evaluate_example_with_trajectory(
                    idx, model, tokenizer, data, device, task_meta)
                trajectories.append(traj)
                if traj_file:
                    traj_file.write(json.dumps(traj, ensure_ascii=False) + '\n')
                    traj_file.flush()
            else:
                is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct += int(is_correct)
    finally:
        if traj_file:
            traj_file.close()

    mean_correct = correct / n if n > 0 else 0.0
    return mean_correct, trajectories


def evaluate_core(model, tokenizer, device, eval_bundle_dir, max_per_task=-1,
                  task_samples=None, output_dir=None, num_trajectory_samples=10):
    """
    运行完整的 CORE 评估

    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        eval_bundle_dir: 评估数据目录
        max_per_task: 全局默认每个任务的最大样本数，-1 表示全部
        task_samples: 字典，指定每个任务的样本数，例如 {"hellaswag_zeroshot": 100}
        output_dir: 输出目录（用于保存 trajectory）
        num_trajectory_samples: 每个任务保存前 N 个样本的 trajectory (0=禁用)
    """
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # 加载 random baseline
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # 评估每个任务
    results = {}
    centered_results = {}
    timing_records = []

    print("\n" + "="*80)
    print("Running CORE Evaluation")
    print("="*80 + "\n")

    task_pbar = tqdm(tasks, desc="Tasks", position=0, unit="task")
    for task_idx, task in enumerate(task_pbar):
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()
        label = task['label']
        task_pbar.set_description(f"[{task_idx+1}/{len(tasks)}] {label}")

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        tqdm.write(f"[{task_idx+1}/{len(tasks)}] Evaluating: {label}")
        tqdm.write(f"  Type: {task_meta['task_type']} | Few-shot: {task_meta['num_fewshot']}")

        # 加载数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 确定该任务使用的样本数
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]
            tqdm.write(f"  Using task-specific sample limit: {task_max_samples}")

        # 子采样（用于快速测试）
        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]
            tqdm.write(f"  Using {len(data)} samples (subsampled)")
        else:
            tqdm.write(f"  Total samples: {len(data)} (evaluating ALL samples)")

        # 评估（使用带 trajectory 的版本）
        accuracy, _ = evaluate_task_with_trajectory(
            model, tokenizer, data, device, task_meta,
            output_dir=output_dir, task_label=label,
            num_trajectory_samples=num_trajectory_samples,
            show_progress=True, progress_position=1)
        results[label] = accuracy

        # 计算 centered result
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        timing_records.append({
            "task": label,
            "start_time": task_start_iso,
            "end_time": task_end_iso,
            "duration_seconds": round(elapsed, 2),
            "num_samples": len(data),
        })
        tqdm.write(f"  -> Accuracy: {accuracy:.4f} | Centered: {centered_result:.4f} | Time: {elapsed:.2f}s\n")

    task_pbar.close()

    # 计算 CORE metric
    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _greedy_assign_tasks(all_tasks, num_gpus, history_timing):
    """
    贪心调度：按预估耗时降序排列任务，每次将最重的任务分给累计负载最轻的 GPU。
    如果没有历史 timing 数据，fall back 到 round-robin。

    Args:
        all_tasks: 任务列表 (core.yaml 中的 icl_tasks)
        num_gpus: GPU 数量
        history_timing: dict, 历史 timing 数据 (timing_summary.json 的内容), 或 None

    Returns:
        dict: {rank: [task_index, ...]} — 每个 GPU 分配的任务索引列表
    """
    # 尝试从历史 timing 中提取每个任务的耗时
    per_task_duration = {}
    if history_timing is not None:
        per_task = history_timing.get("per_task", {})
        if not per_task:
            # Try flat list / records format
            if isinstance(history_timing, list):
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing if "task" in rec}
            elif "timing" in history_timing:
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing["timing"] if "task" in rec}
            elif "records" in history_timing:
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing["records"] if "task" in rec}

        for label, val in per_task.items():
            if isinstance(val, dict):
                per_task_duration[label] = val.get("duration_seconds", 0)
            else:
                per_task_duration[label] = float(val)

    # 如果没有任何历史 timing，fall back 到 round-robin
    if not per_task_duration:
        assignment = {r: [] for r in range(num_gpus)}
        for i in range(len(all_tasks)):
            assignment[i % num_gpus].append(i)
        return assignment, False  # False = 没有使用贪心

    # 贪心调度
    # 1. 为每个任务获取预估耗时 (未见过的任务用中位数)
    known_durations = list(per_task_duration.values())
    median_duration = sorted(known_durations)[len(known_durations) // 2] if known_durations else 60.0

    task_costs = []
    for i, task in enumerate(all_tasks):
        label = task['label']
        cost = per_task_duration.get(label, median_duration)
        task_costs.append((i, label, cost))

    # 2. 按耗时降序排列 (最重的任务优先分配)
    task_costs.sort(key=lambda x: x[2], reverse=True)

    # 3. 贪心分配：每次将最重的任务分给当前累计负载最轻的 GPU
    # Min-heap of (cumulative_cost, rank)
    gpu_heap = [(0.0, r) for r in range(num_gpus)]
    heapq.heapify(gpu_heap)
    assignment = {r: [] for r in range(num_gpus)}

    for task_idx, label, cost in task_costs:
        min_cost, min_rank = heapq.heappop(gpu_heap)
        assignment[min_rank].append(task_idx)
        heapq.heappush(gpu_heap, (min_cost + cost, min_rank))

    return assignment, True  # True = 使用了贪心调度


def _eval_worker(rank, world_size, ckpt_path, tokenizer_path, dtype,
                 eval_bundle_dir, max_per_task, task_samples,
                 all_tasks, random_baselines, result_dict,
                 timing_list, output_dir, num_trajectory_samples,
                 per_rank_indices=None):
    """多 GPU 评估的 worker 进程

    Args:
        per_rank_indices: dict {rank: [task_index, ...]}，由主进程预先计算好的
                          任务分配 (贪心调度)。如果为 None，fall back 到 round-robin。
    """
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('medium')

    # 使用预计算的任务分配，或 fall back 到 round-robin
    if per_rank_indices is not None and rank in per_rank_indices:
        task_indices = list(per_rank_indices[rank])
    else:
        task_indices = [i for i in range(len(all_tasks)) if i % world_size == rank]

    print(f"[GPU {rank}] Loading model... ({len(task_indices)} tasks assigned)")
    sys.stdout.flush()
    model, tokenizer, hparams = load_ebt_model(ckpt_path, tokenizer_path, device, dtype=dtype)

    data_base_path = os.path.join(eval_bundle_dir, "eval_data")

    for task_idx in task_indices:
        task = all_tasks[task_idx]
        label = task['label']
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print(f"[GPU {rank}] [{task_idx+1}/{len(all_tasks)}] Evaluating: {label}")
        sys.stdout.flush()

        # 加载数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 确定样本数
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]

        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]

        # 评估（使用带 trajectory 的版本，不显示 tqdm，trajectory 写到 rank 后缀文件）
        traj_file = None
        if num_trajectory_samples > 0 and output_dir:
            traj_dir = os.path.join(output_dir, "trajectories", label)
            os.makedirs(traj_dir, exist_ok=True)
            traj_path = os.path.join(traj_dir, f"samples_rank{rank}.jsonl")
            traj_file = open(traj_path, 'w', encoding='utf-8')

        n = len(data)
        correct = 0
        try:
            for idx in range(n):
                if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                    is_correct, traj = evaluate_example_with_trajectory(
                        idx, model, tokenizer, data, device, task_meta)
                    if traj_file:
                        traj_file.write(json.dumps(traj, ensure_ascii=False) + '\n')
                        traj_file.flush()
                else:
                    is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
                correct += int(is_correct)
        finally:
            if traj_file:
                traj_file.close()

        accuracy = correct / n if n > 0 else 0.0

        # centered result
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        print(f"[GPU {rank}]   -> {label}: acc={accuracy:.4f} centered={centered_result:.4f} ({elapsed:.2f}s)")
        sys.stdout.flush()

        result_dict[label] = (accuracy, centered_result)
        timing_list.append({
            "task": label,
            "start_time": task_start_iso,
            "end_time": task_end_iso,
            "duration_seconds": round(elapsed, 2),
            "num_samples": n,
            "gpu_rank": rank,
        })


def evaluate_core_multigpu(num_gpus, ckpt_path, tokenizer_path, dtype,
                           eval_bundle_dir, max_per_task, task_samples,
                           output_dir=None, num_trajectory_samples=10,
                           history_timing_path=None):
    """多 GPU 并行评估：使用贪心调度按预估耗时均衡分配任务到各 GPU"""
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    all_tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    # 加载历史 timing 用于贪心调度
    history_timing = None
    if history_timing_path and os.path.isfile(history_timing_path):
        try:
            with open(history_timing_path, 'r', encoding='utf-8') as f:
                history_timing = json.load(f)
        except Exception as e:
            print(f"  Warning: could not load history timing for scheduling: {e}")

    # 贪心调度 (或 fall back 到 round-robin)
    assignment, used_greedy = _greedy_assign_tasks(all_tasks, num_gpus, history_timing)

    print(f"\n{'='*80}")
    scheduling_method = "greedy load-balanced" if used_greedy else "round-robin (no history timing)"
    print(f"Running CORE Evaluation (multi-GPU: {num_gpus} GPUs, {len(all_tasks)} tasks)")
    print(f"Task scheduling: {scheduling_method}")
    print(f"{'='*80}")
    for rank in range(num_gpus):
        task_names = [all_tasks[i]['label'] for i in assignment[rank]]
        # 显示预估耗时 (如果有历史 timing)
        if used_greedy and history_timing is not None:
            per_task = history_timing.get("per_task", {})
            if not per_task and "records" in history_timing:
                per_task = {rec["task"]: rec for rec in history_timing["records"] if "task" in rec}
            total_est = 0
            for idx in assignment[rank]:
                label = all_tasks[idx]['label']
                val = per_task.get(label, {})
                if isinstance(val, dict):
                    total_est += val.get("duration_seconds", 0)
                else:
                    total_est += float(val)
            print(f"  GPU {rank}: {len(task_names)} tasks (est. {total_est:.0f}s = {total_est/60:.1f}min) -- {', '.join(task_names)}")
        else:
            print(f"  GPU {rank}: {len(task_names)} tasks -- {', '.join(task_names)}")
    print()

    # 共享结果字典和 timing 列表
    manager = mp.Manager()
    result_dict = manager.dict()
    timing_list = manager.list()

    # 将每个 rank 的任务索引列表转为 manager.list 以便跨进程传递
    per_rank_indices = manager.dict()
    for rank in range(num_gpus):
        per_rank_indices[rank] = assignment[rank]

    # 使用 mp.spawn — 传递预计算好的任务分配
    mp.spawn(
        _eval_worker,
        args=(num_gpus, ckpt_path, tokenizer_path, dtype,
              eval_bundle_dir, max_per_task, task_samples,
              all_tasks, random_baselines, result_dict,
              timing_list, output_dir, num_trajectory_samples,
              per_rank_indices),
        nprocs=num_gpus,
        join=True,
    )

    # 合并各 rank 的 trajectory JSONL → samples.jsonl
    if num_trajectory_samples > 0 and output_dir:
        traj_base = os.path.join(output_dir, "trajectories")
        if os.path.isdir(traj_base):
            for task_dir_name in os.listdir(traj_base):
                task_dir = os.path.join(traj_base, task_dir_name)
                if not os.path.isdir(task_dir):
                    continue
                rank_files = sorted(glob_module.glob(os.path.join(task_dir, "samples_rank*.jsonl")))
                if rank_files:
                    merged_path = os.path.join(task_dir, "samples.jsonl")
                    with open(merged_path, 'w', encoding='utf-8') as out_f:
                        for rf in rank_files:
                            with open(rf, 'r', encoding='utf-8') as in_f:
                                for line in in_f:
                                    out_f.write(line)
                    # Clean up per-rank files
                    for rf in rank_files:
                        os.remove(rf)

    # 汇总结果
    results = {}
    centered_results = {}
    for label, (acc, centered) in result_dict.items():
        results[label] = acc
        centered_results[label] = centered

    core_metric = sum(centered_results.values()) / len(centered_results) if centered_results else 0.0

    # 收集 timing
    timing_records = list(timing_list)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _find_latest_timing(ebt_dir):
    """在 {ebt_dir}/logs/core_eval/ 下查找最新的 timing_summary.json"""
    search_dir = os.path.join(ebt_dir, "logs", "core_eval")
    if not os.path.isdir(search_dir):
        return None
    candidates = sorted(glob_module.glob(os.path.join(search_dir, "**/timing_summary.json"), recursive=True))
    if not candidates:
        return None
    # Return the latest by modification time
    return max(candidates, key=os.path.getmtime)


def _print_eta_table(history_timing_path, tasks):
    """Load historical timing and print an ETA estimation table."""
    try:
        with open(history_timing_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load history timing: {e}")
        return

    per_task = history.get("per_task", {})
    if not per_task:
        # Try flat list format
        if isinstance(history, list):
            per_task = {rec["task"]: rec["duration_seconds"] for rec in history if "task" in rec}
        elif "timing" in history:
            per_task = {rec["task"]: rec["duration_seconds"] for rec in history["timing"] if "task" in rec}

    if not per_task:
        print(f"  Warning: no per-task timing found in {history_timing_path}")
        return

    task_labels = [t['label'] for t in tasks]
    total_est = 0
    print(f"\n  ETA Estimation (based on {history_timing_path}):")
    print(f"  {'Task':<40s} {'Est. Time':>10s}")
    print(f"  {'-'*40} {'-'*10}")
    for label in task_labels:
        est = per_task.get(label, {})
        if isinstance(est, dict):
            dur = est.get("duration_seconds", 0)
        else:
            dur = float(est)
        total_est += dur
        print(f"  {label:<40s} {dur:>8.1f}s")
    print(f"  {'-'*40} {'-'*10}")
    print(f"  {'TOTAL':<40s} {total_est:>8.1f}s  ({total_est/60:.1f}min)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="EBT CORE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 评估所有样本（完整 CORE score）:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task -1

2. 快速测试（每个任务 100 个样本）:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100

3. 为特定任务设置不同样本数:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100 \\
       --task-samples "hellaswag_zeroshot:500,arc_easy:200"

4. 使用多个 GPU:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --gpus 4

5. 保存 trajectory（每个任务前 20 个样本）:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --num-trajectory-samples 20
        """
    )
    parser.add_argument('--ckpt-path', type=str, required=True, help='EBT checkpoint path')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--eval-bundle-dir', type=str, required=True, help='Path to eval_bundle directory')
    parser.add_argument('--eval-modes', type=str, default='core', help='Evaluation modes (default: core)')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per task globally (-1 = all samples for complete CORE score)')
    parser.add_argument('--task-samples', type=str, default='',
                        help='Task-specific sample limits (format: "task1:num1,task2:num2")')
    parser.add_argument('--device-batch-size', type=int, default=16, help='Batch size per device')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs to use (-1 = auto-detect all available GPUs)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                        help='Model dtype for inference (default: bfloat16)')
    parser.add_argument('--num-trajectory-samples', type=int, default=10,
                        help='Number of trajectory samples to save per task (0 = disable)')
    parser.add_argument('--history-timing', type=str, default='',
                        help='Path to historical timing_summary.json for ETA estimation (auto-detect if empty)')
    args = parser.parse_args()

    # 解析任务特定样本数
    task_samples_dict = {}
    if args.task_samples:
        for item in args.task_samples.split(','):
            if ':' in item:
                task_name, num_samples = item.split(':', 1)
                task_samples_dict[task_name.strip()] = int(num_samples.strip())

    # 设置设备和 GPU
    if args.gpus == -1:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Auto-detected {num_gpus} GPU(s)")
        else:
            num_gpus = 0
            print("No GPU detected, using CPU")
    else:
        num_gpus = args.gpus
        print(f"Using {num_gpus} GPU(s) as specified")

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    # 性能优化: 启用 TF32 (H200/A100 Tensor Core 加速)
    torch.set_float32_matmul_precision('medium')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ETA 预估：加载历史 timing
    ebt_dir = str(Path(__file__).parent.parent)
    history_timing_path = args.history_timing
    if not history_timing_path:
        history_timing_path = _find_latest_timing(ebt_dir)
    if history_timing_path and os.path.isfile(history_timing_path):
        print(f"\nFound historical timing: {history_timing_path}")
        # Load task list for ETA table
        config_path = os.path.join(args.eval_bundle_dir, "core.yaml")
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f)
            _print_eta_table(history_timing_path, _config['icl_tasks'])
    else:
        print("\nNo historical timing found (first run or --history-timing not specified)")

    eval_start_time = time.time()
    eval_start_iso = datetime.now().isoformat()

    # 多 GPU 路径
    if num_gpus > 1 and 'core' in args.eval_modes:
        print(f"\nUsing multi-GPU evaluation with {num_gpus} GPUs")
        core_results = evaluate_core_multigpu(
            num_gpus=num_gpus,
            ckpt_path=args.ckpt_path,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
            eval_bundle_dir=args.eval_bundle_dir,
            max_per_task=args.max_per_task,
            task_samples=task_samples_dict if task_samples_dict else None,
            output_dir=args.output_dir,
            num_trajectory_samples=args.num_trajectory_samples,
            history_timing_path=history_timing_path,
        )
    else:
        # 单 GPU / CPU 路径
        if num_gpus >= 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        print(f"Using device: {device}\n")

        print("="*80)
        print("Loading EBT Model")
        print("="*80 + "\n")

        model, tokenizer, hparams = load_ebt_model(args.ckpt_path, args.tokenizer_path, device, dtype=dtype)

        core_results = None
        if 'core' in args.eval_modes:
            core_results = evaluate_core(
                model, tokenizer, device, args.eval_bundle_dir,
                max_per_task=args.max_per_task,
                task_samples=task_samples_dict if task_samples_dict else None,
                output_dir=args.output_dir,
                num_trajectory_samples=args.num_trajectory_samples,
            )

    eval_end_time = time.time()
    eval_end_iso = datetime.now().isoformat()
    total_seconds = round(eval_end_time - eval_start_time, 2)

    # 打印和保存结果
    if core_results:
        print("\n" + "="*80)
        print("CORE Evaluation Results")
        print("="*80 + "\n")

        print(f"CORE Metric: {core_results['core_metric']:.4f}\n")

        print("Task Results:")
        for task_name, accuracy in sorted(core_results['results'].items()):
            centered = core_results['centered_results'][task_name]
            print(f"  {task_name:40s}: {accuracy:.4f} (centered: {centered:.4f})")

        # 保存结果 (不含 timing，timing 单独写)
        results_to_save = {
            "results": core_results["results"],
            "centered_results": core_results["centered_results"],
            "core_metric": core_results["core_metric"],
        }
        results_file = os.path.join(args.output_dir, "core_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n-> Results saved to: {results_file}")

        # 保存 CSV
        csv_file = os.path.join(args.output_dir, "core_results.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Accuracy', 'Centered', 'CORE'])
            for task_name in sorted(core_results['results'].keys()):
                writer.writerow([
                    task_name,
                    f"{core_results['results'][task_name]:.4f}",
                    f"{core_results['centered_results'][task_name]:.4f}",
                    ''
                ])
            writer.writerow(['OVERALL', '', '', f"{core_results['core_metric']:.4f}"])
        print(f"-> CSV saved to: {csv_file}")

        # 保存 timing_summary.json
        timing_records = core_results.get("timing", [])
        per_task_timing = {}
        for rec in timing_records:
            per_task_timing[rec["task"]] = {
                "duration_seconds": rec["duration_seconds"],
                "num_samples": rec.get("num_samples", -1),
                "start_time": rec.get("start_time", ""),
                "end_time": rec.get("end_time", ""),
            }
        timing_summary = {
            "total_seconds": total_seconds,
            "total_start": eval_start_iso,
            "total_end": eval_end_iso,
            "num_gpus": num_gpus if num_gpus > 1 else 1,
            "per_task": per_task_timing,
            "records": timing_records,
        }
        timing_file = os.path.join(args.output_dir, "timing_summary.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_summary, f, indent=2)
        print(f"-> Timing saved to: {timing_file}")

        # Trajectory 路径提示
        traj_dir = os.path.join(args.output_dir, "trajectories")
        if os.path.isdir(traj_dir):
            traj_tasks = [d for d in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, d))]
            print(f"-> Trajectories saved for {len(traj_tasks)} tasks in: {traj_dir}/")

        # 机器可解析汇总行
        num_tasks = len(core_results['results'])
        avg_acc = np.mean(list(core_results['results'].values()))
        print(f"\n[EVAL_SUMMARY] dataset=core core_metric={core_results['core_metric']:.4f} avg_accuracy={avg_acc:.4f} num_tasks={num_tasks} total_time={total_seconds:.1f}s")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
