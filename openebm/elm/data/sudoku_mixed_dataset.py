"""
Mixed Sudoku + nanochat SFT Dataset
====================================

混合训练：每个 batch 以 60% 概率从 Sudoku v2 数据集采样，40% 从 nanochat SFT 采样。

这样做的目的：
  - 引入通用 SFT 数据防止 Sudoku 过拟合
  - 保留通用语言能力
  - 让模型学到 "对话结束应输出 <|assistant_end|>" 的通用模式

输出格式与 SFTIterableDataset 完全一致: (inputs, targets) shape (B, T)。

v3:
  - `sudoku_ratio` is a public mutable attribute. AdaptiveRatioCallback may rewrite
    it on the fly (the iterable dataset path uses num_workers=0, so attribute mutation
    is visible to the next __iter__ tick without IPC).
  - `last_batch_source` records the source of the most recently yielded batch
    ('sudoku' or 'sft') so the trainer can route per-source-only logic
    (blank-position weighted CE on sudoku batches; SFT-loss-driven retention guard).
"""

import copy
import os
import random

from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from openebm.elm.data.sudoku_dataset_v2 import (
    SudokuSFTV2IterableDataset,
    DEFAULT_DATA_DIR as DEFAULT_SUDOKU_V2_DIR,
)
from openebm.elm.dataset_sft import SFTIterableDataset


DEFAULT_SUDOKU_RATIO = 0.6


class SudokuMixedIterableDataset(_IterableDataset):
    """60% Sudoku v2 + 40% nanochat SFT 混合数据集."""

    STATE_VERSION = "sudoku_mixed_v1"

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        data_dir=None,
        resume_state_dict=None,
        sudoku_ratio=DEFAULT_SUDOKU_RATIO,
        seed=None,
        blank_loss_weight=1.0,
        difficulty_bucket_weights=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.data_dir = data_dir or DEFAULT_SUDOKU_V2_DIR
        if isinstance(resume_state_dict, dict) and "sudoku_ratio" in resume_state_dict:
            sudoku_ratio = float(resume_state_dict["sudoku_ratio"])
        # v3: public mutable; AdaptiveRatioCallback may rewrite this between batches.
        self.sudoku_ratio = sudoku_ratio
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = isinstance(resume_state_dict, dict) and (
            resume_state_dict.get("state_version") == self.STATE_VERSION
            or "sudoku_state" in (resume_state_dict or {})
        )
        self.last_state_dict = None
        self._runtime_initialized = False
        self._seed = seed
        self.it = int(resume_state_dict.get("it", 0)) if isinstance(resume_state_dict, dict) else 0
        self._mix_rng = None
        self._mix_rng_state = (
            copy.deepcopy(resume_state_dict.get("mix_rng_state"))
            if isinstance(resume_state_dict, dict)
            else None
        )
        # v3: source of the most-recently yielded batch ('sudoku' | 'sft').
        self.last_batch_source = (
            resume_state_dict.get("last_batch_source")
            if isinstance(resume_state_dict, dict)
            else None
        )

        # 预估两侧 iterator 的 max_iter 上界。因为按概率采样，每侧实际用量 ≈ max_iter * ratio。
        # 给 1.5 倍缓冲避免 StopIteration（底层 iterator 对 train split 是循环的）
        sudoku_budget = max(1, int(max_iter * sudoku_ratio * 1.5))
        sft_budget = max(1, int(max_iter * (1.0 - sudoku_ratio) * 1.5))

        # 拆分 resume_state_dict 给两侧
        inner_sudoku_state = None
        inner_sft_state = None
        if isinstance(resume_state_dict, dict):
            inner_sudoku_state = resume_state_dict.get("sudoku_state", None)
            inner_sft_state = resume_state_dict.get("sft_state", None)

        self.sudoku_ds = SudokuSFTV2IterableDataset(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            split=split,
            max_iter=sudoku_budget,
            device=device,
            data_dir=self.data_dir,
            resume_state_dict=inner_sudoku_state,
            augment=True,
            seed=seed,
            blank_loss_weight=blank_loss_weight,
            difficulty_bucket_weights=difficulty_bucket_weights,
        )
        self.sft_ds = SFTIterableDataset(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            split=split,
            max_iter=sft_budget,
            device=device,
            resume_state_dict=inner_sft_state,
        )

    # ── State management ───────────────────────────────────────────────

    def _build_state_dict(self):
        """Combine child states with our mix counter."""
        mix_rng_state = (
            self._mix_rng.getstate()
            if self._mix_rng is not None
            else copy.deepcopy(self._mix_rng_state)
        )
        return {
            "state_version": self.STATE_VERSION,
            "split": self.split,
            "it": self.it,
            "sudoku_ratio": self.sudoku_ratio,
            "mix_rng_state": mix_rng_state,
            "last_batch_source": self.last_batch_source,
            "sudoku_state": self.sudoku_ds.state_dict(),
            "sft_state": self.sft_ds.state_dict(),
        }

    def get_dataloader_state(self):
        return self._build_state_dict()

    def state_dict(self):
        return self._build_state_dict()

    def load_state_dict(self, state_dict):
        if self._resume_state_locked and isinstance(self.resume_state_dict, dict):
            current_state = self.resume_state_dict
            if current_state != state_dict:
                print(
                    f"[Sudoku Mixed Resume] Ignore external dataloader overwrite: "
                    f"current_it={current_state.get('it')}, "
                    f"incoming_it={state_dict.get('it') if isinstance(state_dict, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        if isinstance(state_dict, dict):
            if "sudoku_ratio" in state_dict:
                self.sudoku_ratio = float(state_dict["sudoku_ratio"])
            self._mix_rng_state = copy.deepcopy(state_dict.get("mix_rng_state"))
            self._mix_rng = None
            self.last_batch_source = state_dict.get("last_batch_source")
            if "sudoku_state" in state_dict:
                self.sudoku_ds.load_state_dict(state_dict["sudoku_state"])
            if "sft_state" in state_dict:
                self.sft_ds.load_state_dict(state_dict["sft_state"])
            self.it = int(state_dict.get("it", 0))
        self._runtime_initialized = False

    def _init_mix_rng(self):
        base_seed = self._seed if self._seed is not None else 20260508
        rng = random.Random(base_seed + 9973)
        if self._mix_rng_state is not None:
            rng.setstate(self._mix_rng_state)
        else:
            for _ in range(max(0, int(self.it))):
                rng.random()
        self._mix_rng = rng

    # ── Iteration ──────────────────────────────────────────────────────

    def __iter__(self):
        sudoku_iter = iter(self.sudoku_ds)
        sft_iter = iter(self.sft_ds)
        # 用独立 rng 决定每步从哪侧采样，避免干扰底层数据增强 rng
        if self._mix_rng is None:
            self._init_mix_rng()

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            # v3: re-read mutable sudoku_ratio every step so AdaptiveRatioCallback
            # mutations take effect on the next batch.
            current_ratio = float(self.sudoku_ratio)
            pick_sudoku = self._mix_rng.random() < current_ratio
            try:
                if pick_sudoku:
                    batch = next(sudoku_iter)
                    chosen = 'sudoku'
                else:
                    batch = next(sft_iter)
                    chosen = 'sft'
            except StopIteration:
                # 回退到另一侧，尽量不中断训练
                try:
                    if pick_sudoku:
                        batch = next(sft_iter)
                        chosen = 'sft'
                    else:
                        batch = next(sudoku_iter)
                        chosen = 'sudoku'
                except StopIteration:
                    break

            self.it += 1
            self.last_batch_source = chosen
            self.last_state_dict = self._build_state_dict()
            yield batch

            if self.split != "train" and self.it >= self.max_iter:
                break

    def __len__(self):
        return self.max_iter


class SudokuMixedDataLoader(DataLoader):
    """DataLoader wrapper that delegates state_dict to the underlying dataset."""

    def state_dict(self):
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict):
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sudoku_mixed_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device,
    data_dir=None, resume_state_dict=None, sudoku_ratio=DEFAULT_SUDOKU_RATIO, seed=None,
    blank_loss_weight=1.0, difficulty_bucket_weights=None,
):
    """Factory function for mixed Sudoku v2 + nanochat SFT dataloader.

    v3 extras (forwarded to the inner Sudoku v2 dataset):
      - blank_loss_weight: K for blank-position loss weighting (1.0 = off).
      - difficulty_bucket_weights: per-bucket sampling weights aligned with
        SudokuSFTV2IterableDataset.DIFFICULTY_BUCKETS.
    """
    dataset = SudokuMixedIterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        data_dir=data_dir,
        resume_state_dict=resume_state_dict,
        sudoku_ratio=sudoku_ratio,
        seed=seed,
        blank_loss_weight=blank_loss_weight,
        difficulty_bucket_weights=difficulty_bucket_weights,
    )
    dataloader = SudokuMixedDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
