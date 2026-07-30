"""
Sudoku SFT Dataset — 将 Sudoku puzzle/solution 对适配为 EBT SFT 训练格式

复用 dataset_sft.py 的 BOS-aligned bestfit packing 模式，
数据源替换为 Sudoku 对话 (puzzle → solution)。

输出格式与 SFTIterableDataset 完全一致: (inputs, targets) shape (B, T)。
"""

import copy
import os

import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from nanochat.common import get_dist_info


DEFAULT_DATA_DIR = os.environ.get(
    "SUDOKU_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "sudoku_cache"),
)


def board_to_conversation(puzzle, solution):
    """Convert 9x9 puzzle/solution to nanochat chat conversation format."""
    puzzle_str = "\n".join(" ".join(str(c) for c in row) for row in puzzle)
    solution_str = "\n".join(" ".join(str(c) for c in row) for row in solution)
    return {
        "messages": [
            {"role": "user", "content": f"Solve this sudoku:\n{puzzle_str}"},
            {"role": "assistant", "content": solution_str},
        ]
    }


def _load_sudoku_split(split, data_dir):
    """Load cached sudoku data for a given split."""
    split_files = {
        "train": "satnet_train.pt",
        "val": "satnet_val.pt",
        "test": "rrn_test.pt",
    }
    if split not in split_files:
        raise ValueError(f"Unknown split '{split}', expected one of {list(split_files)}")
    path = os.path.join(data_dir, split_files[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sudoku cache not found: {path}\n"
            f"Run: python openebm/elm/data/prepare_sudoku_data.py --data_dir {data_dir}"
        )
    samples = torch.load(path, weights_only=False)
    print(f"[Sudoku] Loaded {len(samples)} samples from {path}")
    return samples


class SudokuSFTIterableDataset(_IterableDataset):
    """Sudoku SFT dataset with BOS-aligned bestfit packing (mirrors SFTIterableDataset)."""

    STATE_VERSION = "sudoku_sft_v1"

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
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = self._can_restore(resume_state_dict)
        self.last_state_dict = None
        self._runtime_initialized = False

        self.row_capacity = self.T + 1
        self.buffer_size = 200  # Sudoku dataset is smaller than SFT mix
        self.conv_buffer = []
        self.cursor = None
        self.consumed = None
        self.epoch = 1
        self.it = 0
        self.ddp_rank = None
        self.ddp_world_size = None

        # Load data eagerly (small dataset, fits in memory)
        effective_split = "val" if split == "val" else "train"
        self.samples = _load_sudoku_split(effective_split, self.data_dir)

    def _can_restore(self, state):
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0

    def _restore_state(self, state, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = int(state.get("row_capacity", self.T + 1))
        self.conv_buffer = copy.deepcopy(state.get("conv_buffer", []))
        self.cursor = int(state.get("cursor", ddp_rank))
        self.consumed = int(state.get("consumed", ddp_rank))
        self.epoch = int(state.get("epoch", 1))
        self.it = int(state.get("it", 0))
        print(
            f"[Sudoku SFT Resume] Rank {ddp_rank}: "
            f"cursor={self.cursor}, consumed={self.consumed}, epoch={self.epoch}, "
            f"it={self.it}, conv_buffer={len(self.conv_buffer)}"
        )

    def _build_state_dict(self, copy_buffer):
        return {
            "state_version": self.STATE_VERSION,
            "split": self.split,
            "cursor": self.cursor,
            "consumed": self.consumed,
            "epoch": self.epoch,
            "it": self.it,
            "buffer_size": self.buffer_size,
            "row_capacity": self.row_capacity,
            "rank": self.ddp_rank,
            "world_size": self.ddp_world_size,
            "conv_buffer": copy.deepcopy(self.conv_buffer) if copy_buffer else self.conv_buffer,
        }

    def __iter__(self):
        """BOS-aligned bestfit packing, mirroring SFTIterableDataset.__iter__"""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset_size = len(self.samples)
        assert dataset_size > 0, f"Empty sudoku dataset for split={self.split}"

        bos_token = self.tokenizer.get_bos_token_id()
        use_cuda = "cuda" in str(self.device)

        if self.split == "train":
            if not self._runtime_initialized:
                if self._can_restore(self.resume_state_dict):
                    self._restore_state(self.resume_state_dict, ddp_rank, ddp_world_size)
                else:
                    self._initialize_fresh_state(ddp_rank, ddp_world_size)
                self._runtime_initialized = True
        else:
            self._initialize_fresh_state(ddp_rank, ddp_world_size)

        def refill_buffer():
            while len(self.conv_buffer) < self.buffer_size:
                sample = self.samples[self.cursor % dataset_size]
                conversation = board_to_conversation(sample["puzzle"], sample["solution"])
                ids, _ = self.tokenizer.render_conversation(conversation)
                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append(ids)

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_lengths = []

            for _ in range(self.B):
                row = []
                padded = False
                while len(row) < self.row_capacity:
                    while len(self.conv_buffer) < self.buffer_size:
                        refill_buffer()
                    remaining = self.row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, conv in enumerate(self.conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv = self.conv_buffer.pop(best_idx)
                        row.extend(conv)
                        self.consumed += ddp_world_size
                    else:
                        content_len = len(row)
                        remaining = self.row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        padded = True
                        break

                if padded:
                    row_lengths.append(content_len)
                else:
                    row_lengths.append(self.row_capacity)
                rows.append(row[:self.row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            for i, content_len in enumerate(row_lengths):
                if content_len < self.row_capacity:
                    targets[i, content_len - 1:] = -1

            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)
            yield inputs, targets

            if self.split != "train" and self.it >= self.max_iter:
                break

    def get_dataloader_state(self):
        if self.cursor is None:
            return self.last_state_dict
        return self._build_state_dict(copy_buffer=True)

    def state_dict(self):
        state = self.get_dataloader_state()
        return {} if state is None else state

    def load_state_dict(self, state_dict):
        if self._resume_state_locked and self._can_restore(self.resume_state_dict):
            current_state = self.resume_state_dict
            incoming_state = state_dict
            if current_state != incoming_state:
                print(
                    f"[Sudoku SFT Resume] Ignore external dataloader overwrite: "
                    f"current_rank={current_state.get('rank')}, "
                    f"incoming_rank={incoming_state.get('rank') if isinstance(incoming_state, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        self._runtime_initialized = False

    def __len__(self):
        return self.max_iter


class SudokuSFTDataLoader(DataLoader):
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


def generate_sudoku_sft_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device,
    data_dir=None, resume_state_dict=None,
):
    """Factory function matching generate_sft_dataloader signature."""
    dataset = SudokuSFTIterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        data_dir=data_dir,
        resume_state_dict=resume_state_dict,
    )
    dataloader = SudokuSFTDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
