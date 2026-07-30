"""
Sudoku SFT Dataset V2 — 数据多样化 + 对称增强
================================================

相对于 v1 的改进：
  1. **自然语言 prompt 多样性**：5 种 prompt 模板 + 3 种 response 模板随机组合
  2. **On-the-fly 对称增强**：digit relabel / band perm / stack perm / row-within-band /
     col-within-stack / transpose —— 所有变换保持数独合法性
  3. **版本隔离**：默认缓存目录 sudoku_cache_v2/，不干扰 v1

输出格式与 SFTIterableDataset 完全一致: (inputs, targets) shape (B, T)。
"""

import copy
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from nanochat.common import get_dist_info


DEFAULT_DATA_DIR = os.environ.get(
    "SUDOKU_DATA_DIR_V2",
    os.path.join(os.path.dirname(__file__), "sudoku_cache_v2"),
)


# ── Prompt / Response 多样化 (v3: 5→20 prompt, 3→8 response, + flat format) ─

# v3: 20 prompt templates spanning tone (terse / conversational / formal / role-played),
# phrasing (declarative / imperative / Q&A / explanatory), and language (en majority + zh-CN).
# Each MUST contain literal `{puzzle}` — selftest below asserts this.
PROMPT_TEMPLATES = [
    # --- v2 originals (kept for back-compat with existing training distribution) ---
    "Solve this sudoku puzzle. Replace each 0 with the correct digit (1-9):\n{puzzle}",
    "Complete the following sudoku grid. Empty cells are marked as 0:\n{puzzle}",
    "Fill in the missing numbers in this 9x9 sudoku:\n{puzzle}",
    "Here is a sudoku puzzle where 0 represents an empty cell. Find the solution:\n{puzzle}",
    "I need help solving this sudoku. Each row, column, and 3x3 box must contain digits 1-9 exactly once:\n{puzzle}",
    # --- terse / imperative ---
    "Sudoku:\n{puzzle}\nSolve.",
    "Fill in the blanks (0 = empty):\n{puzzle}",
    "Solve:\n{puzzle}",
    # --- declarative / conversational ---
    "Can you complete this sudoku puzzle for me? The 0s are empty cells.\n{puzzle}",
    "I'm stuck on this sudoku. Could you finish it?\n{puzzle}",
    "Please solve the sudoku below. Use 1-9 to replace every 0.\n{puzzle}",
    # --- formal ---
    "Given the following sudoku grid where 0 denotes an unknown cell, determine the unique completion such that every row, column, and 3x3 sub-grid contains the digits 1 through 9 exactly once.\n{puzzle}",
    "Problem: complete the sudoku puzzle below.\nConstraints: digits 1-9 must appear exactly once per row, per column, and per 3x3 box.\nInput grid (0 = blank):\n{puzzle}",
    # --- role-play ---
    "You are a sudoku tutor. Walk-through is not required — just give the completed grid.\nPuzzle:\n{puzzle}",
    "You are an expert puzzle solver. Solve this sudoku and return only the answer.\n{puzzle}",
    # --- Q&A ---
    "Question: what is the solution to the following sudoku?\n{puzzle}\nAnswer:",
    "Q: Solve the puzzle below.\n{puzzle}\nA:",
    # --- explanatory / hint context ---
    "Below is an unsolved sudoku puzzle. Each blank cell is marked with 0. Output the fully-completed grid.\n{puzzle}",
    # --- zh-CN (preserves legacy LEGACY_ZH_PROMPT style) ---
    "请解决以下数独：每行、每列、每个 3x3 宫格中需要填入 1-9 各一次（0 表示空格）。\n{puzzle}",
    "请补全这个 9x9 数独，0 代表空格。\n{puzzle}",
]

# v3: 8 response templates covering terse, "Answer:", explanatory, multi-line, JSON-ish.
# Each MUST contain literal `{solution}` — selftest below asserts this.
RESPONSE_TEMPLATES = [
    # --- v2 originals ---
    "{solution}",
    "Here is the completed sudoku:\n{solution}",
    "The solution is:\n{solution}",
    # --- new in v3 ---
    "Answer:\n{solution}",
    "Solution:\n{solution}",
    "Sure, here it is:\n{solution}",
    "The completed grid is below.\n{solution}",
    # JSON-ish (still contains literal `{solution}` substring)
    "{{\"solution\": \"\n{solution}\n\"}}",
]

# v3: Two puzzle/solution rendering formats. `grid` = 9 lines × 9 space-separated digits.
# `flat` = single line of 81 digits (no spaces, no newlines). Default mix 70% grid / 30% flat.
PUZZLE_FORMATS = ['grid', 'flat']
DEFAULT_FORMAT_MIX = {'grid': 0.7, 'flat': 0.3}


def _format_board_grid(board):
    """9 lines × 9 space-separated digits (legacy v2 format)."""
    return "\n".join(" ".join(str(c) for c in row) for row in board)


def _format_board_flat(board):
    """Single line of 81 digits, no spaces, no newlines (v3 spaceless format)."""
    return "".join(str(c) for row in board for c in row)


def _format_board(board, fmt='grid'):
    """Dispatch by format string."""
    if fmt == 'flat':
        return _format_board_flat(board)
    return _format_board_grid(board)


def _pick_format(rng):
    """Sample a format from PUZZLE_FORMATS using DEFAULT_FORMAT_MIX."""
    r = rng.random()
    cum = 0.0
    for fmt in PUZZLE_FORMATS:
        cum += DEFAULT_FORMAT_MIX.get(fmt, 0.0)
        if r < cum:
            return fmt
    return PUZZLE_FORMATS[0]


def board_to_conversation_v2(puzzle, solution, rng=None):
    """Convert 9x9 puzzle/solution to nanochat chat conversation with random templates.

    v3: independently samples a prompt template, response template, puzzle format, and
    solution format per call. Returns (conversation, meta) where `meta` records the
    indices/formats so dataset code can log per-format stats if desired.
    """
    if rng is None:
        rng = random
    puzzle_fmt = _pick_format(rng)
    solution_fmt = _pick_format(rng)
    puzzle_str = _format_board(puzzle, puzzle_fmt)
    solution_str = _format_board(solution, solution_fmt)
    prompt_idx = rng.randrange(len(PROMPT_TEMPLATES))
    response_idx = rng.randrange(len(RESPONSE_TEMPLATES))
    prompt_tpl = PROMPT_TEMPLATES[prompt_idx]
    response_tpl = RESPONSE_TEMPLATES[response_idx]
    conv = {
        "messages": [
            {"role": "user", "content": prompt_tpl.format(puzzle=puzzle_str)},
            {"role": "assistant", "content": response_tpl.format(solution=solution_str)},
        ]
    }
    meta = {
        "prompt_template_idx": prompt_idx,
        "response_template_idx": response_idx,
        "puzzle_format": puzzle_fmt,
        "solution_format": solution_fmt,
    }
    return conv, meta


# v3: import-time selftest — fail loudly if a template is missing its placeholder.
assert len(PROMPT_TEMPLATES) == 20, f"v3 spec requires 20 prompts, got {len(PROMPT_TEMPLATES)}"
assert len(RESPONSE_TEMPLATES) == 8, f"v3 spec requires 8 responses, got {len(RESPONSE_TEMPLATES)}"
assert len(PUZZLE_FORMATS) == 2, f"v3 spec requires 2 formats, got {len(PUZZLE_FORMATS)}"
assert all('{puzzle}' in t for t in PROMPT_TEMPLATES), "every prompt must contain literal '{puzzle}'"
assert all('{solution}' in t for t in RESPONSE_TEMPLATES), "every response must contain literal '{solution}'"


# ── 对称增强 ────────────────────────────────────────────────────────────────

def augment_board(puzzle, solution, rng):
    """Apply random validity-preserving symmetry transform.

    Six transforms stacked:
      1. Digit relabel (1-9 permutation)
      2. Band permutation (3 horizontal band blocks)
      3. Row-within-band permutation
      4. Stack permutation (3 vertical stack blocks)
      5. Col-within-stack permutation
      6. Transpose (50% probability)

    Args:
        puzzle:  9x9 nested list, 0 = empty
        solution: 9x9 nested list, 1-9
        rng:     numpy.random.Generator (preferred) or np.random

    Returns:
        (puzzle_aug, solution_aug) as 9x9 nested lists
    """
    puzzle = np.array(puzzle, dtype=np.int64)
    solution = np.array(solution, dtype=np.int64)

    # 1. digit relabel (keeps 0 fixed)
    perm = rng.permutation(9) + 1  # random permutation of [1..9]
    mapping = np.zeros(10, dtype=np.int64)
    mapping[1:] = perm
    puzzle = mapping[puzzle]
    solution = mapping[solution]

    # 2. band permutation (3 horizontal band blocks of 3 rows each)
    band_order = rng.permutation(3)
    puzzle = np.vstack([puzzle[b * 3:(b + 1) * 3] for b in band_order])
    solution = np.vstack([solution[b * 3:(b + 1) * 3] for b in band_order])

    # 3. row-within-band permutation
    for b in range(3):
        row_order = rng.permutation(3)
        puzzle[b * 3:(b + 1) * 3] = puzzle[b * 3:(b + 1) * 3][row_order]
        solution[b * 3:(b + 1) * 3] = solution[b * 3:(b + 1) * 3][row_order]

    # 4. stack permutation (3 vertical stack blocks of 3 cols each)
    stack_order = rng.permutation(3)
    puzzle = np.hstack([puzzle[:, s * 3:(s + 1) * 3] for s in stack_order])
    solution = np.hstack([solution[:, s * 3:(s + 1) * 3] for s in stack_order])

    # 5. col-within-stack permutation
    for s in range(3):
        col_order = rng.permutation(3)
        puzzle[:, s * 3:(s + 1) * 3] = puzzle[:, s * 3:(s + 1) * 3][:, col_order]
        solution[:, s * 3:(s + 1) * 3] = solution[:, s * 3:(s + 1) * 3][:, col_order]

    # 6. transpose (50%)
    if rng.random() < 0.5:
        puzzle = puzzle.T
        solution = solution.T

    return puzzle.tolist(), solution.tolist()


# ── Dataset loader ──────────────────────────────────────────────────────────

def _load_sudoku_split_v2(split, data_dir):
    """Load cached sudoku v2 data for a given split."""
    split_files = {
        "train": "rrn_train.pt",
        "val": "rrn_val.pt",
        "test": "satnet_test.pt",
    }
    if split not in split_files:
        raise ValueError(f"Unknown split '{split}', expected one of {list(split_files)}")
    path = os.path.join(data_dir, split_files[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sudoku v2 cache not found: {path}\n"
            f"Run: python openebm/elm/data/prepare_sudoku_data_v2.py --data_dir {data_dir}"
        )
    samples = torch.load(path, weights_only=False)
    print(f"[Sudoku v2] Loaded {len(samples)} samples from {path}")
    return samples


class SudokuSFTV2IterableDataset(_IterableDataset):
    """Sudoku SFT dataset v2 with prompt diversity + on-the-fly symmetry augmentation.

    v3 additions:
      - blank_loss_weight (K): emit per-token weight tensor; weight=K on positions whose
        target token corresponds to a blank in the original puzzle, =1 elsewhere, =0 for
        masked targets (-1). When K==1.0 (default off) the weight tensor is omitted to
        keep the (inputs, targets) interface byte-compatible with v2 callers.
      - difficulty_schedule + difficulty_bucket_weights: pre-bucket samples by given-count
        into [hard 17-22, medium 23-28, easy 29-34] and sample bucket-by-bucket. Mutable
        attribute (`bucket_weights`) lets a Lightning callback mutate the schedule per step.
    """

    STATE_VERSION = "sudoku_sft_v2"

    DIFFICULTY_BUCKETS = [
        ('hard',   17, 22),
        ('medium', 23, 28),
        ('easy',   29, 34),
    ]

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
        augment=True,
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
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = self._can_restore(resume_state_dict)
        self.last_state_dict = None
        self._runtime_initialized = False
        # v2 新增：训练时开启增强，验证时关闭（确保 valid_loss 可比较）
        self.augment = augment and (split == "train")
        self._seed = seed

        # v3: blank-position loss weighting (P1). K=1.0 means "off" — no weight tensor emitted.
        self.blank_loss_weight = float(blank_loss_weight)

        # v3: difficulty-aware sampling (P5). Mutable so AdaptiveRatioCallback can swap
        # bucket weights per phase. Train-only; val/test use uniform.
        self.bucket_weights = (
            list(difficulty_bucket_weights)
            if difficulty_bucket_weights is not None and split == "train"
            else None
        )

        self.row_capacity = self.T + 1
        self.buffer_size = 500  # v2 RRN 数据更大
        self.conv_buffer = []
        self.cursor = None
        self.consumed = None
        self.epoch = 1
        self.it = 0
        self.ddp_rank = None
        self.ddp_world_size = None
        self._np_rng = None
        self._py_rng = None

        # v2 训练集用 'train'，验证集用 'val'
        effective_split = "val" if split == "val" else "train"
        self.samples = _load_sudoku_split_v2(effective_split, self.data_dir)

        # v3: pre-bucket samples by given_count for difficulty-aware sampling.
        self._bucket_indices = self._compute_difficulty_buckets() if split == "train" else None

    def _can_restore(self, state):
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _compute_difficulty_buckets(self):
        """v3: bucket sample indices by given-count for difficulty-aware resampling (P5).

        Returns a list of np.ndarray (one per bucket in DIFFICULTY_BUCKETS).
        Samples whose given_count falls outside any defined bucket land in the closest
        bucket so we never silently drop data.
        """
        bucket_idx = [[] for _ in self.DIFFICULTY_BUCKETS]
        for i, sample in enumerate(self.samples):
            puzzle = np.asarray(sample["puzzle"], dtype=np.int64)
            given = int((puzzle != 0).sum())
            placed = False
            for bi, (_, lo, hi) in enumerate(self.DIFFICULTY_BUCKETS):
                if lo <= given <= hi:
                    bucket_idx[bi].append(i)
                    placed = True
                    break
            if not placed:
                # Stuff into nearest bucket by midpoint distance.
                mids = [(lo + hi) / 2.0 for _, lo, hi in self.DIFFICULTY_BUCKETS]
                bi = int(np.argmin([abs(given - m) for m in mids]))
                bucket_idx[bi].append(i)
        arrs = [np.array(idx, dtype=np.int64) for idx in bucket_idx]
        sizes = ", ".join(f"{name}({lo}-{hi})={len(arrs[bi])}"
                          for bi, (name, lo, hi) in enumerate(self.DIFFICULTY_BUCKETS))
        print(f"[Sudoku v2] Difficulty buckets: {sizes}")
        return arrs

    def _init_rngs(self, ddp_rank):
        """Init rng with a rank-dependent seed for reproducibility + cross-rank diversity."""
        base_seed = self._seed if self._seed is not None else 20260508
        seed = base_seed + (ddp_rank or 0) * 10007 + self.it
        self._np_rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0
        self._init_rngs(ddp_rank)

    def _restore_state(self, state, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = int(state.get("row_capacity", self.T + 1))
        self.conv_buffer = copy.deepcopy(state.get("conv_buffer", []))
        self.cursor = int(state.get("cursor", ddp_rank))
        self.consumed = int(state.get("consumed", ddp_rank))
        self.epoch = int(state.get("epoch", 1))
        self.it = int(state.get("it", 0))
        self._init_rngs(ddp_rank)
        print(
            f"[Sudoku v2 Resume] Rank {ddp_rank}: "
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

        # v3: cache decoded single-digit token-id set so we can detect cell tokens in
        # assistant range without re-decoding every time. Uses tokenizer.decode on a small
        # set of probe inputs; if a tokenizer collapses digit runs we'll just fall back
        # to uniform weights for that conversation (best-effort).
        single_digit_token_ids = self._build_single_digit_token_set() \
            if self.blank_loss_weight != 1.0 else None

        if self.split == "train":
            if not self._runtime_initialized:
                if self._can_restore(self.resume_state_dict):
                    self._restore_state(self.resume_state_dict, ddp_rank, ddp_world_size)
                else:
                    self._initialize_fresh_state(ddp_rank, ddp_world_size)
                self._runtime_initialized = True
        else:
            self._initialize_fresh_state(ddp_rank, ddp_world_size)

        def pick_sample_index():
            """v3: difficulty-aware bucket sampling on train; uniform-cursor on val/test."""
            if (
                self.split == "train"
                and self.bucket_weights is not None
                and self._bucket_indices is not None
            ):
                weights = list(self.bucket_weights)
                # Filter empty buckets (defensive — shouldn't happen with RRN data).
                non_empty = [(i, w) for i, w in enumerate(weights)
                             if len(self._bucket_indices[i]) > 0 and w > 0]
                if non_empty:
                    idxs, ws = zip(*non_empty)
                    bi = self._py_rng.choices(idxs, weights=ws, k=1)[0]
                    bucket_arr = self._bucket_indices[bi]
                    # Stride by ddp_world_size into the bucket so different ranks see
                    # different samples (cheap pseudo-distribution).
                    stride_pos = (self.cursor // max(ddp_world_size, 1)) % len(bucket_arr)
                    return int(bucket_arr[stride_pos])
            # default: legacy linear cursor over the whole pool
            return self.cursor % dataset_size

        def refill_buffer():
            while len(self.conv_buffer) < self.buffer_size:
                sample_idx = pick_sample_index()
                sample = self.samples[sample_idx]
                puzzle = sample["puzzle"]
                solution = sample["solution"]
                if self.augment:
                    puzzle, solution = augment_board(puzzle, solution, self._np_rng)
                conversation, meta = board_to_conversation_v2(
                    puzzle, solution, rng=self._py_rng,
                )
                ids, mask = self.tokenizer.render_conversation(conversation)

                # v3: build per-token blank-weight list only when K != 1.0.
                blank_weights = None
                if (
                    self.blank_loss_weight != 1.0
                    and single_digit_token_ids is not None
                    and meta.get("solution_format") == "grid"  # safe-aligned format only
                ):
                    blank_weights = self._compute_blank_weights(
                        ids=ids, mask=mask, puzzle=puzzle,
                        digit_token_ids=single_digit_token_ids,
                    )

                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append((ids, blank_weights))

        emit_weights = (self.blank_loss_weight != 1.0)

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_lengths = []
            row_weights = [] if emit_weights else None

            for _ in range(self.B):
                row = []
                row_w = [1.0] * self.row_capacity if emit_weights else None
                padded = False
                while len(row) < self.row_capacity:
                    while len(self.conv_buffer) < self.buffer_size:
                        refill_buffer()
                    remaining = self.row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, (conv_ids, _bw) in enumerate(self.conv_buffer):
                        conv_len = len(conv_ids)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv_ids, conv_w = self.conv_buffer.pop(best_idx)
                        start = len(row)
                        row.extend(conv_ids)
                        if emit_weights and row_w is not None:
                            if conv_w is None:
                                # Conversation came in without a blank-mask (e.g. flat
                                # format). Leave its slice at weight=1.0.
                                pass
                            else:
                                for k, w in enumerate(conv_w):
                                    row_w[start + k] = w
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
                if emit_weights:
                    row_weights.append(row_w[:self.row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            for i, content_len in enumerate(row_lengths):
                if content_len < self.row_capacity:
                    targets[i, content_len - 1:] = -1

            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)

            if emit_weights:
                weights_tensor = torch.tensor(row_weights, dtype=torch.float32, pin_memory=use_cuda)
                # Align with targets (shifted by 1).
                weights_aligned = weights_tensor[:, 1:].to(
                    device=self.device, dtype=torch.float32, non_blocking=use_cuda,
                )
                yield inputs, targets, weights_aligned
            else:
                yield inputs, targets

            if self.split != "train" and self.it >= self.max_iter:
                break

    # ── v3 helpers for blank-position weighting ─────────────────────────

    def _build_single_digit_token_set(self):
        """Best-effort: build a set of token ids that decode to a single-digit char.

        We probe the tokenizer with each digit "1".."9" both bare and with a leading
        space (typical BPE variants). Tokens whose decoded value (stripped) is a single
        digit are considered "cell tokens".
        """
        probe_ids = set()
        try:
            for d in range(1, 10):
                for txt in (f"{d}", f" {d}"):
                    encoded = self.tokenizer.encode(txt)
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    for tok in encoded:
                        try:
                            decoded = self.tokenizer.decode([int(tok)])
                            if decoded.strip() in {str(x) for x in range(0, 10)}:
                                probe_ids.add(int(tok))
                        except Exception:
                            continue
        except Exception:
            pass
        if not probe_ids:
            print("[Sudoku v2] WARNING: failed to probe single-digit token ids; "
                  "blank-position weighting will fall back to uniform.")
            return None
        return probe_ids

    def _compute_blank_weights(self, ids, mask, puzzle, digit_token_ids):
        """Walk assistant-range tokens, identify the 81 cell tokens (in row-major order),
        and emit a per-token weight list aligned with `ids`. Weight = blank_loss_weight
        for cell tokens whose puzzle position was 0; weight = 1 otherwise. Returns None
        if the alignment fails (fewer than 81 detected digit tokens), letting the caller
        leave the row's weights uniform.
        """
        K = float(self.blank_loss_weight)
        weights = [1.0] * len(ids)
        # Flatten puzzle in row-major order; v2 format is row-major as well so cell-i
        # corresponds to the i-th detected digit in the assistant content.
        if hasattr(puzzle, 'tolist'):
            puzzle_flat = []
            for row in puzzle:
                for c in row:
                    puzzle_flat.append(int(c))
        else:
            puzzle_flat = [int(c) for row in puzzle for c in row]
        if len(puzzle_flat) != 81:
            return None

        cell_idx = 0
        for pos, (tok, m) in enumerate(zip(ids, mask)):
            if m != 1:
                continue
            if tok in digit_token_ids:
                if cell_idx < 81:
                    if puzzle_flat[cell_idx] == 0:
                        weights[pos] = K
                    cell_idx += 1
        if cell_idx < 81:
            # Misalignment — better to skip than to mis-weight.
            return None
        return weights

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
                    f"[Sudoku v2 Resume] Ignore external dataloader overwrite: "
                    f"current_rank={current_state.get('rank')}, "
                    f"incoming_rank={incoming_state.get('rank') if isinstance(incoming_state, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        self._runtime_initialized = False

    def __len__(self):
        return self.max_iter


class SudokuSFTV2DataLoader(DataLoader):
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


def generate_sudoku_sft_v2_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device,
    data_dir=None, resume_state_dict=None, augment=True, seed=None,
    blank_loss_weight=1.0, difficulty_bucket_weights=None,
):
    """Factory function matching generate_sudoku_sft_dataloader signature.

    v3 extras:
      - blank_loss_weight: K in [1.0, ...]; default 1.0 = off (no weight tensor emitted).
      - difficulty_bucket_weights: list of weights aligned with DIFFICULTY_BUCKETS, used
        only on the train split; None = uniform legacy behavior.
    """
    dataset = SudokuSFTV2IterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        data_dir=data_dir,
        resume_state_dict=resume_state_dict,
        augment=augment,
        seed=seed,
        blank_loss_weight=blank_loss_weight,
        difficulty_bucket_weights=difficulty_bucket_weights,
    )
    dataloader = SudokuSFTV2DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
