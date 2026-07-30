#!/usr/bin/env python3
"""Evaluation script for SFT-finetuned EBT on the Sudoku v2 dataset.

For each split (val/test by default; train opt-in) it runs puzzles through
the chat engine and reports cell accuracy, fully-solved rate, parse rate,
timing, and tokens per sample. Results are written to a structured output
tree (per-sample JSONL, summary.json/csv, timing.json, status.json) by
default. Flat-log mode (--log_file) is preserved for backward compat.

Examples:
    # 1) Single GPU, defaults (val+test, --num_samples 2 each).
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --single_gpu

    # 2) Full val+test, multi-GPU auto.
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --full_val --full_test

    # 3) Resume an interrupted run.
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --num_samples 32 --resume

    # 4) Legacy flat log (byte-compatible with the previous script).
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --log_file /tmp/legacy.log

    # 5) Direct torchrun launch.
    torchrun --standalone --nproc_per_node=8 \
        -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --full_val --full_test
"""
from __future__ import annotations

import argparse
import os
import socket
from typing import List, Optional, Tuple

# Only clear distributed env vars if we are NOT a worker spawned by
# torchrun / mp.spawn (those set WORLD_SIZE).
if not int(os.environ.get('WORLD_SIZE', '0')):
    for _var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        if _var in os.environ:
            del os.environ[_var]

import csv
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.multiprocessing as mp

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from openebm.elm.scripts.chat_ebt import EBTChatEngine
from openebm.elm.data.sudoku_dataset_v2 import PROMPT_TEMPLATES as SUDOKU_PROMPT_TEMPLATES

DEFAULT_DATA_DIR = Path('/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2')
DEFAULT_TOKENIZER = '/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer'
DEFAULT_RUN_DIR = Path('/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/d26-ctx2048-sudoku-mixed-0.6-20260508')

LEGACY_ZH_PROMPT = (
    "请解决以下数独谜题。"
    "在每个9×9的网格中，请用1-9填充空格（用0表示），"
    "确保每行、每列以及每个3×3小方格中的数字都不重复。\n\n"
    "数独谜题：\n{puzzle}\n\n"
    "请直接给出完整的解答（每行用空格分隔数字）："
)


def _pick_prompt(prompt_style: str, puzzle_text: str, seed: int, sample_idx: int) -> Tuple[str, int, str]:
    """Return (prompt_text, template_idx, style). template_idx is -1 for legacy_zh."""
    if prompt_style == 'legacy_zh':
        return LEGACY_ZH_PROMPT.format(puzzle=puzzle_text), -1, 'legacy_zh'
    if prompt_style == 'random':
        rng = random.Random(seed + sample_idx * 10007)
        idx = rng.randrange(len(SUDOKU_PROMPT_TEMPLATES))
        return SUDOKU_PROMPT_TEMPLATES[idx].format(puzzle=puzzle_text), idx, 'random'
    return SUDOKU_PROMPT_TEMPLATES[0].format(puzzle=puzzle_text), 0, 'fixed'


def _is_valid_sudoku_grid(pred: Optional[List[int]]) -> bool:
    """True iff pred is a length-81 list of 1-9 with unique row/col/3x3 box digits."""
    if pred is None or len(pred) != 81:
        return False
    if not all(isinstance(v, int) and 1 <= v <= 9 for v in pred):
        return False
    for r in range(9):
        row = pred[r * 9:(r + 1) * 9]
        if len(set(row)) != 9:
            return False
    for c in range(9):
        col = [pred[r * 9 + c] for r in range(9)]
        if len(set(col)) != 9:
            return False
    for br in range(3):
        for bc in range(3):
            box = [pred[(br * 3 + dr) * 9 + (bc * 3 + dc)]
                   for dr in range(3) for dc in range(3)]
            if len(set(box)) != 9:
                return False
    return True


class _Tee:
    """Duplicate writes to multiple streams (e.g., stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class _PrefixStream:
    """Prefix every line written to a stream (used for non-rank-0 workers)."""

    def __init__(self, stream, prefix):
        self.stream = stream
        self.prefix = prefix
        self._at_line_start = True

    def write(self, data):
        if not data:
            return
        out = []
        for ch in data:
            if self._at_line_start and ch != '\n':
                out.append(self.prefix)
                self._at_line_start = False
            out.append(ch)
            if ch == '\n':
                self._at_line_start = True
        try:
            self.stream.write(''.join(out))
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass


def _resolve_log_path(checkpoint: str, run_dir: Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_stem = Path(checkpoint).stem if checkpoint else 'nockpt'
    log_dir = run_dir / 'eval_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f'{ckpt_stem}_{ts}.log'


def _resolve_output_layout(args) -> dict:
    """Decide flat vs structured output mode.

    Rules:
      1. User passed --log_file only  -> flat mode (byte-compat with old script).
      2. User passed --out_dir        -> structured; log_path = <out_dir>/eval.log
      3. Both passed                  -> structured + custom log_path = --log_file
      4. Neither                      -> structured under
                                         <run_dir>/eval_logs/sudoku_samples/<ckpt_stem>_<ts>/
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_stem = Path(args.checkpoint).stem if args.checkpoint else 'nockpt'

    out_dir = getattr(args, 'out_dir', None)
    log_file = getattr(args, 'log_file', None)

    if out_dir is None and log_file is not None:
        return {
            'mode': 'flat',
            'ts': ts,
            'ckpt_stem': ckpt_stem,
            'out_dir': None,
            'log_path': Path(log_file),
        }

    if out_dir is not None:
        out_dir_p = Path(out_dir)
    else:
        out_dir_p = Path(args.run_dir) / 'eval_logs' / 'sudoku_samples' / f'{ckpt_stem}_{ts}'

    log_path = Path(log_file) if log_file is not None else (out_dir_p / 'eval.log')
    return {
        'mode': 'structured',
        'ts': ts,
        'ckpt_stem': ckpt_stem,
        'out_dir': out_dir_p,
        'log_path': log_path,
    }


def _ckpt_sha256_short(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    """Fingerprint the first max_bytes of the ckpt (cheap stable id)."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    except Exception:
        return 'unknown'
    return h.hexdigest()[:16]


def _init_output_tree(layout: dict, splits):
    """rank-0 only. Create config/ and results/per_split/<split>/."""
    if layout['mode'] != 'structured':
        return
    out_dir = layout['out_dir']
    (out_dir / 'config').mkdir(parents=True, exist_ok=True)
    per_split = out_dir / 'results' / 'per_split'
    per_split.mkdir(parents=True, exist_ok=True)
    for s in splits:
        (per_split / s).mkdir(parents=True, exist_ok=True)


def _write_config(args, layout: dict):
    """rank-0 only. Write eval_params.json + eval_ckpt_ref.json."""
    if layout['mode'] != 'structured':
        return
    out_dir = layout['out_dir']
    params = {}
    for k, v in vars(args).items():
        if k.startswith('_'):
            continue
        if isinstance(v, Path):
            params[k] = str(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            params[k] = v
        elif isinstance(v, (list, tuple)):
            params[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            params[k] = str(v)
    params['layout_mode'] = layout['mode']
    params['out_dir'] = str(out_dir)
    params['log_path'] = str(layout['log_path'])
    params['ts'] = layout['ts']
    with open(out_dir / 'config' / 'eval_params.json', 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

    ckpt_ref = {'checkpoint': str(args.checkpoint), 'stem': layout['ckpt_stem']}
    try:
        ckpt_path = Path(args.checkpoint)
        st = ckpt_path.stat()
        ckpt_ref['size_bytes'] = st.st_size
        ckpt_ref['mtime'] = datetime.fromtimestamp(st.st_mtime).isoformat()
        ckpt_ref['sha256_short'] = _ckpt_sha256_short(ckpt_path)
    except Exception as e:
        ckpt_ref['error'] = str(e)
    with open(out_dir / 'config' / 'eval_ckpt_ref.json', 'w') as f:
        json.dump(ckpt_ref, f, indent=2, sort_keys=True)


def _load_done_indices(shard_path: Path) -> set:
    """Parse an existing rank shard and collect already-evaluated indices."""
    done = set()
    if not shard_path.is_file():
        return done
    try:
        with open(shard_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = obj.get('idx')
                    if isinstance(idx, int):
                        done.add(idx)
                except Exception:
                    continue
    except Exception:
        return set()
    return done


def _merge_shards(out_dir: Path, split: str, world_size: int, keep_shards: bool) -> list:
    """Concat per-rank shards, sort by idx, write <split>.jsonl, return rows."""
    shard_dir = out_dir / 'results' / 'per_split' / split
    rows = []
    seen = set()
    for r in range(world_size):
        shard = shard_dir / f'rank{r}.jsonl'
        if not shard.is_file():
            continue
        with open(shard, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                idx = obj.get('idx')
                if idx in seen:
                    continue  # dedupe on resume
                seen.add(idx)
                rows.append(obj)
    rows.sort(key=lambda o: (o.get('idx') if isinstance(o.get('idx'), int) else 0))
    merged = out_dir / 'results' / f'{split}.jsonl'
    with open(merged, 'w') as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    if not keep_shards:
        for r in range(world_size):
            shard = shard_dir / f'rank{r}.jsonl'
            if shard.is_file():
                try:
                    shard.unlink()
                except Exception:
                    pass
        try:
            shard_dir.rmdir()
        except Exception:
            pass
    return rows


def _finalize_outputs(layout: dict, splits, world_size: int, timing: dict, keep_shards: bool,
                      max_error_dumps: int = 0):
    """rank-0 only. Merge shards, write summary/timing/status.

    v3 additions:
      - by_givens bucket rollup written under `summary[split]['by_givens']`
        and as a CSV section.
      - tail-print of the summary table to stdout (i.e. eval.log via _Tee) so
        the log is self-contained.
      - optional dump of the first `max_error_dumps` `fully_solved=False` rows
        per split under `results/errors/<split>/idx_<i>.json`.
    """
    if layout['mode'] != 'structured':
        return
    out_dir = layout['out_dir']

    summary = {}
    overall_correct = 0
    overall_total = 0
    overall_parsed = 0
    overall_solved = 0
    overall_n = 0
    overall_given_total = 0
    overall_given_correct = 0
    overall_filled_total = 0
    overall_filled_correct = 0
    overall_valid_sudoku = 0

    # v3: collect error dumps + per-split rows for by_givens rollup.
    rows_by_split = {}

    for split in splits:
        rows = _merge_shards(out_dir, split, world_size, keep_shards)
        rows_by_split[split] = rows
        n = len(rows)
        c = sum(int(o.get('correct', 0)) for o in rows)
        t = sum(int(o.get('total', 0)) for o in rows)
        parsed = sum(1 for o in rows if o.get('parsed'))
        solved = sum(1 for o in rows if o.get('fully_solved'))
        elapsed = sum(float(o.get('elapsed_s', 0.0)) for o in rows)
        tokens = sum(int(o.get('tokens_generated', 0)) for o in rows)
        given_total = sum(int(o.get('given_total', 0)) for o in rows)
        given_correct = sum(int(o.get('given_correct', 0)) for o in rows)
        filled_total = sum(int(o.get('filled_total', 0)) for o in rows)
        filled_correct = sum(int(o.get('filled_correct', 0)) for o in rows)
        valid_sudoku = sum(1 for o in rows if o.get('is_valid_sudoku'))
        summary[split] = {
            'n': n,
            'cell_correct': c,
            'cell_total': t,
            'cell_acc': (c / t) if t else 0.0,
            'parsed': parsed,
            'parsed_pct': (parsed / n) if n else 0.0,
            'fully_solved': solved,
            'fully_solved_pct': (solved / n) if n else 0.0,
            'given_total': given_total,
            'given_correct': given_correct,
            'given_cell_acc': (given_correct / given_total) if given_total else 0.0,
            'filled_total': filled_total,
            'filled_correct': filled_correct,
            'filled_cell_acc': (filled_correct / filled_total) if filled_total else 0.0,
            'valid_sudoku': valid_sudoku,
            'valid_sudoku_pct': (valid_sudoku / n) if n else 0.0,
            'elapsed_s': elapsed,
            'tokens_generated': tokens,
            'avg_tokens_per_sample': (tokens / n) if n else 0.0,
            'avg_s_per_sample': (elapsed / n) if n else 0.0,
        }

        # v3: by-givens bucket rollup. We bucket on the original puzzle's given count
        # (`given_total` recorded per sample). Reports cell_acc + solved_pct + n per
        # bucket so we can see the difficulty gradient (P5 in the v3 plan).
        by_givens = {}
        for o in rows:
            g = int(o.get('given_total', -1))
            if g <= 0:
                continue
            slot = by_givens.setdefault(g, {
                'n': 0, 'cell_correct': 0, 'cell_total': 0,
                'parsed': 0, 'fully_solved': 0,
                'given_correct': 0, 'given_total': 0,
                'filled_correct': 0, 'filled_total': 0,
            })
            slot['n'] += 1
            slot['cell_correct'] += int(o.get('correct', 0))
            slot['cell_total'] += int(o.get('total', 0))
            slot['parsed'] += int(bool(o.get('parsed')))
            slot['fully_solved'] += int(bool(o.get('fully_solved')))
            slot['given_correct'] += int(o.get('given_correct', 0))
            slot['given_total'] += int(o.get('given_total', 0))
            slot['filled_correct'] += int(o.get('filled_correct', 0))
            slot['filled_total'] += int(o.get('filled_total', 0))
        # Densify with rates.
        for g, slot in by_givens.items():
            n_g = slot['n']
            t_g = slot['cell_total']
            slot['cell_acc'] = (slot['cell_correct'] / t_g) if t_g else 0.0
            slot['fully_solved_pct'] = (slot['fully_solved'] / n_g) if n_g else 0.0
            slot['parsed_pct'] = (slot['parsed'] / n_g) if n_g else 0.0
            slot['given_cell_acc'] = (
                slot['given_correct'] / slot['given_total']
            ) if slot['given_total'] else 0.0
            slot['filled_cell_acc'] = (
                slot['filled_correct'] / slot['filled_total']
            ) if slot['filled_total'] else 0.0
        summary[split]['by_givens'] = {str(k): v for k, v in sorted(by_givens.items())}

        # v3: by-format rollup so we can diagnose grid-vs-flat regression after the fact.
        by_format = {}
        for o in rows:
            pf = o.get('puzzle_format') or 'unknown'
            slot = by_format.setdefault(pf, {'n': 0, 'cell_correct': 0,
                                              'cell_total': 0, 'fully_solved': 0})
            slot['n'] += 1
            slot['cell_correct'] += int(o.get('correct', 0))
            slot['cell_total'] += int(o.get('total', 0))
            slot['fully_solved'] += int(bool(o.get('fully_solved')))
        for pf, slot in by_format.items():
            slot['cell_acc'] = (slot['cell_correct'] / slot['cell_total']) \
                if slot['cell_total'] else 0.0
            slot['fully_solved_pct'] = (slot['fully_solved'] / slot['n']) \
                if slot['n'] else 0.0
        if by_format:
            summary[split]['by_format'] = by_format

        # v3: error dumps (first N unsolved rows per split) for case-study debugging.
        if max_error_dumps > 0:
            err_dir = out_dir / 'results' / 'errors' / split
            err_dir.mkdir(parents=True, exist_ok=True)
            dumped = 0
            for o in rows:
                if dumped >= max_error_dumps:
                    break
                if o.get('fully_solved'):
                    continue
                idx = o.get('idx', dumped)
                try:
                    with open(err_dir / f'idx_{idx}.json', 'w') as f:
                        json.dump(o, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                dumped += 1
        overall_correct += c
        overall_total += t
        overall_parsed += parsed
        overall_solved += solved
        overall_n += n
        overall_given_total += given_total
        overall_given_correct += given_correct
        overall_filled_total += filled_total
        overall_filled_correct += filled_correct
        overall_valid_sudoku += valid_sudoku

    summary['overall'] = {
        'n': overall_n,
        'cell_correct': overall_correct,
        'cell_total': overall_total,
        'cell_acc': (overall_correct / overall_total) if overall_total else 0.0,
        'parsed': overall_parsed,
        'parsed_pct': (overall_parsed / overall_n) if overall_n else 0.0,
        'fully_solved': overall_solved,
        'fully_solved_pct': (overall_solved / overall_n) if overall_n else 0.0,
        'given_total': overall_given_total,
        'given_correct': overall_given_correct,
        'given_cell_acc': (overall_given_correct / overall_given_total) if overall_given_total else 0.0,
        'filled_total': overall_filled_total,
        'filled_correct': overall_filled_correct,
        'filled_cell_acc': (overall_filled_correct / overall_filled_total) if overall_filled_total else 0.0,
        'valid_sudoku': overall_valid_sudoku,
        'valid_sudoku_pct': (overall_valid_sudoku / overall_n) if overall_n else 0.0,
    }

    with open(out_dir / 'results' / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = out_dir / 'results' / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'n', 'cell_correct', 'cell_total', 'cell_acc',
                    'parsed', 'parsed_pct', 'fully_solved', 'fully_solved_pct',
                    'given_cell_acc', 'filled_cell_acc', 'valid_sudoku_pct'])
        for split in list(splits) + ['overall']:
            s = summary.get(split, {})
            w.writerow([
                split,
                s.get('n', 0),
                s.get('cell_correct', 0),
                s.get('cell_total', 0),
                f"{s.get('cell_acc', 0.0):.6f}",
                s.get('parsed', 0),
                f"{s.get('parsed_pct', 0.0):.6f}",
                s.get('fully_solved', 0),
                f"{s.get('fully_solved_pct', 0.0):.6f}",
                f"{s.get('given_cell_acc', 0.0):.6f}",
                f"{s.get('filled_cell_acc', 0.0):.6f}",
                f"{s.get('valid_sudoku_pct', 0.0):.6f}",
            ])

    with open(out_dir / 'results' / 'timing.json', 'w') as f:
        json.dump(timing, f, indent=2, sort_keys=True)

    # v3: by-givens CSV section per split — writes a separate CSV so the v2 summary.csv
    # stays byte-clean for any existing parsers.
    bg_csv_path = out_dir / 'results' / 'by_givens.csv'
    try:
        with open(bg_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['split', 'givens', 'n', 'cell_acc', 'fully_solved_pct',
                        'parsed_pct', 'given_cell_acc', 'filled_cell_acc'])
            for split in splits:
                bg = summary.get(split, {}).get('by_givens', {})
                for g in sorted(bg.keys(), key=lambda x: int(x)):
                    s = bg[g]
                    w.writerow([
                        split, g, s.get('n', 0),
                        f"{s.get('cell_acc', 0.0):.6f}",
                        f"{s.get('fully_solved_pct', 0.0):.6f}",
                        f"{s.get('parsed_pct', 0.0):.6f}",
                        f"{s.get('given_cell_acc', 0.0):.6f}",
                        f"{s.get('filled_cell_acc', 0.0):.6f}",
                    ])
    except Exception as e:
        print(f"[eval_sudoku_samples] WARN: failed to write by_givens.csv: {e}")

    # v3: tail-print summary tables to stdout (which is _Tee'd to eval.log) so the log
    # is self-contained for offline review.
    try:
        print()
        print("================ FINAL SUMMARY (v3) ================")
        for split in list(splits) + ['overall']:
            s = summary.get(split, {})
            print(f"  {split:>8}: n={s.get('n', 0)}  "
                  f"cell_acc={100 * s.get('cell_acc', 0.0):.2f}%  "
                  f"solved={100 * s.get('fully_solved_pct', 0.0):.2f}%  "
                  f"parsed={100 * s.get('parsed_pct', 0.0):.2f}%  "
                  f"given={100 * s.get('given_cell_acc', 0.0):.2f}%  "
                  f"filled={100 * s.get('filled_cell_acc', 0.0):.2f}%")
        print()
        print("================ BY GIVENS ================")
        print(f"  {'split':>8} {'g':>4} {'n':>5} {'cell_acc':>10} {'solved%':>9}")
        for split in splits:
            bg = summary.get(split, {}).get('by_givens', {})
            for g in sorted(bg.keys(), key=lambda x: int(x)):
                s = bg[g]
                print(f"  {split:>8} {g:>4} {s.get('n', 0):>5} "
                      f"{100 * s.get('cell_acc', 0.0):>9.2f}% "
                      f"{100 * s.get('fully_solved_pct', 0.0):>8.2f}%")
        print("=========================================")
    except Exception as e:
        print(f"[eval_sudoku_samples] WARN: tail-print summary failed: {e}")


def _write_status(layout: dict, exit_code: int, error: str | None,
                  started_at: str, finished_at: str):
    if layout['mode'] != 'structured':
        return
    out_dir = layout['out_dir']
    if not out_dir.is_dir():
        return
    status = {
        'exit_code': exit_code,
        'error': error,
        'started_at': started_at,
        'finished_at': finished_at,
    }
    try:
        with open(out_dir / 'status.json', 'w') as f:
            json.dump(status, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _format_board(values):
    return "\n".join(" ".join(str(v) for v in values[r * 9:(r + 1) * 9]) for r in range(9))


def parse_board(text):
    """v3: lenient parser that handles both `grid` (space-separated) and `flat`
    (spaceless 81-digit) outputs, and tolerates preambles like
    "The solution is:\\n...".

    Strategy:
      1. Whitespace-split, keep single-digit ints. If we got ≥81 take the last 81.
      2. Otherwise scan all single-digit chars in the string and take the last 81
         (the "last" choice handles preambles before the actual solution).
    Returns a list of 81 ints, or None on failure.
    """
    if not text:
        return None
    nums = [int(x) for x in text.split() if x.isdigit() and len(x) == 1]
    if len(nums) >= 81:
        return nums[-81:]
    nums = [int(c) for c in text if c.isdigit()]
    if len(nums) >= 81:
        return nums[-81:]
    return None


def evaluate_sample(engine, sample, max_tokens: int = 256,
                    prompt_style: str = 'fixed', seed: int = 0, sample_idx: int = 0):
    flat_puzzle = sample['puzzle']
    if hasattr(flat_puzzle, 'tolist'):
        flat_puzzle = flat_puzzle.tolist()
    flat_solution = sample['solution']
    if hasattr(flat_solution, 'tolist'):
        flat_solution = flat_solution.tolist()
    if hasattr(flat_puzzle[0], '__iter__'):
        flat_puzzle = [v for row in flat_puzzle for v in row]
    if hasattr(flat_solution[0], '__iter__'):
        flat_solution = [v for row in flat_solution for v in row]
    puzzle_text = _format_board(flat_puzzle)
    solution_text = _format_board(flat_solution)
    prompt, tpl_idx, style = _pick_prompt(prompt_style, puzzle_text, seed, sample_idx)

    t0 = time.time()
    response, stats = engine.generate(
        prompt=prompt, max_tokens=max_tokens, temperature=0.0, top_p=0.9, stream=False,
    )
    elapsed_s = time.time() - t0
    tokens_generated = int(stats.get('tokens_generated', 0)) if isinstance(stats, dict) else 0

    pred = parse_board(response)

    # Cell accuracy — split into given vs filled.
    correct, total = 0, 81
    given_mask = [v != 0 for v in flat_puzzle]
    given_total = sum(given_mask)
    filled_total = 81 - given_total
    given_correct = 0
    filled_correct = 0
    if pred is not None:
        for i in range(81):
            if pred[i] == flat_solution[i]:
                correct += 1
                if given_mask[i]:
                    given_correct += 1
                else:
                    filled_correct += 1
    fully_solved = pred is not None and correct == total
    valid = pred is not None and _is_valid_sudoku_grid(pred)

    return {
        'puzzle_text': puzzle_text,
        'solution_text': solution_text,
        'response': response,
        'pred': pred,
        'correct': correct,
        'total': total,
        'given_total': given_total,
        'given_correct': given_correct,
        'filled_total': filled_total,
        'filled_correct': filled_correct,
        'parsed': pred is not None,
        'fully_solved': fully_solved,
        'is_valid_sudoku': valid,
        'prompt_style': style,
        'prompt_template_idx': tpl_idx,
        'prompt_used': prompt,
        # v3: format axis. Eval renders the prompt-side puzzle in `grid` format only,
        # but we still tag the field so `by_format` rollup works once eval is extended
        # to flat. The model-side response_format is best-effort: "flat" if the parsed
        # response had no whitespace among the digit run, else "grid".
        'puzzle_format': 'grid',
        'solution_format': _infer_response_format(response, pred),
        'elapsed_s': elapsed_s,
        'tokens_generated': tokens_generated,
    }


def _infer_response_format(response: Optional[str], pred: Optional[List[int]]) -> str:
    if response is None or pred is None:
        return 'unknown'
    # If the response contains a contiguous 81-digit run with no internal whitespace,
    # call it `flat`. Otherwise `grid`.
    digit_runs = []
    cur = []
    for ch in response:
        if ch.isdigit():
            cur.append(ch)
        else:
            if cur:
                digit_runs.append(len(cur))
                cur = []
    if cur:
        digit_runs.append(len(cur))
    if any(r >= 81 for r in digit_runs):
        return 'flat'
    return 'grid'


SPLIT_FILES = [
    ('train', 'rrn_train.pt'),
    ('val',   'rrn_val.pt'),
    ('test',  'satnet_test.pt'),
]


def _install_sigint_safe_finalize(args, layout, world_size):
    """v3: rank-0 SIGINT/SIGTERM handler. Calls _finalize_outputs against whatever
    shard data already exists, so an interrupted long run still emits summary.json
    + by_givens.csv from the partial JSONL.

    Re-raises the default behavior after finalize so the process still exits.
    """
    import signal

    if layout is None or layout.get('mode') != 'structured':
        return

    finalized = {'done': False}

    def _handler(signum, frame):
        if finalized['done']:
            return
        finalized['done'] = True
        try:
            usable = list(getattr(args, 'splits', None) or [])
            present = [s for s in usable
                       if (layout['out_dir'] / 'results' / 'per_split' / s).exists()
                       or (layout['out_dir'] / 'results' / f'{s}.jsonl').exists()]
            print(f"\n[eval_sudoku_samples] caught signal {signum}; "
                  f"finalizing partial outputs for splits={present}")
            _finalize_outputs(layout, present, world_size, {}, True,
                              max_error_dumps=getattr(args, 'max_error_dumps', 0))
            _write_status(layout, 130, f'signal {signum}',
                          datetime.now().isoformat(), datetime.now().isoformat())
        except Exception as e:
            print(f"[eval_sudoku_samples] sigint finalize error: {e}")
        # Re-raise default behavior on SIGINT so the user gets normal Ctrl-C semantics.
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            sys.exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def load_splits(data_dir):
    splits = {}
    for name, fname in SPLIT_FILES:
        path = Path(data_dir) / fname
        if path.is_file():
            splits[name] = torch.load(path, weights_only=False)
    return splits


def _resolve_split_ranges(args, splits, wanted=None):
    raw = {
        'train': args.num_samples_train if args.num_samples_train is not None else args.num_samples,
        'val':   args.num_samples_val   if args.num_samples_val   is not None else args.num_samples,
        'test':  args.num_samples_test  if args.num_samples_test  is not None else args.num_samples,
    }
    starts = {
        'train': args.start_index_train if args.start_index_train is not None else args.start_index,
        'val':   args.start_index_val   if args.start_index_val   is not None else args.start_index,
        'test':  args.start_index_test  if args.start_index_test  is not None else args.start_index,
    }
    full_flags = {
        'train': args.full or args.full_train,
        'val':   args.full or args.full_val,
        'test':  args.full or args.full_test,
    }
    out = {}
    wanted_set = set(wanted) if wanted is not None else None
    for name, samples in splits.items():
        if wanted_set is not None and name not in wanted_set:
            continue
        total = len(samples)
        start = int(starts.get(name) or 0)
        if start < 0:
            raise ValueError(f'--start_index_{name} must be >= 0')
        start = min(start, total)
        n = raw.get(name)
        if full_flags.get(name) or n is None or n < 0:
            stop = total
        else:
            stop = min(total, start + max(0, int(n)))
        out[name] = {'start': start, 'stop': stop, 'n': max(0, stop - start), 'total': total}
    return out


def _resolve_split_sizes(args, splits, wanted=None):
    ranges = _resolve_split_ranges(args, splits, wanted=wanted)
    return {name: spec['n'] for name, spec in ranges.items()}


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _decide_world_size(args, ngpu: int) -> int:
    if args.single_gpu:
        return 1
    if args.gpus and args.gpus > 0:
        return min(args.gpus, max(ngpu, 1))
    # auto
    return max(ngpu, 1)


def _setup_logging_if_needed(args, is_main: bool, rank: int):
    if is_main:
        layout = getattr(args, '_layout', None)
        if layout is None:
            # Fallback path (shouldn't normally happen — main() precomputes it).
            log_path = args.log_file or _resolve_log_path(args.checkpoint, args.run_dir)
        else:
            log_path = layout['log_path']
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, 'a', buffering=1)
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"[eval_sudoku_samples] checkpoint={args.checkpoint}")
        print(f"[eval_sudoku_samples] log_file={log_path}")
        if layout is not None and layout['mode'] == 'structured':
            print(f"[eval_sudoku_samples] out_dir={layout['out_dir']}")
    else:
        prefix = f"[rank={rank}] "
        sys.stdout = _PrefixStream(sys.__stdout__, prefix)
        sys.stderr = _PrefixStream(sys.__stderr__, prefix)


def _run_eval(args, rank: int, world_size: int):
    import torch.distributed as dist

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # TF32 / matmul precision (applied in-worker; idempotent).
    if getattr(args, 'tf32', True):
        try:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    splits = load_splits(args.data_dir)
    if not splits:
        if rank == 0:
            print(f"No splits found under {args.data_dir}")
        return {}, {}

    # Filter to user-requested splits, warning on unknowns (rank 0 only).
    wanted = list(getattr(args, 'splits', None) or ['val', 'test'])
    usable = [s for s in wanted if s in splits]
    missing = [s for s in wanted if s not in splits]
    if rank == 0 and missing:
        print(f"[eval_sudoku_samples] WARNING: requested splits not found in cache: {missing}")
    if not usable:
        if rank == 0:
            print(f"[eval_sudoku_samples] No usable splits from {wanted}; cache has {list(splits.keys())}")
        return {}, {}

    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    engine = EBTChatEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=dtype,
        show_mcmc=False,
        verbose=(rank == 0),
    )

    split_ranges = _resolve_split_ranges(args, splits, wanted=usable)
    split_n = {s: spec['n'] for s, spec in split_ranges.items()}
    if rank == 0:
        print(f"[eval_sudoku_samples] world_size={world_size} dtype={args.dtype} "
              f"max_tokens={args.max_tokens} splits={usable} sizes={split_n}")
        print(
            "[eval_sudoku_samples] index_ranges="
            + str({s: [spec['start'], spec['stop']] for s, spec in split_ranges.items()})
            + " tqdm shows rank-0 local shard, not the global sample count"
        )

    # Flat work pool across splits for balanced rank workloads.
    pool = [
        (s, i)
        for s in usable
        for i in range(split_ranges[s]['start'], split_ranges[s]['stop'])
    ]
    my_pool = pool[rank::world_size]

    # Per-rank shard (structured mode) — streaming JSONL.
    layout = getattr(args, '_layout', None)
    structured = layout is not None and layout['mode'] == 'structured'
    shard_fps = {}
    done = {s: set() for s in usable}
    if structured:
        for s in usable:
            shard_dir = layout['out_dir'] / 'results' / 'per_split' / s
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f'rank{rank}.jsonl'
            if getattr(args, 'resume', False):
                done[s] = _load_done_indices(shard_path)
            shard_fps[s] = open(shard_path, 'a', buffering=1)

    reduce_device = args.device if str(args.device).startswith('cuda') else 'cpu'

    # Per-split local accumulators: [correct, total, parsed, fully_solved, n]
    local_stats = {s: [0, 0, 0, 0, 0] for s in usable}
    split_elapsed_local = {s: 0.0 for s in usable}

    # Progress bar: rank 0 only; other ranks stay quiet.
    iterable = my_pool
    pbar = None
    if rank == 0 and tqdm is not None:
        pbar = tqdm(total=len(my_pool), desc='eval', dynamic_ncols=True)

    suppress_stdout = getattr(args, 'no_per_sample_print', False)

    t_split_start = {s: time.time() for s in usable}
    t_split_end = {s: None for s in usable}
    seen_counts = {s: 0 for s in usable}
    total_expected = {
        s: len(range(split_ranges[s]['start'], split_ranges[s]['stop'])[rank::world_size])
        for s in usable
    }

    t_all_start = time.time()
    rolling_c, rolling_t = 0, 0

    for (split_name, i) in iterable:
        if getattr(args, 'resume', False) and i in done[split_name]:
            seen_counts[split_name] += 1
            if pbar is not None:
                pbar.update(1)
            continue

        sample = splits[split_name][i]
        try:
            res = evaluate_sample(
                engine, sample,
                max_tokens=args.max_tokens,
                prompt_style=args.prompt_style,
                seed=args.seed,
                sample_idx=i,
            )
        except Exception as e:
            # Record a failed sample but keep going.
            res = {
                'puzzle_text': '', 'solution_text': '', 'response': f'<error: {e}>',
                'pred': None, 'correct': 0, 'total': 81,
                'given_total': 0, 'given_correct': 0,
                'filled_total': 0, 'filled_correct': 0,
                'parsed': False,
                'fully_solved': False, 'is_valid_sudoku': False,
                'prompt_style': args.prompt_style, 'prompt_template_idx': -1,
                'prompt_used': '',
                'puzzle_format': 'grid', 'solution_format': 'unknown',
                'elapsed_s': 0.0, 'tokens_generated': 0,
            }
            if rank == 0:
                print(f"[eval_sudoku_samples] sample error split={split_name} idx={i}: {e}")

        # Stream to per-rank shard.
        if structured:
            obj = {
                'split': split_name, 'idx': i, 'rank': rank,
                'correct': res['correct'], 'total': res['total'],
                'given_total': res['given_total'],
                'given_correct': res['given_correct'],
                'filled_total': res['filled_total'],
                'filled_correct': res['filled_correct'],
                'parsed': res['parsed'], 'fully_solved': res['fully_solved'],
                'is_valid_sudoku': res['is_valid_sudoku'],
                'prompt_style': res['prompt_style'],
                'prompt_template_idx': res['prompt_template_idx'],
                'prompt_used': res['prompt_used'],
                # v3: format axis (puzzle_format = how the prompt rendered the puzzle;
                # solution_format = best-effort inference from the model's response).
                'puzzle_format': res.get('puzzle_format', 'grid'),
                'solution_format': res.get('solution_format', 'unknown'),
                'elapsed_s': round(res['elapsed_s'], 4),
                'tokens_generated': res['tokens_generated'],
                'response': res['response'], 'pred': res['pred'],
            }
            try:
                shard_fps[split_name].write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                if rank == 0:
                    print(f"[eval_sudoku_samples] shard write failed split={split_name} idx={i}: {e}")

        # Per-sample stdout (optional).
        if not suppress_stdout:
            tag = f"--- {split_name} sample {i} (rank={rank}) ---"
            print(f"\n{tag}")
            print('[PUZZLE]')
            print(res['puzzle_text'])
            print('[GROUND TRUTH]')
            print(res['solution_text'])
            print('[MODEL OUTPUT]')
            print(res['response'])
            if structured:
                print(f"[CELL ACCURACY] {res['correct']}/{res['total']} "
                      f"parsed={res['parsed']} solved={res['fully_solved']} "
                      f"t={res['elapsed_s']:.2f}s tokens={res['tokens_generated']}")
            else:
                # Flat mode: preserve the legacy one-line format byte-for-byte.
                print(f"[CELL ACCURACY] {res['correct']}/{res['total']} parsed={res['parsed']}")

        st = local_stats[split_name]
        st[0] += res['correct']
        st[1] += res['total']
        st[2] += int(res['parsed'])
        st[3] += int(res['fully_solved'])
        st[4] += 1
        split_elapsed_local[split_name] += res['elapsed_s']
        seen_counts[split_name] += 1
        if seen_counts[split_name] >= total_expected[split_name]:
            t_split_end[split_name] = time.time()

        rolling_c += res['correct']
        rolling_t += res['total']
        if pbar is not None:
            pbar.update(1)
            if pbar.n % 8 == 0 or pbar.n == pbar.total:
                acc = (rolling_c / rolling_t * 100) if rolling_t else 0.0
                pbar.set_postfix(split=split_name, cell_acc=f"{acc:.2f}%")

    if pbar is not None:
        pbar.close()

    # Close per-rank shards before all-reduce so merged output on rank 0 is sane.
    for fp in shard_fps.values():
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass

    # Fused all_reduce across all splits: [len(usable), 5] float64.
    stats_tensor = torch.tensor(
        [local_stats[s] for s in usable],
        device=reduce_device, dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    # Per-split aggregate print (rank 0) — kept for eval.log readability.
    per_split_agg = {}
    if rank == 0:
        print()
        overall_c, overall_t, overall_solved, overall_parsed, overall_n = 0, 0, 0, 0, 0
        for row, s in zip(stats_tensor.tolist(), usable):
            c, t, p, solved, n = (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            acc = (c / t * 100) if t else 0.0
            solved_pct = (solved / n * 100) if n else 0.0
            parsed_pct = (p / n * 100) if n else 0.0
            print(f">>> {s}: cell_acc={c}/{t}={acc:.2f}% | "
                  f"solved={solved}/{n}={solved_pct:.2f}% | "
                  f"parsed={p}/{n}={parsed_pct:.2f}%")
            per_split_agg[s] = {'c': c, 't': t, 'parsed': p, 'solved': solved, 'n': n}
            overall_c += c
            overall_t += t
            overall_solved += solved
            overall_parsed += p
            overall_n += n
        acc = (overall_c / overall_t * 100) if overall_t else 0.0
        solved_pct = (overall_solved / overall_n * 100) if overall_n else 0.0
        parsed_pct = (overall_parsed / overall_n * 100) if overall_n else 0.0
        print(f"\n===== SUMMARY ({usable}): cell_acc={overall_c}/{overall_t}={acc:.2f}% | "
              f"solved={overall_solved}/{overall_n}={solved_pct:.2f}% | "
              f"parsed={overall_parsed}/{overall_n}={parsed_pct:.2f}% =====")

    # Timing (rank-0's view: per-split elapsed via local max + overall wall clock).
    timing = {}
    if rank == 0:
        now = time.time()
        for s in usable:
            end = t_split_end[s] if t_split_end[s] is not None else now
            timing[s] = {
                'wall_s_rank0': round(end - t_split_start[s], 3),
                'sum_sample_s_rank0': round(split_elapsed_local[s], 3),
            }
        timing['overall'] = {'wall_s_rank0': round(now - t_all_start, 3)}

    return per_split_agg, timing


def _broadcast_layout(args, rank: int, world_size: int):
    """Ensure every rank shares the same structured out_dir / log_path.

    Torchrun launches each rank as an independent Python process that
    calls main() -> _resolve_output_layout() separately, so their
    datetime.now() values drift. We pick rank 0's layout as the source of
    truth and broadcast its timestamp + out_dir + log_path to all ranks.
    mp.spawn / single-process paths are unaffected (args is pickled/shared).
    """
    import torch.distributed as dist
    if world_size <= 1 or not dist.is_initialized():
        return
    layout = getattr(args, '_layout', None)
    if layout is None:
        # Compute something on every rank so the broadcast is safe.
        layout = _resolve_output_layout(args)
        args._layout = layout

    payload = {
        'mode': layout['mode'],
        'ts': layout['ts'],
        'ckpt_stem': layout['ckpt_stem'],
        'out_dir': str(layout['out_dir']) if layout['out_dir'] is not None else None,
        'log_path': str(layout['log_path']),
    }
    obj_list = [payload] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    got = obj_list[0]
    args._layout = {
        'mode': got['mode'],
        'ts': got['ts'],
        'ckpt_stem': got['ckpt_stem'],
        'out_dir': Path(got['out_dir']) if got['out_dir'] is not None else None,
        'log_path': Path(got['log_path']),
    }


def _run_distributed_worker(args):
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    args.device = f'cuda:{local_rank}'
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    # Sync layout across ranks (torchrun spawns independent processes that
    # each computed their own timestamp in main()).
    _broadcast_layout(args, rank=rank, world_size=world_size)
    if rank == 0:
        layout = getattr(args, '_layout', None)
        if layout is not None and layout['mode'] == 'structured':
            _init_output_tree(layout, list(args.splits or []))
            _write_config(args, layout)
    _setup_logging_if_needed(args, is_main=(rank == 0), rank=rank)
    # v3: install SIGINT-safe finalize on rank 0 so a partial run still flushes
    # summary.json + by_givens.csv from existing per-rank shards on Ctrl-C.
    if rank == 0:
        _install_sigint_safe_finalize(args, getattr(args, '_layout', None), world_size)
    started_at = datetime.now().isoformat()
    err_txt = None
    exit_code = 0
    per_split_agg, timing = {}, {}
    try:
        per_split_agg, timing = _run_eval(args, rank=rank, world_size=world_size)
        dist.barrier()
    except Exception as e:
        exit_code = 1
        err_txt = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if rank == 0:
            print(f"[eval_sudoku_samples] ERROR: {err_txt}")
    finally:
        try:
            if rank == 0:
                layout = getattr(args, '_layout', None)
                if layout is not None and layout['mode'] == 'structured':
                    usable = list((getattr(args, 'splits', None) or ['val', 'test']))
                    # Only finalize for splits we actually created dirs for.
                    present = [s for s in usable
                               if (layout['out_dir'] / 'results' / 'per_split' / s).exists()
                               or (layout['out_dir'] / 'results' / f'{s}.jsonl').exists()]
                    _finalize_outputs(layout, present, world_size,
                                      timing, getattr(args, 'keep_shards', False),
                                      max_error_dumps=getattr(args, 'max_error_dumps', 0))
                    _write_status(layout, exit_code, err_txt,
                                  started_at, datetime.now().isoformat())
        except Exception as e:
            print(f"[eval_sudoku_samples] finalize error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()


def _mp_worker_entry(local_rank, world_size, port, args):
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    _run_distributed_worker(args)


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    parser.add_argument('--data_dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--run_dir', type=Path, default=DEFAULT_RUN_DIR,
                        help='Run directory; default structured out_dir lives under <run_dir>/eval_logs/sudoku_samples/.')
    parser.add_argument('--log_file', type=Path, default=None,
                        help='Override the log file path. Passing only --log_file triggers flat mode (legacy).')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Structured output root. None -> auto under <run_dir>/eval_logs/sudoku_samples/<ckpt_stem>_<ts>/.')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Splits to evaluate; default opts out of the 180k rrn_train.')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Default samples per split (used if --num_samples_<split> not set).')
    parser.add_argument('--num_samples_train', type=int, default=None)
    parser.add_argument('--num_samples_val', type=int, default=None)
    parser.add_argument('--num_samples_test', type=int, default=None)
    parser.add_argument('--start_index', type=int, default=0,
                        help='Default start index per split. Range semantics are [start, start + num_samples).')
    parser.add_argument('--start_index_train', type=int, default=None)
    parser.add_argument('--start_index_val', type=int, default=None)
    parser.add_argument('--start_index_test', type=int, default=None)
    parser.add_argument('--full', action='store_true', help='Evaluate all samples in every requested split.')
    parser.add_argument('--full_train', action='store_true')
    parser.add_argument('--full_val', action='store_true')
    parser.add_argument('--full_test', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=256,
                        help='Decode budget per sample (~200 tokens covers an 81-cell solution).')
    parser.add_argument('--prompt_style', default='fixed',
                        choices=['fixed', 'random', 'legacy_zh'],
                        help='Prompt style. Default "fixed" uses PROMPT_TEMPLATES[0] from '
                             'sudoku_dataset_v2.py (matches training domain). '
                             '"random" samples per-puzzle for distribution match. '
                             '"legacy_zh" preserves the old Chinese prompt for historical reruns.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip indices already present in per-rank JSONL shards.')
    parser.add_argument('--no_per_sample_print', action='store_true',
                        help='Suppress per-sample puzzle/GT/output stdout; JSONL is still written.')
    parser.add_argument('--keep_shards', action='store_true',
                        help='Keep per-rank JSONL shards after merging (default: delete).')
    tf32_grp = parser.add_mutually_exclusive_group()
    tf32_grp.add_argument('--tf32', dest='tf32', action='store_true',
                          help='Enable TF32 matmul / cudnn (default).')
    tf32_grp.add_argument('--no_tf32', dest='tf32', action='store_false',
                          help='Disable TF32 matmul / cudnn.')
    parser.set_defaults(tf32=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs to use (0=auto, 1=force single GPU).')
    parser.add_argument('--single_gpu', action='store_true', help='Disable auto multi-GPU.')
    parser.add_argument('--master_port', type=int, default=0,
                        help='Master port for mp.spawn (0=auto).')
    # v3: error dump cap. 0 = off; >0 dumps the first N unsolved samples per split
    # under results/errors/<split>/idx_<i>.json (rank-0 only, finalized post-merge).
    parser.add_argument('--max_error_dumps', type=int, default=0,
                        help='Per-split cap on unsolved-sample JSON dumps (0=disabled).')
    return parser


def main():
    args = _build_parser().parse_args()

    in_distributed = int(os.environ.get('WORLD_SIZE', '0')) > 0

    # Compute layout in the parent so all workers share one timestamp + out_dir.
    # For torchrun-launched workers, the parent IS the first worker — we still
    # compute the layout once and rely on each rank looking at it.
    if not hasattr(args, '_layout') or getattr(args, '_layout', None) is None:
        layout = _resolve_output_layout(args)
        args._layout = layout
    else:
        layout = args._layout

    # rank-0 (or single-process) side: for non-distributed and mp.spawn paths,
    # the parent creates the output tree eagerly. For torchrun-launched ranks
    # the tree is created post-broadcast inside _run_distributed_worker so the
    # out_dir timestamp is the one picked by rank 0.
    is_parent_rank0 = (
        (not in_distributed) or int(os.environ.get('RANK', '0')) == 0
    )
    if layout['mode'] == 'structured' and is_parent_rank0 and not in_distributed:
        _init_output_tree(layout, list(args.splits or []))
        _write_config(args, layout)

    if in_distributed:
        # Scenario A: launched by torchrun or our own mp.spawn worker.
        _run_distributed_worker(args)
        return

    ngpu = torch.cuda.device_count()
    want = _decide_world_size(args, ngpu)

    if want > 1 and ngpu > 1:
        # Scenario B: spawn workers ourselves.
        port = args.master_port if args.master_port > 0 else _find_free_port()
        mp.spawn(_mp_worker_entry, args=(want, port, args), nprocs=want, join=True)
        # Parent-side status: mp.spawn returns only after all workers exit.
        # Status/summary/timing were finalized by rank 0 in-worker.
        return

    # Scenario C: single GPU / CPU.
    _setup_logging_if_needed(args, is_main=True, rank=0)
    # v3: SIGINT-safe finalize for single-process path too.
    _install_sigint_safe_finalize(args, layout, 1)
    started_at = datetime.now().isoformat()
    err_txt = None
    exit_code = 0
    per_split_agg, timing = {}, {}
    try:
        per_split_agg, timing = _run_eval(args, rank=0, world_size=1)
    except Exception as e:
        exit_code = 1
        err_txt = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[eval_sudoku_samples] ERROR: {err_txt}")
    finally:
        if layout['mode'] == 'structured':
            usable = list(args.splits or [])
            present = [s for s in usable
                       if (layout['out_dir'] / 'results' / 'per_split' / s).exists()
                       or (layout['out_dir'] / 'results' / f'{s}.jsonl').exists()]
            _finalize_outputs(layout, present, 1, timing, args.keep_shards,
                              max_error_dumps=getattr(args, 'max_error_dumps', 0))
            _write_status(layout, exit_code, err_txt,
                          started_at, datetime.now().isoformat())
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
