#!/usr/bin/env python3
"""Offline parser for Sudoku eval outputs.

Two input modes:
  1) Flat-mode `eval.log` (legacy `--log_file <path>` style). Regex over the per-sample
     `[CELL ACCURACY] X/81 parsed=… solved=… t=…s tokens=…` lines plus the
     `[PUZZLE]/[GROUND TRUTH]/[MODEL OUTPUT]` blocks before each.
  2) Structured-mode `results/per_split/<split>/rank{0..N}.jsonl` shard tree (the
     post-`_finalize_outputs` summary.json may not exist if the run was interrupted —
     this is the path used for the v3 plan's "current 0.6 partial run").

Emits `eval_metrics.json` next to the input with the same schema as `summary.json`
plus a `by_givens` rollup (matches `_finalize_outputs` v3 schema).

Examples:
    # 1) Parse a flat-mode eval.log
    python -m openebm.elm.scripts.parse_eval_log --log_file /path/to/eval.log

    # 2) Parse the JSONL shard tree from a structured run
    python -m openebm.elm.scripts.parse_eval_log --shards /path/to/results/per_split

    # 3) Custom output path
    python -m openebm.elm.scripts.parse_eval_log --shards X --out /tmp/m.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ── Sudoku-grid sanity helpers ────────────────────────────────────────────────

def _is_valid_sudoku_grid(pred):
    if pred is None or len(pred) != 81:
        return False
    if not all(isinstance(v, int) and 1 <= v <= 9 for v in pred):
        return False
    for r in range(9):
        if len(set(pred[r * 9:(r + 1) * 9])) != 9:
            return False
    for c in range(9):
        if len(set(pred[r * 9 + c] for r in range(9))) != 9:
            return False
    for br in range(3):
        for bc in range(3):
            box = [pred[(br * 3 + dr) * 9 + (bc * 3 + dc)]
                   for dr in range(3) for dc in range(3)]
            if len(set(box)) != 9:
                return False
    return True


def _parse_board(text):
    """Same lenient parser as eval_sudoku_samples.parse_board (v3)."""
    if not text:
        return None
    nums = [int(x) for x in text.split() if x.isdigit() and len(x) == 1]
    if len(nums) >= 81:
        return nums[-81:]
    nums = [int(c) for c in text if c.isdigit()]
    if len(nums) >= 81:
        return nums[-81:]
    return None


def _flatten_grid_text(text):
    """Parse a 9x9 grid block (space-separated, newline-separated) into a flat list."""
    out = []
    for ln in text.strip().splitlines():
        for tok in ln.split():
            if tok.isdigit() and len(tok) == 1:
                out.append(int(tok))
    return out if len(out) == 81 else None


# ── Flat eval.log parser ──────────────────────────────────────────────────────

# Per-sample blocks look like (from eval_sudoku_samples.py, with _Tee timestamps):
#   --- val sample 0 (rank=0) ---
#   [PUZZLE]
#   <9 rows>
#   [GROUND TRUTH]
#   <9 rows>
#   [MODEL OUTPUT]
#   <free text>
#   [CELL ACCURACY] X/81 parsed=… [solved=… t=…s tokens=…]

SAMPLE_HEADER_RE = re.compile(
    r'---\s*(?P<split>\w+)\s+sample\s+(?P<idx>\d+)\s*\(rank=\d+\)\s*---'
)
CELL_ACC_RE = re.compile(
    r'\[CELL ACCURACY\]\s*(?P<correct>\d+)/(?P<total>\d+)\s*'
    r'parsed=(?P<parsed>True|False)'
    r'(?:\s+solved=(?P<solved>True|False))?'
    r'(?:\s+t=(?P<elapsed>[0-9.]+)s)?'
    r'(?:\s+tokens=(?P<tokens>\d+))?'
)


def _strip_timestamp(line: str) -> str:
    """Strip a leading `[YYYY-MM-DD HH:MM:SS]` prefix added by training shell tee."""
    return re.sub(r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*', '', line)


def parse_eval_log(path: Path):
    """Walk a flat-mode eval.log and return a list of per-sample dicts."""
    rows = []
    current = None
    section = None
    section_buf = []

    def flush_section():
        nonlocal section, section_buf
        if current is None or section is None:
            section = None
            section_buf = []
            return
        joined = '\n'.join(section_buf).strip()
        if section == 'PUZZLE':
            current['puzzle_text'] = joined
            current['puzzle_flat'] = _flatten_grid_text(joined)
        elif section == 'GROUND TRUTH':
            current['solution_text'] = joined
            current['solution_flat'] = _flatten_grid_text(joined)
        elif section == 'MODEL OUTPUT':
            current['response'] = joined
        section = None
        section_buf = []

    def commit_current():
        nonlocal current
        if current is not None:
            rows.append(current)
        current = None

    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = _strip_timestamp(raw.rstrip('\n'))
            m = SAMPLE_HEADER_RE.search(line)
            if m:
                flush_section()
                commit_current()
                current = {
                    'split': m.group('split'),
                    'idx': int(m.group('idx')),
                    'puzzle_text': '', 'solution_text': '', 'response': '',
                    'puzzle_flat': None, 'solution_flat': None,
                }
                continue
            if line.strip() == '[PUZZLE]':
                flush_section(); section = 'PUZZLE'; continue
            if line.strip() == '[GROUND TRUTH]':
                flush_section(); section = 'GROUND TRUTH'; continue
            if line.strip() == '[MODEL OUTPUT]':
                flush_section(); section = 'MODEL OUTPUT'; continue
            mca = CELL_ACC_RE.search(line)
            if mca and current is not None:
                flush_section()
                current['correct'] = int(mca.group('correct'))
                current['total'] = int(mca.group('total'))
                current['parsed'] = mca.group('parsed') == 'True'
                solved_s = mca.group('solved')
                current['fully_solved'] = (solved_s == 'True') if solved_s else None
                e_s = mca.group('elapsed')
                current['elapsed_s'] = float(e_s) if e_s else 0.0
                t_s = mca.group('tokens')
                current['tokens_generated'] = int(t_s) if t_s else 0
                continue
            if section is not None:
                section_buf.append(line)

    flush_section()
    commit_current()

    # Post-process: derive solved/parsed/given counts when the log didn't carry them.
    for o in rows:
        pred = _parse_board(o.get('response', ''))
        sol = o.get('solution_flat')
        puz = o.get('puzzle_flat')
        if pred is not None and sol is not None:
            given_mask = [v != 0 for v in puz] if puz else [False] * 81
            given_total = sum(given_mask)
            filled_total = 81 - given_total
            given_correct = 0
            filled_correct = 0
            correct = 0
            for i in range(81):
                if pred[i] == sol[i]:
                    correct += 1
                    if given_mask[i]:
                        given_correct += 1
                    else:
                        filled_correct += 1
            o.setdefault('correct', correct)
            o.setdefault('total', 81)
            o['given_total'] = given_total
            o['given_correct'] = given_correct
            o['filled_total'] = filled_total
            o['filled_correct'] = filled_correct
            if o.get('fully_solved') is None:
                o['fully_solved'] = (correct == 81)
            o.setdefault('parsed', True)
            o['is_valid_sudoku'] = _is_valid_sudoku_grid(pred)
            o['pred'] = pred
        else:
            o.setdefault('given_total', 0)
            o.setdefault('given_correct', 0)
            o.setdefault('filled_total', 0)
            o.setdefault('filled_correct', 0)
            o.setdefault('fully_solved', False)
            o.setdefault('is_valid_sudoku', False)
            o.setdefault('parsed', pred is not None)
            o['pred'] = pred

    return rows


# ── JSONL shard parser ────────────────────────────────────────────────────────

def parse_shards(shards_dir: Path):
    """Walk results/per_split/<split>/rank*.jsonl and return rows grouped by split."""
    out = {}
    for split_dir in sorted(p for p in shards_dir.iterdir() if p.is_dir()):
        split = split_dir.name
        seen_idx = set()
        rows = []
        for shard in sorted(split_dir.glob('rank*.jsonl')):
            try:
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
                        if idx in seen_idx:
                            continue
                        seen_idx.add(idx)
                        rows.append(obj)
            except Exception as e:
                print(f"[parse_eval_log] failed to read {shard}: {e}", file=sys.stderr)
        rows.sort(key=lambda o: (o.get('idx') if isinstance(o.get('idx'), int) else 0))
        out[split] = rows
    return out


# ── Summary computation (mirrors eval_sudoku_samples._finalize_outputs v3) ─────

def _summarize_rows(rows, split):
    n = len(rows)
    c = sum(int(o.get('correct', 0)) for o in rows)
    t = sum(int(o.get('total', 0)) for o in rows)
    parsed = sum(1 for o in rows if o.get('parsed'))
    solved = sum(1 for o in rows if o.get('fully_solved'))
    given_total = sum(int(o.get('given_total', 0)) for o in rows)
    given_correct = sum(int(o.get('given_correct', 0)) for o in rows)
    filled_total = sum(int(o.get('filled_total', 0)) for o in rows)
    filled_correct = sum(int(o.get('filled_correct', 0)) for o in rows)
    valid_sudoku = sum(1 for o in rows if o.get('is_valid_sudoku'))
    summary = {
        'n': n,
        'cell_correct': c, 'cell_total': t,
        'cell_acc': (c / t) if t else 0.0,
        'parsed': parsed, 'parsed_pct': (parsed / n) if n else 0.0,
        'fully_solved': solved,
        'fully_solved_pct': (solved / n) if n else 0.0,
        'given_total': given_total, 'given_correct': given_correct,
        'given_cell_acc': (given_correct / given_total) if given_total else 0.0,
        'filled_total': filled_total, 'filled_correct': filled_correct,
        'filled_cell_acc': (filled_correct / filled_total) if filled_total else 0.0,
        'valid_sudoku': valid_sudoku,
        'valid_sudoku_pct': (valid_sudoku / n) if n else 0.0,
    }
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
    for g, slot in by_givens.items():
        n_g = slot['n']; t_g = slot['cell_total']
        slot['cell_acc'] = (slot['cell_correct'] / t_g) if t_g else 0.0
        slot['fully_solved_pct'] = (slot['fully_solved'] / n_g) if n_g else 0.0
        slot['parsed_pct'] = (slot['parsed'] / n_g) if n_g else 0.0
        slot['given_cell_acc'] = (
            slot['given_correct'] / slot['given_total']
        ) if slot['given_total'] else 0.0
        slot['filled_cell_acc'] = (
            slot['filled_correct'] / slot['filled_total']
        ) if slot['filled_total'] else 0.0
    summary['by_givens'] = {str(k): v for k, v in sorted(by_givens.items())}
    return summary


def _build_full_summary(rows_by_split):
    out = {}
    overall = []
    for split, rows in rows_by_split.items():
        out[split] = _summarize_rows(rows, split)
        overall.extend(rows)
    if overall:
        out['overall'] = _summarize_rows(overall, 'overall')
        # Drop nested by_givens for `overall` to keep the JSON small.
        out['overall'].pop('by_givens', None)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_summary(summary):
    print()
    print("================ FINAL SUMMARY (parse_eval_log) ================")
    for split, s in summary.items():
        if not isinstance(s, dict):
            continue
        print(f"  {split:>8}: n={s.get('n', 0)}  "
              f"cell_acc={100 * s.get('cell_acc', 0.0):.2f}%  "
              f"solved={100 * s.get('fully_solved_pct', 0.0):.2f}%  "
              f"parsed={100 * s.get('parsed_pct', 0.0):.2f}%  "
              f"given={100 * s.get('given_cell_acc', 0.0):.2f}%  "
              f"filled={100 * s.get('filled_cell_acc', 0.0):.2f}%")
    print()
    print("================ BY GIVENS ================")
    print(f"  {'split':>8} {'g':>4} {'n':>5} {'cell_acc':>10} {'solved%':>9}")
    for split, s in summary.items():
        if not isinstance(s, dict):
            continue
        bg = s.get('by_givens', {})
        for g in sorted(bg.keys(), key=lambda x: int(x)):
            r = bg[g]
            print(f"  {split:>8} {g:>4} {r.get('n', 0):>5} "
                  f"{100 * r.get('cell_acc', 0.0):>9.2f}% "
                  f"{100 * r.get('fully_solved_pct', 0.0):>8.2f}%")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--log_file', type=Path, default=None,
                     help='Flat-mode eval.log file emitted by --log_file.')
    src.add_argument('--shards', type=Path, default=None,
                     help='Structured per_split shard root '
                          '(e.g. <run>/results/per_split). Reads rank*.jsonl.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output path. Default: <input dir>/eval_metrics.json')
    parser.add_argument('--quiet', action='store_true',
                        help='Skip stdout summary table; only write the JSON file.')
    args = parser.parse_args()

    if args.log_file is not None:
        rows = parse_eval_log(args.log_file)
        # group by split
        rows_by_split = {}
        for o in rows:
            rows_by_split.setdefault(o.get('split', 'unknown'), []).append(o)
        out_path = args.out or args.log_file.parent / 'eval_metrics.json'
    else:
        rows_by_split = parse_shards(args.shards)
        out_path = args.out or args.shards.parent / 'eval_metrics.json'

    summary = _build_full_summary(rows_by_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[parse_eval_log] wrote {out_path}")

    if not args.quiet:
        _print_summary(summary)


if __name__ == '__main__':
    main()
