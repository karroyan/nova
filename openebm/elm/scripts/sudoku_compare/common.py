#!/usr/bin/env python3
"""Shared helpers for Sudoku comparison evaluation.

The EBM evaluator already defines the dataset split files, board formatting,
and Sudoku validity check. This module imports and reuses those definitions
when possible, with a minimal fallback so the LLM scripts remain debuggable in
lightweight environments.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openebm.elm.scripts import eval_sudoku_samples as ebm_eval

    DEFAULT_DATA_DIR = Path(ebm_eval.DEFAULT_DATA_DIR)
    SPLIT_FILES = list(ebm_eval.SPLIT_FILES)
    _ebm_format_board = ebm_eval._format_board
    _ebm_is_valid_sudoku_grid = ebm_eval._is_valid_sudoku_grid
    _ebm_load_splits = ebm_eval.load_splits
except Exception:  # pragma: no cover - used only if EBM imports are unavailable.
    DEFAULT_DATA_DIR = Path(
        "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2"
    )
    SPLIT_FILES = [
        ("train", "rrn_train.pt"),
        ("val", "rrn_val.pt"),
        ("test", "satnet_test.pt"),
    ]

    def _ebm_format_board(values: Sequence[int]) -> str:
        return "\n".join(
            " ".join(str(int(v)) for v in values[r * 9 : (r + 1) * 9])
            for r in range(9)
        )

    def _ebm_is_valid_sudoku_grid(pred: Optional[List[int]]) -> bool:
        if pred is None or len(pred) != 81:
            return False
        if not all(isinstance(v, int) and 1 <= v <= 9 for v in pred):
            return False
        for r in range(9):
            if len(set(pred[r * 9 : (r + 1) * 9])) != 9:
                return False
        for c in range(9):
            if len({pred[r * 9 + c] for r in range(9)}) != 9:
                return False
        for br in range(3):
            for bc in range(3):
                box = [
                    pred[(br * 3 + dr) * 9 + (bc * 3 + dc)]
                    for dr in range(3)
                    for dc in range(3)
                ]
                if len(set(box)) != 9:
                    return False
        return True

    def _ebm_load_splits(data_dir: Path) -> Dict[str, list]:
        import torch

        out: Dict[str, list] = {}
        for split, fname in SPLIT_FILES:
            path = Path(data_dir) / fname
            if path.is_file():
                out[split] = torch.load(path, weights_only=False)
        return out


DEFAULT_RESULTS_ROOT = Path(
    "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare"
)

DEFAULT_HF_BASE = Path(
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub"
)

DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "qwen3_1p7b": DEFAULT_HF_BASE / "models--Qwen--Qwen3-1.7B",
    "llama3p2_1b": DEFAULT_HF_BASE / "models--meta-llama--Llama-3.2-1B-Instruct",
    "qwen3p6_27b": DEFAULT_HF_BASE / "models--Qwen--Qwen3.6-27B",
    "deepseek_r1_0528": DEFAULT_HF_BASE / "models--deepseek-ai--DeepSeek-R1-0528",
}

DEFAULT_MODEL_PARAMS: Dict[str, str] = {
    "qwen3_1p7b": "1.7B",
    "llama3p2_1b": "1B",
    "qwen3p6_27b": "27B",
    "deepseek_r1_0528": "671B",
    "ebm": "~1B",
}

SMALL_MODEL_ALIASES = {"qwen3_1p7b", "llama3p2_1b"}
LARGE_MODEL_ALIASES = {"qwen3p6_27b", "deepseek_r1_0528"}


@dataclass
class ParseResult:
    pred: Optional[List[int]]
    strategy: str
    failure_reason: str = ""
    candidate_count: int = 0


def flatten_board(board: Any) -> List[int]:
    """Return a row-major list of 81 ints from tensor/list/nested-list samples."""
    if hasattr(board, "tolist"):
        board = board.tolist()
    if len(board) == 81 and not hasattr(board[0], "__iter__"):
        return [int(v) for v in board]
    flat: List[int] = []
    for row in board:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if hasattr(row, "__iter__") and not isinstance(row, (str, bytes)):
            flat.extend(int(v) for v in row)
        else:
            flat.append(int(row))
    if len(flat) != 81:
        raise ValueError(f"Expected 81 cells, got {len(flat)}")
    return flat


def format_board(values: Sequence[int], blank: str = "0") -> str:
    if len(values) != 81:
        raise ValueError(f"Expected 81 cells, got {len(values)}")
    rendered = [blank if int(v) == 0 else str(int(v)) for v in values]
    return "\n".join(
        " ".join(rendered[r * 9 : (r + 1) * 9])
        for r in range(9)
    )


def load_split(data_dir: Path, split: str = "test") -> List[dict]:
    split_to_file = dict(SPLIT_FILES)
    if split not in split_to_file:
        raise ValueError(f"Unknown split '{split}', expected one of {sorted(split_to_file)}")
    path = Path(data_dir) / split_to_file[split]
    if not path.is_file():
        raise FileNotFoundError(f"Sudoku split file not found: {path}")
    import torch

    return list(torch.load(path, weights_only=False))


def select_indices(total: int, start_index: int = 0, num_samples: int = -1) -> List[int]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    stop = total if num_samples is None or num_samples < 0 else min(total, start_index + num_samples)
    return list(range(min(start_index, total), stop))


def sample_boards(sample: dict) -> Tuple[List[int], List[int]]:
    return flatten_board(sample["puzzle"]), flatten_board(sample["solution"])


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "\n", text, flags=re.IGNORECASE | re.DOTALL)


def _priority_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    markers = [
        ("final_answer", r"final\s+answer\s*:"),
        ("final_grid", r"final\s+grid\s*:"),
        ("completed_grid", r"completed\s+grid\s*:"),
        ("solution", r"solution\s*:"),
        ("answer", r"answer\s*:"),
    ]
    for label, marker in markers:
        matches = list(re.finditer(marker, text, flags=re.IGNORECASE))
        if matches:
            sections.append((f"after_{label}", text[matches[-1].end() :]))

    close_tag = text.lower().rfind("</think>")
    if close_tag >= 0:
        sections.append(("after_think", text[close_tag + len("</think>") :]))

    stripped = _strip_thinking(text)
    sections.append(("without_think", stripped))
    sections.append(("raw", text))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for name, section in sections:
        key = section.strip()
        if key and key not in seen:
            deduped.append((name, section))
            seen.add(key)
    return deduped


def _row_digits(line: str) -> Optional[List[int]]:
    s = line.strip()
    if not s:
        return None
    s = s.strip("` ")
    if "|" in s:
        cells = [cell.strip() for cell in s.strip("|").split("|")]
        cells = [cell for cell in cells if cell]
        if len(cells) == 10 and re.fullmatch(r"(?:row|r)?\s*[1-9]", cells[0], flags=re.I):
            cells = cells[1:]
        if len(cells) == 9 and all(re.fullmatch(r"[0-9]", cell) for cell in cells):
            return [int(cell) for cell in cells]
    s = re.sub(r"^\s*(?:[-*]\s*)?(?:row|r)?\s*[1-9]\s*[:.)|-]\s*", "", s, flags=re.I)
    s = s.strip("`| ")
    digits = [int(ch) for ch in re.findall(r"[0-9]", s)]
    if len(digits) == 9:
        return digits
    return None


def _is_completed_board(pred: Sequence[int]) -> bool:
    return len(pred) == 81 and all(isinstance(v, int) and 1 <= int(v) <= 9 for v in pred)


def _extract_grid_line_candidates(text: str) -> List[List[int]]:
    rows: List[List[int]] = []
    candidates: List[List[int]] = []
    for line in text.splitlines():
        row = _row_digits(line)
        if row is None:
            # Blank/separator lines inside markdown tables should not erase a
            # nearly complete grid; prose should.
            if rows and re.fullmatch(r"[\s`|:.-]*", line):
                continue
            if len(rows) >= 9:
                candidates.append([v for r in rows[-9:] for v in r])
            rows = []
            continue
        rows.append(row)
        if len(rows) >= 9:
            tail = rows[-9:]
            candidates.append([v for r in tail for v in r])
    if len(rows) >= 9:
        candidates.append([v for r in rows[-9:] for v in r])
    return candidates


def _extract_digit_run_candidates(text: str) -> List[List[int]]:
    candidates: List[List[int]] = []
    for match in re.finditer(r"[0-9]{81,}", text):
        run = match.group(0)
        candidates.append([int(ch) for ch in run[-81:]])
    return candidates


def _extract_whitespace_digit_candidates(text: str) -> List[List[int]]:
    nums = [int(tok) for tok in text.split() if tok.isdigit() and len(tok) == 1]
    if len(nums) >= 81:
        return [nums[-81:]]
    return []


def _extract_all_digit_candidates(text: str) -> List[List[int]]:
    nums = [int(ch) for ch in re.findall(r"[0-9]", text)]
    if len(nums) >= 81:
        return [nums[-81:]]
    return []


def _candidate_rows(section_name: str, section: str) -> List[Tuple[List[int], str]]:
    """Return candidate final grids in encounter order for one text section."""
    candidates: List[Tuple[List[int], str]] = []
    for pred in _extract_grid_line_candidates(section):
        candidates.append((pred, f"{section_name}:grid_lines"))
    for pred in _extract_digit_run_candidates(section):
        candidates.append((pred, f"{section_name}:digit_run"))

    # Free-form all-digit extraction is intentionally limited to explicit answer
    # sections. In raw chain-of-thought it can accidentally collect coordinates
    # or candidate notes and turn them into a bogus Sudoku grid.
    explicit_answer_section = section_name.startswith("after_") and section_name != "after_think"
    if explicit_answer_section:
        for pred in _extract_whitespace_digit_candidates(section):
            candidates.append((pred, f"{section_name}:whitespace_digits"))
        for pred in _extract_all_digit_candidates(section):
            candidates.append((pred, f"{section_name}:all_digits"))
    return candidates


def parse_llm_board(text: str) -> ParseResult:
    """Extract the final 81 Sudoku digits from raw model output.

    Parsed means we found a completed 9x9 answer grid containing only digits
    1-9. Echoed puzzles or intermediate grids with 0/`.` blanks are treated as
    parse failures instead of answer grids.
    """
    if not text:
        return ParseResult(None, "empty", failure_reason="empty_response")

    total_candidates = 0
    rejected_incomplete = 0
    for section_name, section in _priority_sections(text):
        candidates = _candidate_rows(section_name, section)
        total_candidates += len(candidates)
        for extractor_name in ["grid_lines", "digit_run", "whitespace_digits", "all_digits"]:
            valid_candidates = [
                (pred, strategy)
                for pred, strategy in candidates
                if strategy.endswith(f":{extractor_name}")
                and len(pred) == 81
                and _is_completed_board(pred)
            ]
            if valid_candidates:
                pred, strategy = valid_candidates[-1]
                return ParseResult(pred, strategy, candidate_count=total_candidates)
        rejected_incomplete += sum(1 for pred, _ in candidates if len(pred) == 81)

    if rejected_incomplete:
        return ParseResult(
            None,
            "failed:incomplete_or_non_1_9_grid",
            failure_reason="found_81_cells_but_not_completed_1_9_grid",
            candidate_count=total_candidates,
        )
    return ParseResult(
        None,
        "failed:no_grid",
        failure_reason="no_completed_9x9_grid_found",
        candidate_count=total_candidates,
    )


def infer_response_format(response: Optional[str], pred: Optional[List[int]]) -> str:
    if response is None or pred is None:
        return "unknown"
    if any(len(run) >= 81 for run in re.findall(r"[0-9]+", response)):
        return "flat"
    return "grid"


def score_prediction(
    puzzle: Sequence[int],
    solution: Sequence[int],
    pred: Optional[List[int]],
    elapsed_s: float = 0.0,
    tokens_generated: int = 0,
) -> Dict[str, Any]:
    puzzle = [int(v) for v in puzzle]
    solution = [int(v) for v in solution]
    pred_ints: Optional[List[int]] = None
    try:
        pred_len = len(pred) if pred is not None else 0
    except Exception:
        pred_len = 0
    if pred is not None and pred_len == 81:
        try:
            pred_ints = [int(v) for v in pred]
        except Exception:
            pred_ints = None
    parsed = pred_ints is not None
    given_mask = [v != 0 for v in puzzle]
    given_total = sum(given_mask)
    filled_total = 81 - given_total

    correct = 0
    given_correct = 0
    filled_correct = 0
    if parsed:
        assert pred_ints is not None
        for i, got in enumerate(pred_ints):
            if got == solution[i]:
                correct += 1
                if given_mask[i]:
                    given_correct += 1
                else:
                    filled_correct += 1

    fully_solved = parsed and correct == 81
    valid_sudoku = parsed and _ebm_is_valid_sudoku_grid(pred_ints)
    return {
        "correct": correct,
        "total": 81,
        "given_total": given_total,
        "given_correct": given_correct,
        "filled_total": filled_total,
        "filled_correct": filled_correct,
        "parsed": bool(parsed),
        "fully_solved": bool(fully_solved),
        "is_valid_sudoku": bool(valid_sudoku),
        "elapsed_s": float(elapsed_s),
        "tokens_generated": int(tokens_generated or 0),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], model: str, params: str = "") -> Dict[str, Any]:
    n = len(rows)
    cell_correct = sum(int(r.get("correct", 0)) for r in rows)
    cell_total = sum(int(r.get("total", 0)) for r in rows)
    given_total = sum(int(r.get("given_total", 0)) for r in rows)
    given_correct = sum(int(r.get("given_correct", 0)) for r in rows)
    filled_total = sum(int(r.get("filled_total", 0)) for r in rows)
    filled_correct = sum(int(r.get("filled_correct", 0)) for r in rows)
    parsed = sum(1 for r in rows if r.get("parsed"))
    solved = sum(1 for r in rows if r.get("fully_solved"))
    valid_sudoku = sum(1 for r in rows if r.get("is_valid_sudoku"))
    elapsed = sum(float(r.get("elapsed_s", 0.0) or 0.0) for r in rows)
    tokens = sum(int(r.get("tokens_generated", 0) or 0) for r in rows)

    by_givens: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        g = int(r.get("given_total", -1))
        if g < 0:
            continue
        slot = by_givens.setdefault(
            str(g),
            {
                "n": 0,
                "filled_correct": 0,
                "filled_total": 0,
                "fully_solved": 0,
                "parsed": 0,
            },
        )
        slot["n"] += 1
        slot["filled_correct"] += int(r.get("filled_correct", 0))
        slot["filled_total"] += int(r.get("filled_total", 0))
        slot["fully_solved"] += int(bool(r.get("fully_solved")))
        slot["parsed"] += int(bool(r.get("parsed")))
    for slot in by_givens.values():
        slot["cell_acc"] = (
            slot["filled_correct"] / slot["filled_total"] if slot["filled_total"] else 0.0
        )
        slot["board_acc"] = slot["fully_solved"] / slot["n"] if slot["n"] else 0.0
        slot["valid_rate"] = slot["parsed"] / slot["n"] if slot["n"] else 0.0

    filled_acc = filled_correct / filled_total if filled_total else 0.0
    all_cell_acc = cell_correct / cell_total if cell_total else 0.0
    summary = {
        "model": model,
        "params": params,
        "split": "test",
        "n": n,
        "board_acc": solved / n if n else 0.0,
        # Main cell metric requested for the comparison: original blank cells only.
        "cell_acc": filled_acc,
        "filled_cell_acc": filled_acc,
        "all_cell_acc": all_cell_acc,
        "valid_rate": parsed / n if n else 0.0,
        "valid_sudoku_rate": valid_sudoku / n if n else 0.0,
        "avg_time": elapsed / n if n else 0.0,
        "avg_s_per_sample": elapsed / n if n else 0.0,
        "avg_tokens_per_sample": tokens / n if n else 0.0,
        "cell_correct": cell_correct,
        "cell_total": cell_total,
        "given_total": given_total,
        "given_correct": given_correct,
        "given_cell_acc": given_correct / given_total if given_total else 0.0,
        "filled_total": filled_total,
        "filled_correct": filled_correct,
        "parsed": parsed,
        "parsed_pct": parsed / n if n else 0.0,
        "fully_solved": solved,
        "fully_solved_pct": solved / n if n else 0.0,
        "valid_sudoku": valid_sudoku,
        "valid_sudoku_pct": valid_sudoku / n if n else 0.0,
        "elapsed_s": elapsed,
        "tokens_generated": tokens,
        "by_givens": dict(sorted(by_givens.items(), key=lambda kv: int(kv[0]))),
    }
    return summary


def read_jsonl(path: Path, strict: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    bad = 0
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise
                bad += 1
                if bad <= 3:
                    print(f"[sudoku_compare] WARNING: skipping malformed JSONL line {path}:{lineno}")
    if bad > 3:
        print(f"[sudoku_compare] WARNING: skipped {bad} malformed JSONL lines in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def dedupe_sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        idx = row.get("idx")
        if isinstance(idx, int):
            by_idx[idx] = row
    return [by_idx[idx] for idx in sorted(by_idx)]


def sanitize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-") or "model"


def resolve_hf_snapshot(path: Path, revision: Optional[str] = None) -> Path:
    """Resolve either a normal HF folder or a cache root with snapshots/."""
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    if revision:
        candidate = snapshots / revision
        if (candidate / "config.json").is_file():
            return candidate
    ref_main = path / "refs" / "main"
    if ref_main.is_file():
        ref = ref_main.read_text().strip()
        candidate = snapshots / ref
        if (candidate / "config.json").is_file():
            return candidate
    if snapshots.is_dir():
        candidates = [p for p in snapshots.iterdir() if (p / "config.json").is_file()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return path


def model_spec_to_name_path(spec: str) -> Tuple[str, Path]:
    """Parse NAME=PATH, a default alias, or a raw local path."""
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        return sanitize_model_name(name), Path(raw_path)
    if spec in DEFAULT_MODEL_PATHS:
        return spec, DEFAULT_MODEL_PATHS[spec]
    path = Path(spec)
    return sanitize_model_name(path.name), path


def params_for_model(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    return DEFAULT_MODEL_PARAMS.get(name, "")


def pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{100.0 * float(value):.2f}%"
