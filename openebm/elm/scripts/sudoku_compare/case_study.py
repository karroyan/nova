#!/usr/bin/env python3
"""Export representative Sudoku comparison cases as Markdown."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .common import (
        DEFAULT_DATA_DIR,
        DEFAULT_RESULTS_ROOT,
        LARGE_MODEL_ALIASES,
        SMALL_MODEL_ALIASES,
        format_board,
        load_split,
        read_jsonl,
        sample_boards,
    )
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from openebm.elm.scripts.sudoku_compare.common import (  # type: ignore
        DEFAULT_DATA_DIR,
        DEFAULT_RESULTS_ROOT,
        LARGE_MODEL_ALIASES,
        SMALL_MODEL_ALIASES,
        format_board,
        load_split,
        read_jsonl,
        sample_boards,
    )


def resolve_jsonl_path(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    for candidate in [path / "test.jsonl", path / "results" / "test.jsonl"]:
        if candidate.is_file():
            return candidate
    return None


def rows_by_idx(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        idx = row.get("idx")
        if isinstance(idx, int):
            out[idx] = row
    return out


def discover_llm_jsonls(llm_root: Path, explicit_dirs: List[Path]) -> List[Tuple[str, Path]]:
    paths: List[Tuple[str, Path]] = []
    if explicit_dirs:
        for item in explicit_dirs:
            resolved = resolve_jsonl_path(item)
            if resolved is not None:
                paths.append((item.parent.name if item.is_file() else item.name, resolved))
            else:
                print(f"[case_study] WARNING: no test.jsonl found under {item}")
        return paths

    if not llm_root.is_dir():
        return paths
    for child in sorted(llm_root.iterdir()):
        if not child.is_dir():
            continue
        resolved = resolve_jsonl_path(child)
        if resolved is not None:
            paths.append((child.name, resolved))
    return paths


def is_solved(row: Optional[Dict[str, Any]]) -> bool:
    return bool(row and row.get("fully_solved"))


def is_wrong(row: Optional[Dict[str, Any]]) -> bool:
    return row is not None and not row.get("fully_solved")


def given_count(samples: Sequence[dict], idx: int) -> int:
    puzzle, _ = sample_boards(samples[idx])
    return sum(1 for v in puzzle if int(v) != 0)


def select_cases(
    candidates: List[int],
    samples: Sequence[dict],
    used: set,
    limit: int,
    prefer_low_givens: bool = True,
) -> List[int]:
    ordered = sorted(
        candidates,
        key=lambda idx: (given_count(samples, idx) if prefer_low_givens else 0, idx),
    )
    selected = [idx for idx in ordered if idx not in used][:limit]
    if len(selected) < limit:
        selected.extend(idx for idx in ordered if idx not in selected)  # allow fallback duplicates
    selected = selected[:limit]
    used.update(selected)
    return selected


def cell_acc(row: Optional[Dict[str, Any]]) -> float:
    if not row:
        return 0.0
    total = int(row.get("filled_total", 0) or 0)
    if total <= 0:
        return 0.0
    return float(row.get("filled_correct", 0) or 0) / total


def status(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "missing"
    if not row.get("parsed"):
        return "invalid"
    if row.get("fully_solved"):
        return "correct"
    return f"wrong, blank-cell acc={100.0 * cell_acc(row):.2f}%"


def model_meta(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return ""
    parts = []
    if row.get("parser_strategy"):
        parts.append(f"parser={row.get('parser_strategy')}")
    if row.get("parse_failure_reason"):
        parts.append(f"parse_failure={row.get('parse_failure_reason')}")
    if row.get("tokens_generated") is not None:
        parts.append(f"tokens={row.get('tokens_generated')}")
    if row.get("elapsed_s") is not None:
        try:
            parts.append(f"elapsed={float(row.get('elapsed_s')):.3f}s")
        except Exception:
            pass
    if row.get("response_was_truncated"):
        parts.append("response_in_test_jsonl=truncated; see traces/all.jsonl")
    return "; ".join(parts)


def render_prediction(pred: Any, solution: Sequence[int]) -> str:
    if not isinstance(pred, list) or len(pred) != 81:
        return "INVALID/UNPARSED"
    rows = []
    for r in range(9):
        cells = []
        for c in range(9):
            i = r * 9 + c
            got = pred[i]
            try:
                got_int = int(got)
            except Exception:
                cells.append(f"[{got}]")
                continue
            if got_int == int(solution[i]):
                cells.append(str(got_int))
            else:
                cells.append(f"[{got}]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def model_order(model_names: Sequence[str], small_names: set, large_names: set, ebm_name: str) -> List[str]:
    def key(name: str) -> Tuple[int, str]:
        if name == ebm_name:
            return (0, name)
        if name in small_names:
            return (1, name)
        if name in large_names:
            return (2, name)
        return (3, name)

    return sorted(model_names, key=key)


def write_case(
    lines: List[str],
    title: str,
    idx: int,
    samples: Sequence[dict],
    model_rows: Dict[str, Dict[int, Dict[str, Any]]],
    ordered_models: Sequence[str],
) -> None:
    puzzle, solution = sample_boards(samples[idx])
    lines.append(f"## {title} (idx={idx}, givens={given_count(samples, idx)})")
    lines.append("")
    lines.append("Puzzle:")
    lines.append("```text")
    lines.append(format_board(puzzle))
    lines.append("```")
    lines.append("")
    lines.append("Ground truth:")
    lines.append("```text")
    lines.append(format_board(solution))
    lines.append("```")
    lines.append("")
    lines.append("Predictions, with wrong cells wrapped in brackets:")
    lines.append("")
    for model in ordered_models:
        row = model_rows.get(model, {}).get(idx)
        pred = row.get("pred") if row else None
        lines.append(f"### {model}: {status(row)}")
        meta = model_meta(row)
        if meta:
            lines.append("")
            lines.append(f"`{meta}`")
        lines.append("```text")
        lines.append(render_prediction(pred, solution))
        lines.append("```")
        lines.append("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ebm-result-dir", type=Path, default=None)
    parser.add_argument("--ebm-name", default="EBM")
    parser.add_argument("--llm-root", type=Path, default=None)
    parser.add_argument("--llm-result-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--small-models", nargs="*", default=sorted(SMALL_MODEL_ALIASES))
    parser.add_argument("--large-models", nargs="*", default=sorted(LARGE_MODEL_ALIASES))
    parser.add_argument("--cases-per-type", type=int, default=5)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ebm_result_dir = args.ebm_result_dir or (args.results_root / "ebm")
    llm_root = args.llm_root or (args.results_root / "llm")
    out_md = args.out_md or (args.results_root / "case_study.md")
    samples = load_split(args.data_dir, split="test")

    ebm_jsonl = resolve_jsonl_path(ebm_result_dir)
    if ebm_jsonl is None:
        raise SystemExit(f"No EBM test.jsonl found under {ebm_result_dir}")

    model_rows: Dict[str, Dict[int, Dict[str, Any]]] = {
        args.ebm_name: rows_by_idx(ebm_jsonl),
    }
    for default_name, path in discover_llm_jsonls(llm_root, args.llm_result_dirs):
        rows = rows_by_idx(path)
        model_name = default_name
        for row in rows.values():
            if row.get("model"):
                model_name = str(row["model"])
                break
        model_rows[model_name] = rows

    small_names = set(args.small_models)
    large_names = set(args.large_models)
    available_small = [n for n in model_rows if n in small_names]
    available_large = [n for n in model_rows if n in large_names]
    ordered_models = model_order(list(model_rows), small_names, large_names, args.ebm_name)

    all_indices = sorted(model_rows[args.ebm_name].keys())
    category_a = [
        idx
        for idx in all_indices
        if is_solved(model_rows[args.ebm_name].get(idx))
        and any(is_wrong(model_rows[name].get(idx)) for name in available_small)
    ]
    category_b = [
        idx
        for idx in all_indices
        if is_solved(model_rows[args.ebm_name].get(idx))
        and any(is_wrong(model_rows[name].get(idx)) for name in available_large)
    ]
    def disagreement_count(idx: int) -> int:
        return sum(
            int(is_wrong(rows.get(idx)))
            for model, rows in model_rows.items()
            if model != args.ebm_name
        )

    category_c = sorted(
        all_indices,
        key=lambda idx: (given_count(samples, idx), -disagreement_count(idx), idx),
    )

    used = set()
    selected = [
        (
            "A. EBM correct, same-scale LLM wrong",
            select_cases(category_a, samples, used, args.cases_per_type),
        ),
        (
            "B. EBM correct, large LLM wrong",
            select_cases(category_b, samples, used, args.cases_per_type),
        ),
        (
            "C. Low-given hard puzzles",
            select_cases(category_c, samples, used, args.cases_per_type),
        ),
    ]

    lines: List[str] = [
        "# Sudoku Case Study",
        "",
        f"Data: `{args.data_dir}` SATNet test split.",
        f"EBM rows: `{ebm_jsonl}`.",
        f"LLM root: `{llm_root}`.",
        "",
        "Wrong cells are highlighted with square brackets. `invalid` means the parser could not extract a completed 1-9 final grid.",
        "Full raw LLM outputs and representative correct/wrong/near-miss/parse-failed cases are stored under each model's `traces/` directory.",
        "",
    ]
    for title, indices in selected:
        if not indices:
            lines.append(f"## {title}")
            lines.append("")
            lines.append("No matching case was found in the available result files.")
            lines.append("")
            continue
        for idx in indices:
            write_case(lines, title, idx, samples, model_rows, ordered_models)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[case_study] wrote {out_md}")


if __name__ == "__main__":
    main()
