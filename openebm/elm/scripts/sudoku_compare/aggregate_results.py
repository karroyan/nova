#!/usr/bin/env python3
"""Aggregate EBM and LLM Sudoku results into a comparison table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .common import DEFAULT_MODEL_PARAMS, DEFAULT_RESULTS_ROOT, load_json, pct
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from openebm.elm.scripts.sudoku_compare.common import (  # type: ignore
        DEFAULT_MODEL_PARAMS,
        DEFAULT_RESULTS_ROOT,
        load_json,
        pct,
    )


def resolve_summary_path(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    candidates = [
        path / "summary.json",
        path / "results" / "summary.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def extract_metric_row(
    summary_path: Path,
    model_name: Optional[str] = None,
    params: Optional[str] = None,
    split: str = "test",
) -> Dict[str, Any]:
    raw = load_json(summary_path)
    if split in raw or "overall" in raw:
        # EBM structured output from eval_sudoku_samples.py.
        split_summary = raw.get(split) or raw.get("overall") or {}
        model = model_name or "EBM"
        param_str = params or DEFAULT_MODEL_PARAMS.get("ebm", "")
        return {
            "model": model,
            "params": param_str,
            "board_acc": float(split_summary.get("fully_solved_pct", 0.0)),
            "cell_acc": float(
                split_summary.get("filled_cell_acc", split_summary.get("cell_acc", 0.0))
            ),
            "valid_rate": float(split_summary.get("parsed_pct", 0.0)),
            "avg_time": float(split_summary.get("avg_s_per_sample", 0.0)),
            "n": int(split_summary.get("n", 0)),
            "source": str(summary_path),
        }

    model = model_name or raw.get("model") or summary_path.parent.name
    param_str = params or raw.get("params") or DEFAULT_MODEL_PARAMS.get(str(model), "")
    return {
        "model": model,
        "params": param_str,
        "board_acc": float(raw.get("board_acc", raw.get("fully_solved_pct", 0.0))),
        "cell_acc": float(raw.get("cell_acc", raw.get("filled_cell_acc", 0.0))),
        "valid_rate": float(raw.get("valid_rate", raw.get("parsed_pct", 0.0))),
        "avg_time": float(raw.get("avg_time", raw.get("avg_s_per_sample", 0.0))),
        "n": int(raw.get("n", 0)),
        "source": str(summary_path),
    }


def discover_llm_summary_paths(llm_root: Path, explicit_dirs: List[Path]) -> List[Path]:
    if explicit_dirs:
        paths = []
        for item in explicit_dirs:
            resolved = resolve_summary_path(item)
            if resolved is not None:
                paths.append(resolved)
            else:
                print(f"[aggregate_results] WARNING: no summary found under {item}")
        return paths

    if not llm_root.is_dir():
        return []
    paths = []
    for child in sorted(llm_root.iterdir()):
        if not child.is_dir():
            continue
        resolved = resolve_summary_path(child)
        if resolved is not None:
            paths.append(resolved)
    return paths


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["model", "params", "board_acc", "cell_acc", "valid_rate", "avg_time"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model", "")),
                    str(row.get("params", "")),
                    pct(float(row.get("board_acc", 0.0))),
                    pct(float(row.get("cell_acc", 0.0))),
                    pct(float(row.get("valid_rate", 0.0))),
                    f"{float(row.get('avg_time', 0.0)):.3f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "params", "board_acc", "cell_acc", "valid_rate", "avg_time", "n", "source"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--ebm-result-dir", type=Path, default=None)
    parser.add_argument("--ebm-name", default="EBM")
    parser.add_argument("--ebm-params", default=DEFAULT_MODEL_PARAMS.get("ebm", "~1B"))
    parser.add_argument("--llm-root", type=Path, default=None)
    parser.add_argument("--llm-result-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--no-ebm", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ebm_result_dir = args.ebm_result_dir or (args.results_root / "ebm")
    llm_root = args.llm_root or (args.results_root / "llm")
    csv_out = args.csv_out or (args.results_root / "comparison.csv")
    rows: List[Dict[str, Any]] = []

    if not args.no_ebm:
        ebm_summary = resolve_summary_path(ebm_result_dir)
        if ebm_summary is None:
            print(f"[aggregate_results] WARNING: no EBM summary found under {ebm_result_dir}")
        else:
            rows.append(
                extract_metric_row(
                    ebm_summary,
                    model_name=args.ebm_name,
                    params=args.ebm_params,
                    split=args.split,
                )
            )

    llm_paths = discover_llm_summary_paths(llm_root, args.llm_result_dirs)
    for summary_path in llm_paths:
        rows.append(extract_metric_row(summary_path, split=args.split))

    if not rows:
        raise SystemExit("No summaries found. Run EBM/LLM evaluation first or pass result dirs.")

    print(markdown_table(rows))
    write_csv(csv_out, rows)
    print(f"\n[aggregate_results] wrote {csv_out}")


if __name__ == "__main__":
    main()
