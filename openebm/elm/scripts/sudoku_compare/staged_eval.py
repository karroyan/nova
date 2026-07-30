#!/usr/bin/env python3
"""Utilities for staged Sudoku comparison runs.

The shell launcher owns model execution. This module handles deterministic
stage bookkeeping, EBM result reuse/materialization, and per-stage reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_DATA_DIR = Path(
    "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2"
)
DEFAULT_RESULTS_ROOT = Path(
    "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare"
)
DEFAULT_MODEL_PARAMS: Dict[str, str] = {
    "qwen3_1p7b": "1.7B",
    "llama3p2_1b": "1B",
    "qwen3p6_27b": "27B",
    "deepseek_r1_0528": "671B",
    "ebm": "~1B",
}
SMALL_MODEL_ALIASES = {"qwen3_1p7b", "llama3p2_1b"}
LARGE_MODEL_ALIASES = {"qwen3p6_27b", "deepseek_r1_0528"}

_COMMON: Any = None


def common_module() -> Any:
    global _COMMON
    if _COMMON is not None:
        return _COMMON
    try:
        from . import common as mod
    except ImportError:  # pragma: no cover - direct file execution fallback.
        import sys

        sys.path.append(str(Path(__file__).resolve().parents[4]))
        from openebm.elm.scripts.sudoku_compare import common as mod  # type: ignore

    _COMMON = mod
    return mod


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


DEFAULT_MODEL_ORDER = [
    "qwen3_1p7b",
    "llama3p2_1b",
    "qwen3p6_27b",
    "deepseek_r1_0528",
    "EBM",
]


def dedupe_sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return common_module().dedupe_sort_rows(rows)


def format_board(values: Sequence[int], blank: str = "0") -> str:
    return common_module().format_board(values, blank=blank)


def load_split(data_dir: Path, split: str = "test") -> List[dict]:
    return common_module().load_split(data_dir, split=split)


def pct(value: float) -> str:
    return common_module().pct(value)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return common_module().read_jsonl(path)


def sample_boards(sample: dict) -> Tuple[List[int], List[int]]:
    return common_module().sample_boards(sample)


def summarize_rows(rows: Sequence[Dict[str, Any]], model: str, params: str = "") -> Dict[str, Any]:
    return common_module().summarize_rows(rows, model=model, params=params)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    return common_module().write_json(path, obj)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    return common_module().write_jsonl(path, rows)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def resolve_test_jsonl(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    for candidate in [path / "results" / "test.jsonl", path / "test.jsonl"]:
        if candidate.is_file():
            return candidate
    return None


def load_result_rows(path: Path) -> List[Dict[str, Any]]:
    jsonl = resolve_test_jsonl(path)
    if jsonl is None:
        return []
    return dedupe_sort_rows(read_jsonl(jsonl))


def filter_rows(rows: Iterable[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        idx = row.get("idx")
        if isinstance(idx, int) and start <= idx < end:
            out.append(row)
    return dedupe_sort_rows(out)


def coverage(rows: Sequence[Dict[str, Any]], start: int, end: int) -> Dict[str, Any]:
    present = {row.get("idx") for row in rows if isinstance(row.get("idx"), int)}
    missing = [idx for idx in range(start, end) if idx not in present]
    return {
        "start": start,
        "end": end,
        "expected": max(0, end - start),
        "present": max(0, end - start) - len(missing),
        "missing": len(missing),
        "missing_first": missing[:20],
        "complete": not missing,
    }


def ebm_structured_summary(
    rows: Sequence[Dict[str, Any]],
    params: str,
    source: str,
    start: int,
    end: int,
    stage_index: int,
) -> Dict[str, Any]:
    base = summarize_rows(rows, model="EBM", params=params)
    split = {
        "n": base["n"],
        "cell_correct": base["cell_correct"],
        "cell_total": base["cell_total"],
        "cell_acc": base["all_cell_acc"],
        "parsed": base["parsed"],
        "parsed_pct": base["parsed_pct"],
        "fully_solved": base["fully_solved"],
        "fully_solved_pct": base["fully_solved_pct"],
        "given_total": base["given_total"],
        "given_correct": base["given_correct"],
        "given_cell_acc": base["given_cell_acc"],
        "filled_total": base["filled_total"],
        "filled_correct": base["filled_correct"],
        "filled_cell_acc": base["filled_cell_acc"],
        "valid_sudoku": base["valid_sudoku"],
        "valid_sudoku_pct": base["valid_sudoku_pct"],
        "elapsed_s": base["elapsed_s"],
        "tokens_generated": base["tokens_generated"],
        "avg_tokens_per_sample": base["avg_tokens_per_sample"],
        "avg_s_per_sample": base["avg_s_per_sample"],
        "by_givens": base.get("by_givens", {}),
    }
    return {
        "test": split,
        "overall": dict(split),
        "staged_view": {
            "created_at": now_iso(),
            "source": source,
            "start_index": start,
            "end_index": end,
            "stage_index": stage_index,
            "range_semantics": "[start_index, end_index)",
        },
    }


def materialize_ebm(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.end_index < args.start_index:
        raise SystemExit("invalid EBM materialization range")

    checked: List[Dict[str, Any]] = []
    for source_dir in args.source_dirs:
        rows = load_result_rows(source_dir)
        cov = coverage(rows, args.start_index, args.end_index)
        checked.append({"source": str(source_dir), **cov})
        if not cov["complete"]:
            continue

        selected = filter_rows(rows, args.start_index, args.end_index)
        results_dir = args.out_dir / "results"
        config_dir = args.out_dir / "config"
        results_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(results_dir / "test.jsonl", selected)
        summary = ebm_structured_summary(
            selected,
            params=args.params,
            source=str(source_dir),
            start=args.start_index,
            end=args.end_index,
            stage_index=args.stage_index,
        )
        write_json(results_dir / "summary.json", summary)
        write_json(
            config_dir / "staged_view.json",
            {
                "created_at": now_iso(),
                "source": str(source_dir),
                "out_dir": str(args.out_dir),
                "start_index": args.start_index,
                "end_index": args.end_index,
                "stage_index": args.stage_index,
                "coverage": cov,
            },
        )
        write_json(
            args.out_dir / "status.json",
            {
                "exit_code": 0,
                "message": "materialized staged EBM view from existing rows",
                "updated_at": now_iso(),
            },
        )
        print(
            "[staged_eval] materialized EBM "
            f"source={source_dir} out={args.out_dir} n={len(selected)} "
            f"range=[{args.start_index},{args.end_index})"
        )
        return

    print(json.dumps({"checked": checked}, ensure_ascii=False, indent=2))
    raise SystemExit(
        "No EBM source covers the requested range "
        f"[{args.start_index},{args.end_index})."
    )


def discover_llm_rows(llm_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not llm_root.is_dir():
        return out
    for child in sorted(llm_root.iterdir()):
        if not child.is_dir():
            continue
        rows = load_result_rows(child)
        if not rows:
            continue
        model = child.name
        for row in rows:
            if row.get("model"):
                model = str(row["model"])
                break
        out[model] = rows
    return out


def ordered_model_names(names: Iterable[str], explicit_order: Sequence[str]) -> List[str]:
    order = {name: idx for idx, name in enumerate(explicit_order)}
    return sorted(names, key=lambda name: (order.get(name, len(order)), name))


def metric_row(
    model: str,
    rows: Sequence[Dict[str, Any]],
    expected_n: int,
    params: str = "",
) -> Dict[str, Any]:
    param_label = params or DEFAULT_MODEL_PARAMS.get(model, DEFAULT_MODEL_PARAMS.get(model.lower(), ""))
    summary = summarize_rows(rows, model=model, params=param_label)
    summary["expected_n"] = expected_n
    summary["missing_n"] = max(0, expected_n - int(summary.get("n", 0) or 0))
    return summary


def write_comparison_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "params",
        "n",
        "expected_n",
        "missing_n",
        "board_acc",
        "cell_acc",
        "valid_rate",
        "parsed_pct",
        "valid_sudoku_pct",
        "avg_time",
        "avg_tokens_per_sample",
        "elapsed_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["model", "params", "n", "missing", "board_acc", "cell_acc", "valid_rate", "avg_time"]
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
                    str(row.get("n", 0)),
                    str(row.get("missing_n", 0)),
                    pct(float(row.get("board_acc", 0.0) or 0.0)),
                    pct(float(row.get("cell_acc", 0.0) or 0.0)),
                    pct(float(row.get("valid_rate", 0.0) or 0.0)),
                    f"{float(row.get('avg_time', 0.0) or 0.0):.3f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def status(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "missing"
    if not row.get("parsed"):
        return "invalid"
    if row.get("fully_solved"):
        return "correct"
    total = int(row.get("filled_total", 0) or 0)
    acc = 0.0 if total <= 0 else float(row.get("filled_correct", 0) or 0) / total
    return f"wrong, blank-cell acc={100.0 * acc:.2f}%"


def model_meta(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return ""
    parts: List[str] = []
    for key, label in [
        ("parser_strategy", "parser"),
        ("parse_failure_reason", "parse_failure"),
        ("tokens_generated", "tokens"),
    ]:
        if row.get(key) not in (None, ""):
            parts.append(f"{label}={row.get(key)}")
    if row.get("elapsed_s") is not None:
        try:
            parts.append(f"elapsed={float(row.get('elapsed_s')):.3f}s")
        except Exception:
            pass
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
            cells.append(str(got_int) if got_int == int(solution[i]) else f"[{got}]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def given_count(samples: Sequence[dict], idx: int) -> int:
    puzzle, _ = sample_boards(samples[idx])
    return sum(1 for v in puzzle if int(v) != 0)


def select_cases(candidates: Sequence[int], samples: Sequence[dict], limit: int) -> List[int]:
    return sorted(set(candidates), key=lambda idx: (given_count(samples, idx), idx))[:limit]


def write_case_study(
    path: Path,
    samples: Sequence[dict],
    model_rows: Dict[str, Dict[int, Dict[str, Any]]],
    ordered_models: Sequence[str],
    start: int,
    end: int,
    cases_per_type: int,
    label: str,
) -> None:
    ebm_rows = model_rows.get("EBM", {})
    small_names = set(SMALL_MODEL_ALIASES)
    large_names = set(LARGE_MODEL_ALIASES)
    all_indices = [idx for idx in range(start, end) if idx in ebm_rows]

    def solved(model: str, idx: int) -> bool:
        return bool(model_rows.get(model, {}).get(idx, {}).get("fully_solved"))

    def wrong(model: str, idx: int) -> bool:
        row = model_rows.get(model, {}).get(idx)
        return row is not None and not row.get("fully_solved")

    category_a = [
        idx
        for idx in all_indices
        if solved("EBM", idx) and any(wrong(model, idx) for model in ordered_models if model in small_names)
    ]
    category_b = [
        idx
        for idx in all_indices
        if solved("EBM", idx) and any(wrong(model, idx) for model in ordered_models if model in large_names)
    ]
    parse_fail = [
        idx
        for idx in range(start, end)
        if any(
            row.get(idx) is not None and not row.get(idx, {}).get("parsed")
            for model, row in model_rows.items()
            if model != "EBM"
        )
    ]
    low_given = sorted(range(start, end), key=lambda idx: (given_count(samples, idx), idx))
    selected = [
        ("A. EBM correct, same-scale LLM wrong", select_cases(category_a, samples, cases_per_type)),
        ("B. EBM correct, large LLM wrong", select_cases(category_b, samples, cases_per_type)),
        ("C. Low-given hard puzzles", low_given[:cases_per_type]),
        ("D. LLM parse failures", select_cases(parse_fail, samples, cases_per_type)),
    ]

    lines = [
        f"# Sudoku Case Study ({label})",
        "",
        f"Range: `[{start}, {end})`.",
        "",
        "Wrong cells are highlighted with square brackets. `invalid` means the parser could not extract a completed 1-9 final grid.",
        "",
    ]
    for title, indices in selected:
        lines.append(f"## {title}")
        lines.append("")
        if not indices:
            lines.append("No matching case was found in this range.")
            lines.append("")
            continue
        for idx in indices:
            puzzle, solution = sample_boards(samples[idx])
            lines.append(f"### idx={idx}, givens={given_count(samples, idx)}")
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
            for model in ordered_models:
                row = model_rows.get(model, {}).get(idx)
                pred = row.get("pred") if row else None
                lines.append(f"#### {model}: {status(row)}")
                meta = model_meta(row)
                if meta:
                    lines.append(f"`{meta}`")
                lines.append("```text")
                lines.append(render_prediction(pred, solution))
                lines.append("```")
                lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_one_report(
    out_dir: Path,
    label: str,
    rows_by_model: Dict[str, List[Dict[str, Any]]],
    samples: Sequence[dict],
    start: int,
    end: int,
    ordered_models: Sequence[str],
    cases_per_type: int,
) -> List[Dict[str, Any]]:
    expected_n = max(0, end - start)
    filtered_by_model = {
        model: filter_rows(rows, start, end)
        for model, rows in rows_by_model.items()
    }
    rows = [
        metric_row(model, filtered_by_model.get(model, []), expected_n)
        for model in ordered_models
        if model in rows_by_model
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(out_dir / "comparison.csv", rows)
    table = markdown_table(rows)
    (out_dir / "comparison.md").write_text(
        "\n".join(
            [
                f"# Sudoku Comparison ({label})",
                "",
                f"Range: `[{start}, {end})`, expected samples per model: `{expected_n}`.",
                "",
                table,
                "",
                "Metric口径: `board_acc` 以完整 9x9 全对为正确；`cell_acc` 只统计原题空格位置；解析失败计入分母且视为错误。",
                "",
            ]
        )
    )
    model_rows = {
        model: {int(row["idx"]): row for row in filter_rows(rows0, start, end) if isinstance(row.get("idx"), int)}
        for model, rows0 in rows_by_model.items()
    }
    write_case_study(
        out_dir / "case_study.md",
        samples=samples,
        model_rows=model_rows,
        ordered_models=ordered_models,
        start=start,
        end=end,
        cases_per_type=cases_per_type,
        label=label,
    )
    write_json(
        out_dir / "summary.json",
        {
            "label": label,
            "range": {"start": start, "end": end, "expected_n": expected_n},
            "metrics": rows,
            "comparison_csv": str(out_dir / "comparison.csv"),
            "comparison_md": str(out_dir / "comparison.md"),
            "case_study_md": str(out_dir / "case_study.md"),
        },
    )
    return rows


def write_stage_reports(args: argparse.Namespace) -> None:
    samples = load_split(args.data_dir, split="test")
    rows_by_model = discover_llm_rows(args.llm_root)
    ebm_rows = load_result_rows(args.ebm_dir)
    if ebm_rows:
        rows_by_model["EBM"] = ebm_rows
    if not rows_by_model:
        raise SystemExit("No model rows found for staged report.")

    ordered = ordered_model_names(rows_by_model.keys(), args.model_order)
    stage_dir = args.out_dir / "stages" / f"stage_{args.stage_index:04d}"
    current_rows = write_one_report(
        stage_dir / "current",
        label=f"stage {args.stage_index} current",
        rows_by_model=rows_by_model,
        samples=samples,
        start=args.stage_start,
        end=args.stage_end,
        ordered_models=ordered,
        cases_per_type=args.cases_per_type,
    )
    cumulative_rows = write_one_report(
        stage_dir / "cumulative",
        label=f"stage {args.stage_index} cumulative",
        rows_by_model=rows_by_model,
        samples=samples,
        start=args.cumulative_start,
        end=args.cumulative_end,
        ordered_models=ordered,
        cases_per_type=args.cases_per_type,
    )
    write_json(
        stage_dir / "stage_summary.json",
        {
            "stage_index": args.stage_index,
            "stage_range": {"start": args.stage_start, "end": args.stage_end},
            "cumulative_range": {
                "start": args.cumulative_start,
                "end": args.cumulative_end,
            },
            "current": current_rows,
            "cumulative": cumulative_rows,
            "created_at": now_iso(),
        },
    )
    print("[staged_eval] cumulative comparison")
    print(markdown_table(cumulative_rows))
    print(f"[staged_eval] wrote stage reports: {stage_dir}")


def load_manifest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def write_stage_event(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    manifest.setdefault("schema_version", 1)
    manifest.update(
        {
            "run_id": args.run_id,
            "stage_size": args.stage_size,
            "start_index": args.start_index,
            "target_total": args.target_total,
            "total_test": args.total_test,
            "updated_at": now_iso(),
        }
    )
    manifest.setdefault("events", [])
    manifest.setdefault("models", {})
    manifest.setdefault("stages", {})
    event = {
        "time": now_iso(),
        "event": args.event,
        "phase": args.phase,
        "status": args.status,
        "stage_index": args.stage_index,
        "stage_start": args.stage_start,
        "stage_end": args.stage_end,
        "models": args.models,
        "log": args.log,
    }
    manifest["events"].append(event)
    key = f"{args.stage_index:04d}"
    stage = manifest["stages"].setdefault(
        key,
        {"stage_start": args.stage_start, "stage_end": args.stage_end, "events": []},
    )
    stage["events"].append(event)
    completed = (args.event == "finish" and args.status == "success") or (
        args.event == "skip" and args.status == "reused"
    )
    if completed:
        for model in args.models:
            progress = manifest["models"].setdefault(model, {})
            progress["completed_end"] = max(int(progress.get("completed_end", 0) or 0), args.stage_end)
            progress["last_stage"] = args.stage_index
            progress["updated_at"] = now_iso()
    write_json_atomic(args.manifest, manifest)


def print_total(args: argparse.Namespace) -> None:
    print(len(load_split(args.data_dir, split=args.split)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_total = sub.add_parser("total", help="Print split size.")
    p_total.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_total.add_argument("--split", default="test")
    p_total.set_defaults(func=print_total)

    p_ebm = sub.add_parser("materialize-ebm", help="Materialize an EBM staged view.")
    p_ebm.add_argument("--source-dirs", nargs="+", type=Path, required=True)
    p_ebm.add_argument("--out-dir", type=Path, required=True)
    p_ebm.add_argument("--start-index", type=int, required=True)
    p_ebm.add_argument("--end-index", type=int, required=True)
    p_ebm.add_argument("--stage-index", type=int, default=0)
    p_ebm.add_argument("--params", default=DEFAULT_MODEL_PARAMS.get("ebm", "~1B"))
    p_ebm.set_defaults(func=materialize_ebm)

    p_report = sub.add_parser("report", help="Write current and cumulative staged reports.")
    p_report.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_report.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_ROOT / "reports")
    p_report.add_argument("--ebm-dir", type=Path, required=True)
    p_report.add_argument("--llm-root", type=Path, required=True)
    p_report.add_argument("--stage-index", type=int, required=True)
    p_report.add_argument("--stage-start", type=int, required=True)
    p_report.add_argument("--stage-end", type=int, required=True)
    p_report.add_argument("--cumulative-start", type=int, required=True)
    p_report.add_argument("--cumulative-end", type=int, required=True)
    p_report.add_argument("--cases-per-type", type=int, default=5)
    p_report.add_argument("--model-order", nargs="*", default=DEFAULT_MODEL_ORDER)
    p_report.set_defaults(func=write_stage_reports)

    p_event = sub.add_parser("stage-event", help="Append/update staged manifest state.")
    p_event.add_argument("--manifest", type=Path, required=True)
    p_event.add_argument("--run-id", required=True)
    p_event.add_argument("--stage-size", type=int, required=True)
    p_event.add_argument("--start-index", type=int, required=True)
    p_event.add_argument("--target-total", type=int, required=True)
    p_event.add_argument("--total-test", type=int, required=True)
    p_event.add_argument("--stage-index", type=int, required=True)
    p_event.add_argument("--stage-start", type=int, required=True)
    p_event.add_argument("--stage-end", type=int, required=True)
    p_event.add_argument("--event", choices=["start", "finish", "skip", "fail"], required=True)
    p_event.add_argument("--phase", required=True)
    p_event.add_argument("--status", default="")
    p_event.add_argument("--models", nargs="*", default=[])
    p_event.add_argument("--log", default="")
    p_event.set_defaults(func=write_stage_event)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
