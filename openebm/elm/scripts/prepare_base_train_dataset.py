#!/usr/bin/env python3
"""
Download a token-budgeted local base pretraining corpus for OpenEBM.

The output format intentionally matches nanochat's FineWeb parquet layout:
  <repo>/data/<dataset>/shard_00000.parquet
  <repo>/data/<dataset>/shard_00001.parquet
  ...

Each parquet file contains a single "text" column. The existing nanochat
dataloader uses all sorted shards except the last for train, and the last shard
for validation.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_TOKENS = 7_340_032_000
DEFAULT_MARGIN = 0.05
TEXT_COLUMNS = ("text", "content", "raw_content", "document", "doc")


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    patterns: tuple[str, ...]
    tokenized_gpt2: bool = False


DATASET_REPOS: dict[str, tuple[RepoSpec, ...]] = {
    "climbmix": (
        RepoSpec(
            repo_id="OptimalScale/ClimbMix",
            patterns=("*.parquet", "*.jsonl", "*.jsonl.gz", "*.jsonl.zst", "*.jsonl.zstd"),
        ),
        RepoSpec(
            repo_id="nvidia/Nemotron-ClimbMix",
            patterns=("*.parquet", "*.jsonl", "*.jsonl.gz", "*.jsonl.zst", "*.jsonl.zstd"),
        ),
        RepoSpec(
            repo_id="nvidia/Nemotron-ClimbMix",
            patterns=("*tokenized*.jsonl", "*tokenized*.jsonl.gz", "*tokenized*.jsonl.zst", "*tokenized*.jsonl.zstd"),
            tokenized_gpt2=True,
        ),
    ),
    "dclm": (
        RepoSpec(
            repo_id="mlfoundations/dclm-baseline-1.0-parquet",
            patterns=("*.parquet",),
        ),
        RepoSpec(
            repo_id="mlfoundations/dclm-baseline-1.0",
            patterns=("*.jsonl.zst", "*.jsonl.zstd", "*.jsonl.gz", "*.jsonl"),
        ),
    ),
}


class ParquetShardWriter:
    def __init__(self, out_dir: Path, tokens_per_shard: int, row_group_size: int, compression: str):
        self.out_dir = out_dir
        self.tokens_per_shard = tokens_per_shard
        self.row_group_size = row_group_size
        self.compression = compression
        self.schema = pa.schema([("text", pa.string())])
        self.writer: pq.ParquetWriter | None = None
        self.shard_index = 0
        self.shard_tokens = 0
        self.total_docs = 0
        self.shard_token_counts: list[int] = []

    def _open(self) -> None:
        if self.writer is not None:
            return
        path = self.out_dir / f"shard_{self.shard_index:05d}.parquet"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.writer = pq.ParquetWriter(tmp_path, self.schema, compression=self.compression)

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        tmp_path = self.out_dir / f"shard_{self.shard_index:05d}.parquet.tmp"
        path = self.out_dir / f"shard_{self.shard_index:05d}.parquet"
        tmp_path.replace(path)
        self.shard_token_counts.append(self.shard_tokens)
        self.writer = None
        self.shard_index += 1
        self.shard_tokens = 0

    def write(self, rows: list[str], token_count: int) -> None:
        if not rows:
            return
        self._open()
        assert self.writer is not None
        table = pa.table({"text": rows}, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.shard_tokens += token_count
        self.total_docs += len(rows)
        if self.shard_tokens >= self.tokens_per_shard:
            self._close()

    def close(self) -> None:
        self._close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=("climbmix", "dclm"), default=["climbmix", "dclm"])
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--tokens-per-output-shard", type=int, default=20_000_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--keep-sources", action="store_true")
    parser.add_argument("--allow-tokenized-climbmix", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-input-files", type=int, default=0, help="0 means no limit; useful for smoke tests")
    return parser.parse_args()


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def natural_key(path: str) -> list[object]:
    parts: list[object] = []
    buf = ""
    for ch in path:
        if ch.isdigit():
            buf += ch
        else:
            if buf:
                parts.append(int(buf))
                buf = ""
            parts.append(ch)
    if buf:
        parts.append(int(buf))
    return parts


def bytes_in_tree(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def check_free_space(path: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"Free space under {path} is {free_gb:.2f} GiB, below --min-free-gb={min_free_gb}")


def load_tokenizer(repo_root: Path):
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(repo_root / "data"))
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    log(f"Loaded tokenizer from {os.environ['NANOCHAT_BASE_DIR']}/tokenizer (bos={bos})")
    return tokenizer, bos


def matching_files(api: HfApi, spec: RepoSpec) -> list[str]:
    log(f"Listing dataset repo {spec.repo_id}")
    files = api.list_repo_files(spec.repo_id, repo_type="dataset")
    matched = [
        name
        for name in files
        if not name.endswith("/")
        and any(fnmatch.fnmatch(name, pattern) for pattern in spec.patterns)
        and not Path(name).name.startswith(".")
    ]
    matched = sorted(set(matched), key=natural_key)
    log(f"Matched {len(matched)} candidate file(s) in {spec.repo_id}")
    return matched


def download_file(spec: RepoSpec, remote_path: str, local_dir: Path, hf_cache_dir: Path) -> Path:
    repo_local_dir = local_dir / spec.repo_id.replace("/", "__")
    repo_local_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {spec.repo_id}:{remote_path}")
    try:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=repo_local_dir,
            cache_dir=hf_cache_dir,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=repo_local_dir,
            cache_dir=hf_cache_dir,
        )
    return Path(path)


def open_text_lines(path: Path) -> Iterator[str]:
    name = path.name
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif name.endswith(".zst") or name.endswith(".zstd"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading zstd jsonl requires the Python package 'zstandard'. "
                "Prefer the parquet mirror or install zstandard on the cpu-worker."
            ) from exc
        with path.open("rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            text_stream = getattr(reader, "read", None)
            if text_stream is None:
                raise RuntimeError("Could not create zstd stream reader")
            import io

            with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as handle:
                yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def extract_json_text(obj: object) -> str | None:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in TEXT_COLUMNS:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def detokenize_gpt2_json(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    token_ids = None
    for key in ("tokens", "token_ids", "input_ids", "ids"):
        value = obj.get(key)
        if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
            token_ids = value
            break
    if token_ids is None:
        return extract_json_text(obj)
    import tiktoken

    return tiktoken.get_encoding("gpt2").decode(token_ids)


def iter_jsonl_texts(path: Path, tokenized_gpt2: bool) -> Iterator[str]:
    for line_number, line in enumerate(open_text_lines(path), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if line_number <= 5:
                log(f"Skipping invalid JSON in {path.name}:{line_number}")
            continue
        text = detokenize_gpt2_json(obj) if tokenized_gpt2 else extract_json_text(obj)
        if text:
            yield text


def parquet_text_column(path: Path) -> str:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    for column in TEXT_COLUMNS:
        if column in names:
            return column
    raise RuntimeError(f"No text-like column found in {path}; columns={names}")


def iter_parquet_texts(path: Path) -> Iterator[str]:
    pf = pq.ParquetFile(path)
    column = parquet_text_column(path)
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[column])
        for value in table.column(column).to_pylist():
            if isinstance(value, str) and value:
                yield value


def iter_texts(path: Path, tokenized_gpt2: bool) -> Iterator[str]:
    if path.name.endswith(".parquet"):
        yield from iter_parquet_texts(path)
    else:
        yield from iter_jsonl_texts(path, tokenized_gpt2=tokenized_gpt2)


def maybe_remove_source(path: Path, sources_dir: Path, keep_sources: bool) -> None:
    if keep_sources:
        return
    try:
        path.relative_to(sources_dir)
    except ValueError:
        return
    if path.exists() and path.is_file():
        path.unlink()


def write_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = out_dir / "manifest.json.tmp"
    final = out_dir / "manifest.json"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(final)


def prepare_dataset(args: argparse.Namespace, dataset_name: str, tokenizer, bos: int) -> None:
    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data"
    out_dir = data_root / dataset_name
    sources_dir = out_dir / "_sources"
    hf_cache_dir = args.hf_cache_dir or (data_root / "hf_cache")
    target_with_margin = int(args.target_tokens * (1.0 + args.margin))

    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and int(manifest.get("total_tokens", 0)) >= args.target_tokens:
            log(f"{dataset_name}: existing complete manifest has {manifest['total_tokens']:,} tokens; skipping")
            return
        raise RuntimeError(f"{dataset_name}: incomplete existing manifest found at {manifest_path}; rerun with --force")

    if args.force:
        for old_file in out_dir.glob("shard_*.parquet"):
            old_file.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    api = HfApi()
    writer = ParquetShardWriter(
        out_dir=out_dir,
        tokens_per_shard=args.tokens_per_output_shard,
        row_group_size=args.row_group_size,
        compression=args.parquet_compression,
    )
    manifest: dict = {
        "dataset": dataset_name,
        "target_tokens": args.target_tokens,
        "target_tokens_with_margin": target_with_margin,
        "total_tokens": 0,
        "total_docs": 0,
        "complete": False,
        "files": [],
        "output_dir": str(out_dir),
    }

    total_tokens = 0
    input_files_seen = 0
    selected_specs = DATASET_REPOS[dataset_name]
    try:
        for spec in selected_specs:
            if spec.tokenized_gpt2 and not args.allow_tokenized_climbmix:
                log(f"Skipping tokenized fallback {spec.repo_id}; pass --allow-tokenized-climbmix to enable it")
                continue
            remote_files = matching_files(api, spec)
            if not remote_files:
                continue
            for remote_path in remote_files:
                if args.max_input_files and input_files_seen >= args.max_input_files:
                    log(f"{dataset_name}: stopping early because --max-input-files={args.max_input_files}")
                    break
                check_free_space(data_root, args.min_free_gb)
                local_path = download_file(spec, remote_path, sources_dir, hf_cache_dir)
                input_files_seen += 1

                file_tokens = 0
                file_docs = 0
                pending_rows: list[str] = []
                pending_tokens = 0
                for text in iter_texts(local_path, tokenized_gpt2=spec.tokenized_gpt2):
                    token_count = len(tokenizer.encode(text, prepend=bos))
                    if token_count <= 1:
                        continue
                    pending_rows.append(text)
                    pending_tokens += token_count
                    file_tokens += token_count
                    file_docs += 1
                    total_tokens += token_count
                    if len(pending_rows) >= args.row_group_size:
                        writer.write(pending_rows, pending_tokens)
                        pending_rows = []
                        pending_tokens = 0
                    if total_tokens >= target_with_margin:
                        break
                writer.write(pending_rows, pending_tokens)
                maybe_remove_source(local_path, sources_dir, keep_sources=args.keep_sources)

                manifest["files"].append(
                    {
                        "repo_id": spec.repo_id,
                        "path": remote_path,
                        "docs": file_docs,
                        "tokens": file_tokens,
                    }
                )
                manifest["total_tokens"] = total_tokens
                manifest["total_docs"] = writer.total_docs
                manifest["shards_written"] = writer.shard_index + (1 if writer.writer is not None else 0)
                write_manifest(out_dir, manifest)
                log(
                    f"{dataset_name}: processed {spec.repo_id}:{remote_path} "
                    f"docs={file_docs:,} file_tokens={file_tokens:,} total_tokens={total_tokens:,}"
                )
                if total_tokens >= target_with_margin:
                    break
            if total_tokens >= target_with_margin:
                break
        writer.close()
    except Exception:
        writer.close()
        raise

    if total_tokens < args.target_tokens:
        raise RuntimeError(
            f"{dataset_name}: stopped after {total_tokens:,} tokens, below target {args.target_tokens:,}. "
            "Check repo availability or increase --max-input-files."
        )

    manifest["total_tokens"] = total_tokens
    manifest["total_docs"] = writer.total_docs
    manifest["shards_written"] = writer.shard_index
    manifest["shard_token_counts"] = writer.shard_token_counts
    manifest["complete"] = True
    manifest["disk_bytes"] = bytes_in_tree(out_dir)
    write_manifest(out_dir, manifest)
    log(
        f"{dataset_name}: complete tokens={total_tokens:,} docs={writer.total_docs:,} "
        f"shards={writer.shard_index:,} disk={format_gib(manifest['disk_bytes'])}"
    )


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    os.environ.setdefault("HF_HOME", str(args.repo_root / "data" / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.repo_root / "data" / "hf_datasets_cache"))
    log(f"repo_root={args.repo_root}")
    log(f"datasets={args.datasets}")
    log(f"target_tokens={args.target_tokens:,}; margin={args.margin:.2%}")
    log(f"effective_stop_tokens={int(args.target_tokens * (1.0 + args.margin)):,}")
    tokenizer, bos = load_tokenizer(args.repo_root)
    for dataset_name in args.datasets:
        prepare_dataset(args, dataset_name, tokenizer, bos)
    data_root = args.repo_root / "data"
    log(f"Final data dir size: {format_gib(bytes_in_tree(data_root))} at {data_root}")


if __name__ == "__main__":
    main()
