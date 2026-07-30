"""Inspect random samples from Sudoku v2 datasets (RRN train / SATNet test).

Usage:
    python openebm/elm/data/inspect_sudoku_v2_samples.py
    python openebm/elm/data/inspect_sudoku_v2_samples.py --n 5 --split satnet_test
    python openebm/elm/data/inspect_sudoku_v2_samples.py --data_dir /path/to/sudoku_cache_v2
"""

import argparse
import os
import random

import torch


DEFAULT_DATA_DIR = os.environ.get(
    "SUDOKU_DATA_DIR_V2",
    "/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2",
)

SPLIT_TO_FILE = {
    "rrn_train": "rrn_train.pt",
    "rrn_val": "rrn_val.pt",
    "satnet_test": "satnet_test.pt",
}


def render_grid(board):
    """Render a 9x9 board (list of list of int, 0=empty) as ASCII."""
    lines = []
    for r in range(9):
        if r % 3 == 0 and r > 0:
            lines.append("------+-------+------")
        cells = []
        for c in range(9):
            if c % 3 == 0 and c > 0:
                cells.append("|")
            v = board[r][c]
            cells.append(str(v) if v != 0 else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def count_hints(puzzle):
    return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--split", type=str, default="all",
                    choices=["all", "rrn_train", "rrn_val", "satnet_test"])
    ap.add_argument("--n", type=int, default=3, help="#samples per split")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = list(SPLIT_TO_FILE.keys()) if args.split == "all" else [args.split]

    for split in splits:
        path = os.path.join(args.data_dir, SPLIT_TO_FILE[split])
        print("=" * 70)
        print(f"Split: {split}    File: {path}")
        if not os.path.exists(path):
            print("  [missing]")
            continue
        data = torch.load(path, weights_only=False)
        print(f"  total samples: {len(data)}")
        if len(data) == 0:
            continue

        idxs = rng.sample(range(len(data)), k=min(args.n, len(data)))
        for i, idx in enumerate(idxs):
            sample = data[idx]
            puzzle = sample["puzzle"]
            solution = sample["solution"]
            source = sample.get("source", "?")
            n_hints = count_hints(puzzle)
            print("\n" + "-" * 70)
            print(f"  [{split}] sample #{i+1} (idx={idx}, source={source}, hints={n_hints})")
            print("  Puzzle:")
            for line in render_grid(puzzle).split("\n"):
                print("    " + line)
            print("  Solution:")
            for line in render_grid(solution).split("\n"):
                print("    " + line)
        print()


if __name__ == "__main__":
    main()
