"""
Sudoku 数据下载与预处理脚本

数据源:
  - SATNet 训练集: https://powei.tw/sudoku.zip (one-hot .pt)
  - RRN 测试集: https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1 (CSV 81位数字串)

输出: 缓存为 .pt 文件，每个包含 list of dict:
  {"puzzle": [[int]*9]*9, "solution": [[int]*9]*9, "source": str}

用法:
  python prepare_sudoku_data.py [--data_dir PATH]
"""

import argparse
import csv
import os
import sys
import tempfile
import urllib.request
import zipfile

import torch


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "sudoku_cache")

SATNET_URL = "https://powei.tw/sudoku.zip"
RRN_URL = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"


def download_and_extract(url, dest_dir, desc=""):
    """Download a zip file and extract to dest_dir. Returns extraction path."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, os.path.basename(url).split("?")[0])
    if not os.path.exists(zip_path):
        print(f"[{desc}] Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[{desc}] Saved to {zip_path}")
    else:
        print(f"[{desc}] Zip already exists: {zip_path}")

    print(f"[{desc}] Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    print(f"[{desc}] Extracted to {dest_dir}")
    return dest_dir


# ── SATNet ──────────────────────────────────────────────────────────────────

def load_satnet(data_dir):
    """
    Load SATNet sudoku data.

    features.pt: shape [N, 9, 9, 9] one-hot (given cells have a hot digit, empty cells are all-zero)
    labels.pt:   shape [N, 9, 9, 9] one-hot (complete solution, all cells filled)

    Returns list of dicts with puzzle/solution as 9x9 int lists (1-9, 0=empty).
    """
    features_path = os.path.join(data_dir, "sudoku", "features.pt")
    labels_path = os.path.join(data_dir, "sudoku", "labels.pt")

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"SATNet features.pt not found at {features_path}. "
            f"Expected 'sudoku/' subdirectory inside {data_dir}."
        )

    print(f"[SATNet] Loading features from {features_path}")
    features = torch.load(features_path, weights_only=True)  # [N, 9, 9, 9]
    print(f"[SATNet] Loading labels from {labels_path}")
    labels = torch.load(labels_path, weights_only=True)  # [N, 9, 9, 9]

    print(f"[SATNet] features shape: {features.shape}, labels shape: {labels.shape}")

    # Convert one-hot to digit values
    # features: given cells have one-hot encoding, empty cells are all-zero
    # Use argmax+1 for solution; for puzzle, check if any digit is given
    solution_digits = labels.argmax(dim=3) + 1  # [N, 9, 9], values 1-9

    # For puzzle: cells with all-zero one-hot → 0 (empty), otherwise argmax+1
    has_value = features.sum(dim=3) > 0  # [N, 9, 9] bool
    puzzle_digits = (features.argmax(dim=3) + 1) * has_value.long()  # 0 for empty

    samples = []
    N = features.shape[0]
    for i in range(N):
        samples.append({
            "puzzle": puzzle_digits[i].tolist(),
            "solution": solution_digits[i].tolist(),
            "source": "satnet",
        })

    print(f"[SATNet] Loaded {len(samples)} samples")
    return samples


# ── RRN ─────────────────────────────────────────────────────────────────────

def parse_rrn_csv(csv_path):
    """
    Parse RRN CSV file. Each row: (question, answer) as 81-char digit strings.
    question: 0 = empty cell, 1-9 = given digit
    answer: 1-9 for all cells
    """
    samples = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for question, answer in reader:
            puzzle = []
            solution = []
            for r in range(9):
                puzzle_row = []
                solution_row = []
                for c in range(9):
                    idx = r * 9 + c
                    puzzle_row.append(int(question[idx]))
                    solution_row.append(int(answer[idx]))
                puzzle.append(puzzle_row)
                solution.append(solution_row)
            samples.append({
                "puzzle": puzzle,
                "solution": solution,
                "source": "rrn",
            })
    return samples


def load_rrn(data_dir):
    """Load RRN test set from extracted CSV files."""
    test_csv = os.path.join(data_dir, "sudoku-hard", "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"RRN test.csv not found at {test_csv}. "
            f"Expected 'sudoku-hard/' subdirectory inside {data_dir}."
        )
    print(f"[RRN] Loading test set from {test_csv}")
    samples = parse_rrn_csv(test_csv)
    print(f"[RRN] Loaded {len(samples)} test samples")
    return samples


# ── Main ────────────────────────────────────────────────────────────────────

def prepare_sudoku_data(data_dir):
    """Download, preprocess, and cache all sudoku data."""
    os.makedirs(data_dir, exist_ok=True)

    # Check if already cached
    satnet_train_path = os.path.join(data_dir, "satnet_train.pt")
    satnet_val_path = os.path.join(data_dir, "satnet_val.pt")
    rrn_test_path = os.path.join(data_dir, "rrn_test.pt")

    if all(os.path.exists(p) for p in [satnet_train_path, satnet_val_path, rrn_test_path]):
        print(f"All cache files already exist in {data_dir}, skipping download.")
        print(f"  satnet_train.pt: {os.path.getsize(satnet_train_path)} bytes")
        print(f"  satnet_val.pt:   {os.path.getsize(satnet_val_path)} bytes")
        print(f"  rrn_test.pt:     {os.path.getsize(rrn_test_path)} bytes")
        return

    # Use a temp directory for raw downloads, cache dir for processed output
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── SATNet ──
    print("\n" + "=" * 60)
    print("  Processing SATNet dataset")
    print("=" * 60)
    download_and_extract(SATNET_URL, raw_dir, desc="SATNet")
    satnet_samples = load_satnet(raw_dir)

    # Split 90/10 for train/val
    n_total = len(satnet_samples)
    n_train = int(n_total * 0.9)
    satnet_train = satnet_samples[:n_train]
    satnet_val = satnet_samples[n_train:]
    print(f"[SATNet] Split: {len(satnet_train)} train, {len(satnet_val)} val")

    torch.save(satnet_train, satnet_train_path)
    torch.save(satnet_val, satnet_val_path)
    print(f"[SATNet] Saved {satnet_train_path}")
    print(f"[SATNet] Saved {satnet_val_path}")

    # ── RRN ──
    print("\n" + "=" * 60)
    print("  Processing RRN dataset")
    print("=" * 60)
    download_and_extract(RRN_URL, raw_dir, desc="RRN")
    rrn_test = load_rrn(raw_dir)

    torch.save(rrn_test, rrn_test_path)
    print(f"[RRN] Saved {rrn_test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  satnet_train.pt: {len(satnet_train)} samples")
    print(f"  satnet_val.pt:   {len(satnet_val)} samples")
    print(f"  rrn_test.pt:     {len(rrn_test)} samples")
    print(f"  Cache dir:       {data_dir}")

    # Verify a sample
    sample = satnet_train[0]
    print(f"\n  Sample puzzle (first row):    {sample['puzzle'][0]}")
    print(f"  Sample solution (first row):  {sample['solution'][0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Sudoku datasets")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SUDOKU_DATA_DIR", DEFAULT_DATA_DIR),
        help="Directory to store cached data (default: SUDOKU_DATA_DIR env or openebm/elm/data/sudoku_cache/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-process even if cache exists",
    )
    args = parser.parse_args()

    if args.force:
        import shutil
        for fname in ["satnet_train.pt", "satnet_val.pt", "rrn_test.pt"]:
            path = os.path.join(args.data_dir, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed {path}")

    prepare_sudoku_data(args.data_dir)
