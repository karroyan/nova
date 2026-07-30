"""
Sudoku 数据下载与预处理脚本 V2
================================

与 V1 的区别：
  - **训练集 = RRN**（sudoku-hard/{train,valid,test}.csv 合并，~180k 样本，17-34 hints 困难题）
  - **测试集 = SATNet**（~10000 样本，~27 hints 简单题，作为 OOD 测试）
  - 输出缓存目录独立：sudoku_cache_v2/

数据源:
  - SATNet: https://powei.tw/sudoku.zip (one-hot .pt)
  - RRN:    https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1 (CSV 81位数字串)

输出:
  sudoku_cache_v2/rrn_train.pt    -- RRN train+valid 合并（训练集）
  sudoku_cache_v2/rrn_val.pt      -- RRN test 的 10% 切分（验证集）
  sudoku_cache_v2/satnet_test.pt  -- SATNet 全量（OOD 测试集）

用法:
  python prepare_sudoku_data_v2.py [--data_dir PATH]
"""

import argparse
import csv
import os
import urllib.request
import zipfile

import torch


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "sudoku_cache_v2")

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

def load_satnet_all(data_dir):
    """Load SATNet sudoku data (features.pt + labels.pt), return all samples."""
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

    solution_digits = labels.argmax(dim=3) + 1  # [N, 9, 9], values 1-9
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
    """Parse RRN CSV file. Each row: (question, answer) as 81-char digit strings."""
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


def load_rrn_all_splits(data_dir):
    """Load all RRN splits. Returns (train+valid merged, test) as two lists."""
    base = os.path.join(data_dir, "sudoku-hard")
    train_csv = os.path.join(base, "train.csv")
    valid_csv = os.path.join(base, "valid.csv")
    test_csv = os.path.join(base, "test.csv")
    for p in (train_csv, valid_csv, test_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"RRN CSV not found at {p}. "
                f"Expected 'sudoku-hard/' subdirectory inside {data_dir}."
            )

    print(f"[RRN] Loading train set from {train_csv}")
    train_samples = parse_rrn_csv(train_csv)
    print(f"[RRN] Loaded {len(train_samples)} train samples")

    print(f"[RRN] Loading valid set from {valid_csv}")
    valid_samples = parse_rrn_csv(valid_csv)
    print(f"[RRN] Loaded {len(valid_samples)} valid samples")

    print(f"[RRN] Loading test set from {test_csv}")
    test_samples = parse_rrn_csv(test_csv)
    print(f"[RRN] Loaded {len(test_samples)} test samples")

    merged_train = train_samples + valid_samples
    return merged_train, test_samples


# ── Main ────────────────────────────────────────────────────────────────────

def prepare_sudoku_data_v2(data_dir):
    """Download, preprocess, and cache all sudoku data for V2 layout."""
    os.makedirs(data_dir, exist_ok=True)

    rrn_train_path = os.path.join(data_dir, "rrn_train.pt")
    rrn_val_path = os.path.join(data_dir, "rrn_val.pt")
    satnet_test_path = os.path.join(data_dir, "satnet_test.pt")

    if all(os.path.exists(p) for p in [rrn_train_path, rrn_val_path, satnet_test_path]):
        print(f"All V2 cache files already exist in {data_dir}, skipping.")
        print(f"  rrn_train.pt:    {os.path.getsize(rrn_train_path)} bytes")
        print(f"  rrn_val.pt:      {os.path.getsize(rrn_val_path)} bytes")
        print(f"  satnet_test.pt:  {os.path.getsize(satnet_test_path)} bytes")
        return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── RRN as TRAIN/VAL ──
    print("\n" + "=" * 60)
    print("  Processing RRN dataset (training source in v2)")
    print("=" * 60)
    download_and_extract(RRN_URL, raw_dir, desc="RRN")
    rrn_train_merged, rrn_test = load_rrn_all_splits(raw_dir)

    # 训练集 = RRN train+valid 合并
    # 验证集 = RRN test 的 10% 切分
    n_test = len(rrn_test)
    n_val = max(1, int(n_test * 0.1))
    rrn_val = rrn_test[:n_val]
    # 注意：剩余的 RRN test 样本并入训练集（v2 要求 SATNet 作为 OOD 测试集）
    rrn_train_final = rrn_train_merged + rrn_test[n_val:]

    print(f"[RRN] v2 split: {len(rrn_train_final)} train (merged), {len(rrn_val)} val")

    torch.save(rrn_train_final, rrn_train_path)
    torch.save(rrn_val, rrn_val_path)
    print(f"[RRN] Saved {rrn_train_path}")
    print(f"[RRN] Saved {rrn_val_path}")

    # ── SATNet as TEST ──
    print("\n" + "=" * 60)
    print("  Processing SATNet dataset (OOD test source in v2)")
    print("=" * 60)
    download_and_extract(SATNET_URL, raw_dir, desc="SATNet")
    satnet_test = load_satnet_all(raw_dir)

    torch.save(satnet_test, satnet_test_path)
    print(f"[SATNet] Saved {satnet_test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary (V2)")
    print("=" * 60)
    print(f"  rrn_train.pt:    {len(rrn_train_final)} samples (training)")
    print(f"  rrn_val.pt:      {len(rrn_val)} samples (validation)")
    print(f"  satnet_test.pt:  {len(satnet_test)} samples (OOD test)")
    print(f"  Cache dir:       {data_dir}")

    sample = rrn_train_final[0]
    print(f"\n  Sample puzzle (first row):   {sample['puzzle'][0]}")
    print(f"  Sample solution (first row): {sample['solution'][0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Sudoku datasets (V2)")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SUDOKU_DATA_DIR_V2", DEFAULT_DATA_DIR),
        help="Directory to store cached V2 data "
             "(default: SUDOKU_DATA_DIR_V2 env or openebm/elm/data/sudoku_cache_v2/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-process even if cache exists",
    )
    args = parser.parse_args()

    if args.force:
        for fname in ["rrn_train.pt", "rrn_val.pt", "satnet_test.pt"]:
            path = os.path.join(args.data_dir, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed {path}")

    prepare_sudoku_data_v2(args.data_dir)
