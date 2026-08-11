#!/bin/bash
################################################################################
# KV Cache vs No-Cache 对拍脚本
#
# 用同一 checkpoint、同一 seed、同一确定性采样 (temp=0) 各跑一次 eval，然后
# 比较：
#   1. 生成文本是否一致（temp=0 时应该 100% 一致）
#   2. PPL/BPB 是否一致（teacher-forced 路径不走 cache，本就该一致）
#   3. 每条样本的 decode_time_sec 加速比
#
# 使用:
#   CKPT_PATH=/path/to/ckpt.ckpt EVAL_TASK=gsm8k bash runs/eval_ebt_kvcache_compare.sh
#
# 环境变量沿用 eval_ebt.sh 的所有 (CKPT_PATH, EVAL_TASK, BATCH_SIZE, ...)
# 额外可调:
#   COMPARE_TEMP        默认 0 (deterministic，文本必须完全一致)
#   COMPARE_LIMIT       默认 5  (limit_test_batches，加速对拍)
#   COMPARE_MAX_GEN     默认 64 (infer_max_gen_len)
################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

if [ -z "${CKPT_PATH:-}" ]; then
    echo "❌ 必须设置 CKPT_PATH"
    echo "用法: CKPT_PATH=/path/to/ckpt.ckpt EVAL_TASK=gsm8k bash $0"
    exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint 不存在: $CKPT_PATH"
    exit 1
fi

export EVAL_TASK="${EVAL_TASK:-gsm8k}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LIMIT_TEST_BATCHES="${COMPARE_LIMIT:-5}"
export INFER_TEMP="${COMPARE_TEMP:-0}"
export INFER_TOPP="${INFER_TOPP:-1.0}"
export INFER_MAX_GEN_LEN="${COMPARE_MAX_GEN:-64}"
export INFER_BLOCK_SIZE=1   # cache 仅支持 sequential
export INFER_BLOCK_USE_REFINE=false
export INFER_BLOCK_REFINE_STEPS=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="$EBT_DIR/logs/kvcache_compare/${EVAL_TASK}_${TIMESTAMP}"
mkdir -p "$BASE_OUT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "KV Cache 对拍"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task: $EVAL_TASK | limit: $LIMIT_TEST_BATCHES | temp: $INFER_TEMP | max_gen: $INFER_MAX_GEN_LEN"
echo "Checkpoint: $CKPT_PATH"
echo "输出根目录: $BASE_OUT"
echo ""

# ---------- 1. baseline (no cache) ----------
echo "▶️  Run A: 无 KV cache"
export INFER_USE_KV_CACHE=false
export EVAL_RUN_DIR="$BASE_OUT/nocache"
bash "$SCRIPT_DIR/eval_ebt.sh" || true

# ---------- 2. with cache ----------
echo ""
echo "▶️  Run B: 有 KV cache"
export INFER_USE_KV_CACHE=true
export EVAL_RUN_DIR="$BASE_OUT/withcache"
bash "$SCRIPT_DIR/eval_ebt.sh" || true

# ---------- 3. compare ----------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 对拍结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PY="${CONDA_PREFIX:-}/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || command -v python)"; fi

"$PY" - "$BASE_OUT/nocache" "$BASE_OUT/withcache" "$EVAL_TASK" <<'PYEOF'
import json, os, sys, glob, statistics

root_a, root_b, task = sys.argv[1:]

def find_results(root, task):
    hits = glob.glob(os.path.join(root, "**", task, "**", "results.jsonl"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(root, "**", "results.jsonl"), recursive=True)
    return hits[0] if hits else None

fa = find_results(root_a, task)
fb = find_results(root_b, task)
if not fa or not fb:
    print(f"❌ 找不到 results.jsonl: A={fa} B={fb}")
    sys.exit(1)

def load(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

A = load(fa); B = load(fb)
n = min(len(A), len(B))
print(f"对比样本数: {n}  (A={len(A)}, B={len(B)})")
print(f"A: {fa}")
print(f"B: {fb}")
print()

# 1) text equality
identical = 0
diffs = []
for i in range(n):
    ga = A[i].get("generation", "")
    gb = B[i].get("generation", "")
    if ga == gb:
        identical += 1
    else:
        diffs.append((i, ga, gb))
print(f"[文本] 完全一致: {identical}/{n}")
for i, ga, gb in diffs[:3]:
    print(f"  ❗ sample {i} differs:")
    print(f"    A: {ga[:200]!r}")
    print(f"    B: {gb[:200]!r}")

# 2) decode time
def mean_time(lst):
    ts = [x.get("decode_time_sec") for x in lst if x.get("decode_time_sec") is not None]
    return statistics.mean(ts) if ts else None

ta, tb = mean_time(A), mean_time(B)
if ta and tb:
    print(f"\n[时间] 平均 decode_time_sec  A(no-cache)={ta:.4f}s  B(cache)={tb:.4f}s  加速={ta/tb:.2f}x")
PYEOF

echo ""
echo "完整输出在: $BASE_OUT"
