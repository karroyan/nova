"""Profile the KV-cache decode path to find the remaining overhead.

Usage:
    cd nova/ebt
    CKPT_PATH=/path/to/last.ckpt python runs/profile_kvcache.py
"""
import os
import sys
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Path setup so we can import EBT modules without installing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils import EBTModelArgs
from ar_ebt_time_embed import EBTTimeConcat, make_kv_cache


def main():
    # Use the same shape ballpark as a real d26 NLP model
    args = EBTModelArgs(
        dim=1664, n_layers=26, n_heads=13, n_kv_heads=13,
        ffn_dim_multiplier=4.0,
        max_batch_size=8, max_seq_len=320,
        ebt_norm="rms", ebt_act_func="silu",
        weight_initialization="xavier",
        block_mode="dense_token",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EBTTimeConcat(args, max_mcmc_steps=2).to(device).to(torch.float32).eval()
    print(f"device={device}, layers={args.n_layers}, dim={args.dim}, n_heads={args.n_heads}")

    B = 4
    P = 64                # prompt length
    GEN = 64              # decode steps
    print(f"batch={B}, prompt={P}, generate={GEN}")

    # Random "tokens" embedded outside the model (we feed the trunk directly here)
    ctx_full = torch.randn(B, P + GEN, args.dim, device=device)
    pred_full = torch.randn(B, P + GEN, args.dim, device=device)

    # ----- WARMUP -----
    print("warmup...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(
                torch.cat([ctx_full[:, :P], pred_full[:, :P]], dim=1),
                start_pos=0, mcmc_step=0,
                context_len=P, pred_len=P, block_mode="dense_token",
            )
    if device == "cuda":
        torch.cuda.synchronize()

    # ----- A: NO-CACHE -----
    print("\n--- no-cache ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        for cur_pos in range(P, P + GEN):
            _ = model(
                torch.cat([ctx_full[:, :cur_pos], pred_full[:, :cur_pos]], dim=1),
                start_pos=0, mcmc_step=0,
                context_len=cur_pos, pred_len=cur_pos, block_mode="dense_token",
            )
    if device == "cuda":
        torch.cuda.synchronize()
    no_cache_time = time.perf_counter() - t0
    print(f"no-cache total: {no_cache_time:.3f}s, {no_cache_time/GEN*1000:.1f}ms/token")

    # ----- B: CACHE -----
    print("\n--- cache ---")
    cache = make_kv_cache(model, bsz=B, max_seqlen=P + GEN + 4,
                          dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        # prefill
        _ = model(
            torch.cat([ctx_full[:, :P], pred_full[:, :P]], dim=1),
            start_pos=0, mcmc_step=0,
            context_len=P, pred_len=P, block_mode="dense_token",
            kv_cache=cache,
        )
        # decode
        for cur_pos in range(P, P + GEN - 1):
            _ = model(
                torch.cat([ctx_full[:, cur_pos:cur_pos+1], pred_full[:, cur_pos:cur_pos+1]], dim=1),
                start_pos=0, mcmc_step=0,
                context_len=1, pred_len=1, block_mode="dense_token",
                kv_cache=cache,
            )
    if device == "cuda":
        torch.cuda.synchronize()
    cache_time = time.perf_counter() - t0
    print(f"cache total: {cache_time:.3f}s, {cache_time/GEN*1000:.1f}ms/token")
    print(f"\nspeedup: {no_cache_time/cache_time:.2f}x")

    # ----- Profile cache path -----
    print("\n--- profiling cache path (top 20 by self CUDA time) ---")
    cache.reset()
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        with torch.no_grad():
            _ = model(
                torch.cat([ctx_full[:, :P], pred_full[:, :P]], dim=1),
                start_pos=0, mcmc_step=0,
                context_len=P, pred_len=P, block_mode="dense_token",
                kv_cache=cache,
            )
            for cur_pos in range(P, P + min(16, GEN - 1)):
                _ = model(
                    torch.cat([ctx_full[:, cur_pos:cur_pos+1], pred_full[:, cur_pos:cur_pos+1]], dim=1),
                    start_pos=0, mcmc_step=0,
                    context_len=1, pred_len=1, block_mode="dense_token",
                    kv_cache=cache,
                )
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
