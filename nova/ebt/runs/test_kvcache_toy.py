"""Toy KV-cache verification — no checkpoint, no tokenizer.

Build a tiny EBTTimeConcat with random weights and verify:
  1. prefill (no cache vs with cache) produces identical energies.
  2. multi-step decode produces energies identical to full no-cache forwards.

Run from nova/ebt/ directory:
    python runs/test_kvcache_toy.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from utils import EBTModelArgs
from ar_ebt_time_embed import EBTTimeConcat, make_kv_cache


def main():
    torch.manual_seed(0)
    args = EBTModelArgs(
        dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
        ffn_dim_multiplier=2.0,
        max_batch_size=2, max_seq_len=24,
        ebt_norm="rms", ebt_act_func="silu",
        weight_initialization="xavier",
        block_mode="dense_token",
    )
    model = EBTTimeConcat(args, max_mcmc_steps=2).to(torch.float32).eval()
    B = 1
    S0 = 4   # initial prompt length
    torch.manual_seed(42)
    all_ctx = torch.randn(B, S0 + 2, args.dim)
    all_pred = torch.randn(B, S0 + 2, args.dim)

    def full_forward_at(N):
        """Run no-cache forward with first N ctx + first N pred latents."""
        with torch.no_grad():
            e = model(
                torch.cat([all_ctx[:, :N], all_pred[:, :N]], dim=1),
                start_pos=0, mcmc_step=0,
                context_len=N, pred_len=N, block_mode="dense_token",
            )
        return e[:, -1, 0]

    gt = [full_forward_at(N) for N in (S0, S0 + 1, S0 + 2)]
    print(f"GT energies at last pos for N=S0..S0+2: {[x.item() for x in gt]}")

    # Cache path
    cache = make_kv_cache(model, bsz=B, max_seqlen=24, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        e0 = model(
            torch.cat([all_ctx[:, :S0], all_pred[:, :S0]], dim=1),
            start_pos=0, mcmc_step=0,
            context_len=S0, pred_len=S0, block_mode="dense_token",
            kv_cache=cache,
        )
        print(f"after prefill: cached_len={cache.cached_len}")
        e1 = model(
            torch.cat([all_ctx[:, S0:S0+1], all_pred[:, S0:S0+1]], dim=1),
            start_pos=0, mcmc_step=0,
            context_len=1, pred_len=1, block_mode="dense_token",
            kv_cache=cache,
        )
        print(f"after decode1: cached_len={cache.cached_len}")
        e2 = model(
            torch.cat([all_ctx[:, S0+1:S0+2], all_pred[:, S0+1:S0+2]], dim=1),
            start_pos=0, mcmc_step=0,
            context_len=1, pred_len=1, block_mode="dense_token",
            kv_cache=cache,
        )
        print(f"after decode2: cached_len={cache.cached_len}")

    cache_path = [e0[:, -1, 0], e1[:, -1, 0], e2[:, -1, 0]]
    print()
    all_ok = True
    for i, (g, c) in enumerate(zip(gt, cache_path)):
        diff = (g - c).abs().item()
        tag = "PASS" if diff < 1e-4 else "FAIL"
        if tag == "FAIL":
            all_ok = False
        print(f"step {i}: diff = {diff:.2e}  {tag}")

    print("\nResult:", "ALL PASS  ✓" if all_ok else "SOME FAILED  ✗")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
