"""Probe RNG / logits divergence between no-cache and cache paths on a real ckpt.

Performs ONE end-to-end forward at cur_pos = min_prompt_len + 1 via two paths
and compares the final logits at the new position.

Usage:
    cd nova/ebt
    CKPT_PATH=/path/to/last.ckpt /path/to/python runs/probe_kvcache_rng.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
# nanochat is imported by some pickled objects in older ckpts; add to path.
for _cand in (
    "/mnt/shared-storage-user/lixueyan/nar/nova_kvcache",
    "/mnt/shared-storage-user/lixueyan/nar",
    "/mnt/shared-storage-user/lixueyan/nar/nova",
):
    if os.path.isdir(os.path.join(_cand, "nanochat")) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import torch
try:
    from lightning.pytorch import seed_everything
except ModuleNotFoundError:
    try:
        from pytorch_lightning import seed_everything
    except ModuleNotFoundError:
        # Minimal fallback so the probe doesn't require lightning at all.
        import random as _random
        import numpy as _np
        def seed_everything(seed, workers=False):
            _random.seed(seed)
            _np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


class _FakeTokenizer:
    """Minimal tokenizer stub. EBT_NLP only needs get_vocab_size()."""
    def __init__(self, vocab_size):
        self._v = vocab_size
    def get_vocab_size(self):
        return self._v


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]

    class _Args:
        pass
    args = _Args()
    for k, v in hparams.items():
        setattr(args, k, v)
    args.execution_mode = "inference"

    # Build a fake tokenizer; vocab_size comes from the embeddings weight in the ckpt
    state = ckpt["state_dict"]
    emb_key = None
    for k in state.keys():
        if k.endswith("embeddings.weight"):
            emb_key = k
            break
    assert emb_key is not None, "could not find embeddings weight in ckpt"
    V = state[emb_key].shape[0]
    args.tokenizer_obj = _FakeTokenizer(V)
    print(f"vocab_size from ckpt: {V}")

    from modeling_ebt import EBT_NLP
    model = EBT_NLP(args)

    new_sd = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v
    result = model.load_state_dict(new_sd, strict=False)
    print(f"missing: {len(result.missing_keys)} (first 3): {result.missing_keys[:3]}")
    print(f"unexpected: {len(result.unexpected_keys)} (first 3): {result.unexpected_keys[:3]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return model, V


def main():
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        print("CKPT_PATH env var required")
        sys.exit(1)

    model, V = load_model(ckpt_path)
    device = next(model.parameters()).device

    B = 1
    P = 8       # short prompt
    target_pos = P + 1
    torch.manual_seed(0)
    full_tokens = torch.randint(0, V, (B, target_pos), device=device)
    print(f"\nSetup: B={B}, P={P}, predict next at pos={target_pos}, V={V}, device={device}, dtype={next(model.parameters()).dtype}")

    # ---- Make corrupt_embeddings DETERMINISTIC to remove RNG as a variable. ----
    # Both paths then start MCMC from the SAME init at every position, so any
    # remaining logit diff is purely the cache math (not random init).
    import modeling_ebt
    _orig_corrupt = model.corrupt_embeddings
    def _det_corrupt(embeddings, target_length=None):
        if target_length is None:
            target_length = embeddings.shape[1]
        # Deterministic, position-dependent init: value at position t is a
        # fixed function of t, so cache (which slices last row) and no-cache
        # (full) agree position-by-position.
        base = torch.arange(target_length, device=embeddings.device, dtype=torch.float32)
        t = base.view(1, target_length, 1).expand(embeddings.shape[0], target_length, model.vocab_size)
        return (t * 0.001)
    model.corrupt_embeddings = _det_corrupt

    # -------- PATH A: no-cache full forward (cur_pos=target_pos) --------
    with torch.no_grad():
        a_outputs = model.forward(
            full_tokens, start_pos=0, learning=False, return_raw_logits=True,
        )
    a_logits_last = a_outputs[0][-1]   # (B, target_pos, V)
    a_new = a_logits_last[:, -1, :]    # (B, V) — prediction for new pos
    print(f"\n[no-cache] last-step logits shape: {tuple(a_logits_last.shape)}")
    print(f"[no-cache] top-3 logits at new pos: {a_new[0].topk(3).values.tolist()}")
    print(f"[no-cache] argmax: {a_new.argmax(-1).item()}")

    # -------- PATH B: cache prefill + decode --------
    from ar_ebt_time_embed import make_kv_cache
    cache_dtype = next(model.parameters()).dtype
    kv_cache = make_kv_cache(
        model.transformer, bsz=B, max_seqlen=target_pos + 4,
        dtype=cache_dtype, device=device,
    )
    with torch.no_grad():
        prefill_tokens = full_tokens[:, :P]
        _ = model.forward(
            prefill_tokens, start_pos=0, learning=False,
            return_raw_logits=True, kv_cache=kv_cache,
        )
        print(f"\n[cache] after prefill cached_len = {kv_cache.cached_len}")
        new_tok = full_tokens[:, P:P+1]
        b_out = model.forward(
            new_tok, start_pos=0, learning=False,
            return_raw_logits=True, kv_cache=kv_cache,
        )
    b_logits_last = b_out[0][-1]      # (B, 1, V)
    b_new = b_logits_last[:, -1, :]   # (B, V)
    print(f"[cache] after decode cached_len = {kv_cache.cached_len}")
    print(f"[cache] last-step logits shape: {tuple(b_logits_last.shape)}")
    print(f"[cache] top-3 logits at new pos: {b_new[0].topk(3).values.tolist()}")
    print(f"[cache] argmax: {b_new.argmax(-1).item()}")

    # -------- COMPARE --------
    diff = (a_new - b_new).abs()
    print(f"\n=== logit diff at new pos (DETERMINISTIC init, RNG removed) ===")
    print(f"max |diff| : {diff.max().item():.6e}")
    print(f"mean|diff| : {diff.mean().item():.6e}")
    print(f"a_argmax = {a_new.argmax(-1).item()}, b_argmax = {b_new.argmax(-1).item()}")
    same = (a_new.argmax(-1) == b_new.argmax(-1)).all().item()
    print(f"argmax match: {same}")
    print()
    if diff.max().item() < 1e-2:
        print("=> cache math is CORRECT; the eval 0/10 is purely RNG init divergence.")
    else:
        print("=> cache math has a REAL divergence; investigate attention / cache write.")


if __name__ == "__main__":
    main()
