import torch
import torch.nn.functional as F
from tokenizer import AutoTokenizer
import matplotlib.pyplot as plt
import io
import time
import math
# import base64
# from PIL import Image
import numpy as np
# NOTE THIS WORKS FOR PLOTTING LANDSCAPE JUST DOESNT DO PER LANDSCAPE FOR TIME EMBED MODELS
# most of this code is from https://github.com/meta-llama/llama/blob/main/llama/generation.py#L129

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def call_model_forward_decode(hparams, model, input_tokens, start_pos, bsz, kv_cache=None):
    #TODO eventually add back kv caching, for now start_pos is not supported  in baseline transformer and EBT so start_pos can only be 0
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            if kv_cache is not None:
                raise NotImplementedError("KV cache is not supported with --infer_ebt_advanced.")
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0] # dont return a list just return the final predicted logits
        else:
            forward_kwargs = dict(start_pos=0, learning=False, return_raw_logits=True)
            if kv_cache is not None:
                forward_kwargs["kv_cache"] = kv_cache
            ebt_outputs = model.forward(input_tokens, **forward_kwargs)
            logits = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else:
        if kv_cache is not None:
            raise NotImplementedError("KV cache is currently only implemented for EBT models.")
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
    return logits

def call_model_forward_ppl(hparams, model, input_tokens, start_pos, bsz):
    #TODO same issues with start pos as above
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0] # dont return a list just return the final predicted logits
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
            logits = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else:
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
        energies = None
    return logits, energies

def _get_tokenizer(hparams):
    """Get or create cached tokenizer. Avoids re-wrapping NanoChatTokenizerWrapper per batch."""
    if hasattr(hparams, '_cached_gen_tokenizer'):
        return hparams._cached_gen_tokenizer

    from nanochat_tokenizer_adapter import NanoChatTokenizerWrapper

    if hasattr(hparams, 'tokenizer_obj') and hparams.tokenizer_obj is not None:
        tokenizer = hparams.tokenizer_obj
        if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
            tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)
    elif hasattr(hparams, 'tokenizer_path'):
        tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_path, clean_up_tokenization_spaces=False)
    else:
        if isinstance(hparams.tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        else:
            tokenizer = hparams.tokenizer
            if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
                tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)

    hparams._cached_gen_tokenizer = tokenizer
    return tokenizer


def _draft_block_tokens(hparams, model, tokens, cur_pos, block_size, input_text_mask, temperature, top_p, bsz):
    draft_tokens = []
    draft_logits = []
    for offset in range(block_size):
        pos = cur_pos + offset
        input_tokens = tokens[:, :pos]
        logits = call_model_forward_decode(hparams, model, input_tokens, 0, bsz)
        next_logits = logits[:, -1]
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(next_logits, dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(input_text_mask[:, pos], tokens[:, pos], next_token)
        tokens[:, pos] = next_token
        draft_tokens.append(next_token.unsqueeze(1))
        draft_logits.append(next_logits.unsqueeze(1))
    return torch.cat(draft_tokens, dim=1), torch.cat(draft_logits, dim=1)


def _decode_block_from_logits(block_logits, temperature, top_p):
    block_size = block_logits.shape[1]
    decoded = []
    for offset in range(block_size):
        step_logits = block_logits[:, offset, :]
        if temperature > 0:
            probs = torch.softmax(step_logits / temperature, dim=-1)
            step_token = sample_top_p(probs, top_p).reshape(-1)
        else:
            step_token = torch.argmax(step_logits, dim=-1).reshape(-1)
        decoded.append(step_token.unsqueeze(1))
    return torch.cat(decoded, dim=1)


BLOCK_MODE_CHOICES = (
    "dense_token",
    "mtp_mcmc",
    "future_latent_non_causal",
    "blockwise",
)

# Block modes that use the new "K future latents per source position" semantic
# at inference time. Mirrors utils.EXPLICIT_BLOCK_LATENT_MODES; redeclared
# here to keep generate.py importable in environments where utils.py is not
# on the python path.
EXPLICIT_BLOCK_LATENT_MODES_INFER = ("future_latent_non_causal", "blockwise")


def _resolve_inference_strategy(hparams, infer_block_size=None):
    """Resolve the *inference strategy* (sequential vs direct_block vs refine).

    This is distinct from the attention-semantic ``block_mode``; see
    :func:`_resolve_attention_block_mode` for the latter.
    """
    if infer_block_size is None:
        infer_block_size = max(1, int(getattr(hparams, "infer_block_size", 1)))
    infer_block_mode = str(getattr(hparams, "infer_block_mode", "auto"))
    if infer_block_mode not in ("auto", "sequential", "direct_block"):
        raise ValueError(f"Unsupported infer_block_mode: {infer_block_mode}. Expected 'auto', 'sequential' or 'direct_block'.")
    if infer_block_mode == "auto":
        return "direct_block" if infer_block_size > 1 else "sequential"
    return infer_block_mode


def _resolve_attention_block_mode(hparams, model=None):
    """Resolve the attention-semantic ``block_mode`` for an inference run.

    Priority:
      1. Explicit ``hparams.block_mode`` set by the CLI / config.
      2. ``model._block_mode`` recorded by :class:`EBT_NLP` during construction.
      3. ``"dense_token"`` as a last-resort default matching main-branch EBT.

    The resolved value is validated against :data:`BLOCK_MODE_CHOICES`.
    Callers must treat this as the single source of truth for how the
    attention stack interprets context/pred tokens; training and inference
    must never disagree on this value.
    """
    block_mode = getattr(hparams, "block_mode", None)
    if block_mode is None and model is not None:
        block_mode = getattr(model, "_block_mode", None)
    if block_mode is None:
        block_mode = "dense_token"
    if block_mode not in BLOCK_MODE_CHOICES:
        raise ValueError(
            f"Unknown block_mode={block_mode!r}; must be one of {BLOCK_MODE_CHOICES}"
        )
    return block_mode


def _check_inference_block_mode_compat(attention_block_mode, inference_strategy, infer_block_size, infer_block_use_refine, infer_block_refine_steps):
    """Fail fast when (attention block_mode x inference_strategy) is not implemented.

    This is the single chokepoint that enforces the user-facing rule:
    "training and inference must use identical block_mode semantics". No
    silent fallback is allowed; if a combination is not yet implemented,
    we raise ``NotImplementedError`` with a pointer at the user.
    """
    if attention_block_mode not in BLOCK_MODE_CHOICES:
        raise ValueError(f"Unsupported block_mode={attention_block_mode!r}")

    if attention_block_mode in EXPLICIT_BLOCK_LATENT_MODES_INFER:
        # The new modes use a C+K trunk layout at inference time; both
        # sequential (K=1) and direct_block (K>1) are natively supported by
        # EBT_NLP.forward + ebt_refine_block_explicit_block_latent. Refine
        # is also fine: it just runs MCMC on the K latents anchored at the
        # end of the (real) context.
        return

    if attention_block_mode not in ("dense_token", "mtp_mcmc"):
        raise ValueError(f"Unsupported block_mode={attention_block_mode!r}")

    if inference_strategy == "direct_block" and infer_block_size > 1:
        # direct_block with block_size>1 means the trunk must accept
        # pred_len != context_len. Under dense_token and mtp_mcmc the
        # attention stack requires a symmetric layout (pred_len ==
        # context_len), so this combination is explicitly unsupported and
        # must NOT silently fall back to sequential. The real non-symmetric
        # direct_block path is reserved for the future 'blockwise'
        # block_mode.
        raise NotImplementedError(
            f"direct_block inference with infer_block_size={infer_block_size} is not "
            f"implemented for block_mode={attention_block_mode!r}. The non-symmetric "
            f"block attention required for direct_block is reserved for block_mode='blockwise' "
            f"(not yet implemented). Use infer_block_mode=sequential, or reduce "
            f"infer_block_size to 1, or train a blockwise-mode checkpoint."
        )

    # refine uses a symmetric pairing (real=[ctx,draft[:-1]], pred=[ctx[1:],draft])
    # internally, so it is safe under dense_token/mtp_mcmc. Nothing to check here.
    _ = (infer_block_use_refine, infer_block_refine_steps)


def generate_text(model, batch, hparams):
    tokenizer = _get_tokenizer(hparams)

    # Safe access to eos_token_id with fallback to bos_token_id (for compatibility)
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        tokenizer_pad_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        tokenizer_pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'bos_token_id'):
        tokenizer_pad_token_id = tokenizer.bos_token_id
    else:
        tokenizer_pad_token_id = 0  # Fallback to 0 (common for BOS/EOS/PAD in GPT-NeoX)
        
    questions, answers = batch

    # Handle both dict/dict-like format (from DataCollator) and tensor format
    # Normalize to ensure we can access via ['input_ids']
    if not isinstance(questions, dict) and not hasattr(questions, '__getitem__'):
        # questions is a raw tensor, wrap it
        ids = questions if isinstance(questions, torch.Tensor) else torch.tensor(questions, dtype=torch.long)
        attn_mask = (ids != tokenizer_pad_token_id).long()
        # Create dict-like wrapper
        questions = {'input_ids': ids, 'attention_mask': attn_mask}

    if not isinstance(answers, dict) and not hasattr(answers, '__getitem__'):
        # answers is a raw tensor, wrap it
        ans_ids = answers if isinstance(answers, torch.Tensor) else torch.tensor(answers, dtype=torch.long)
        answers = {'input_ids': ans_ids}

    # Now we can safely access via dict keys
    ids = questions['input_ids']
    attn_mask = questions["attention_mask"]
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo
    infer_block_size = max(1, int(getattr(hparams, "infer_block_size", 1)))
    infer_block_use_refine = bool(getattr(hparams, "infer_block_use_refine", True))
    infer_block_refine_steps = int(getattr(hparams, "infer_block_refine_steps", 0))
    infer_block_init_logit_scale = float(getattr(hparams, "infer_block_init_logit_scale", 8.0))
    effective_block_mode = _resolve_inference_strategy(hparams, infer_block_size=infer_block_size)
    # attention_block_mode is the attention semantic (dense_token / mtp_mcmc /
    # future_latent_non_causal / blockwise). It governs the trunk dispatch and
    # must match the value the checkpoint was trained with. Inference paths
    # below route strictly via this value.
    attention_block_mode = _resolve_attention_block_mode(hparams, model=model)
    _check_inference_block_mode_compat(
        attention_block_mode=attention_block_mode,
        inference_strategy=effective_block_mode,
        infer_block_size=infer_block_size,
        infer_block_use_refine=infer_block_use_refine,
        infer_block_refine_steps=infer_block_refine_steps,
    )
    if infer_block_size > 1 and logprobs:
        raise NotImplementedError("logprobs=True is not supported in block inference mode yet")
    if infer_block_size > 1 and effective_block_mode == "direct_block" and hparams.model_name != "ebt":
        raise NotImplementedError("direct_block inference mode is currently only supported for EBT models.")
    if infer_block_size > 1 and effective_block_mode == "direct_block":
        ebt_type = str(getattr(hparams, "ebt_type", "default"))
        if ebt_type not in ("default", "time_embed"):
            raise NotImplementedError(
                f"direct_block inference currently requires ebt_type in [default, time_embed]; got ebt_type={ebt_type}."
            )
    # ppl = model.forward_loss_wrapper(questions, phase="test")['perplexity'].item() # just in case want to debug model PPL

    prompt_tokens = [] #NOTE this was to fix a bug where this generation code was not working for bs > 1 due to pad_token_id being same as eos_token_id and min_prompt_len being wrong
    for row_ids, row_mask in zip(ids, attn_mask):
        seq_len = row_mask.sum().item()         # number of *real* tokens
        prompt_tokens.append(row_ids[:seq_len].tolist())
    
    params = model.transformer.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    # if max_prompt_len > hparams.context_length:
    #     over_length_prompt = max(prompt_tokens, key=len)
    #     print(f"Prompt exceeding max length ({max_prompt_len} > {hparams.context_length}):")
    #     print(tokenizer.decode(over_length_prompt))
    assert max_prompt_len <= hparams.context_length
    total_len = min(hparams.context_length, max_gen_len + max_prompt_len)
    pad_id = tokenizer_pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    block_energy_before_accum = torch.zeros(bsz, device="cuda", dtype=torch.float32)
    block_energy_after_accum = torch.zeros(bsz, device="cuda", dtype=torch.float32)
    block_energy_count = torch.zeros(bsz, device="cuda", dtype=torch.float32)
    gen_start = time.perf_counter()
    with torch.no_grad():
        if min_prompt_len == total_len:
            logits = call_model_forward_decode(hparams, model, tokens, prev_pos, bsz)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        if infer_block_size <= 1:
            # KV cache (sequential decode only; dense_token / mtp_mcmc EBT only).
            use_kv_cache = bool(
                getattr(hparams, "infer_use_kv_cache", False)
                and hparams.model_name == "ebt"
                and not getattr(hparams, "infer_ebt_advanced", False)
                and not logprobs
                and getattr(model, "_block_mode", "dense_token") in ("dense_token", "mtp_mcmc")
            )
            kv_cache = None
            if use_kv_cache:
                from ar_ebt_time_embed import make_kv_cache
                cache_dtype = next(model.parameters()).dtype
                kv_cache = make_kv_cache(
                    model.transformer, bsz=bsz, max_seqlen=total_len + 4,
                    dtype=cache_dtype, device=tokens.device,
                )
                eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))

                # PREFILL: one full forward over tokens[:, :min_prompt_len].
                # Predict tokens[:, min_prompt_len] from logits[:, -1].
                if min_prompt_len > 0:
                    prefill_tokens = tokens[:, :min_prompt_len]
                    logits = call_model_forward_decode(
                        hparams, model, prefill_tokens, prev_pos, bsz, kv_cache=kv_cache
                    )
                    cur_pos = min_prompt_len
                    if cur_pos < total_len:
                        if temperature > 0:
                            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                            next_token = sample_top_p(probs, top_p)
                        else:
                            next_token = torch.argmax(logits[:, -1], dim=-1)
                        next_token = next_token.reshape(-1)
                        next_token = torch.where(
                            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                        )
                        tokens[:, cur_pos] = next_token
                        eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_token_id)
                        prev_pos = cur_pos

                # DECODE: feed only the just-generated token; cache holds the rest.
                if not bool(eos_reached.all()):
                    for cur_pos in range(min_prompt_len + 1, total_len):
                        new_tok = tokens[:, cur_pos - 1:cur_pos]   # (B, 1)
                        logits = call_model_forward_decode(
                            hparams, model, new_tok, prev_pos, bsz, kv_cache=kv_cache
                        )
                        if temperature > 0:
                            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                            next_token = sample_top_p(probs, top_p)
                        else:
                            next_token = torch.argmax(logits[:, -1], dim=-1)
                        next_token = next_token.reshape(-1)
                        next_token = torch.where(
                            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                        )
                        tokens[:, cur_pos] = next_token
                        eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_token_id)
                        prev_pos = cur_pos
                        if all(eos_reached):
                            break
            else:
                for cur_pos in range(min_prompt_len, total_len):
                    input_tokens = tokens[:, :cur_pos] # NOTE removed prev_pos since are not using start_pos in model forward for now, TODO eventually add back
                    logits = call_model_forward_decode(hparams, model, input_tokens, prev_pos, bsz)
                    if temperature > 0:
                        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                        next_token = sample_top_p(probs, top_p)
                    else:
                        next_token = torch.argmax(logits[:, -1], dim=-1)

                    next_token = next_token.reshape(-1)
                    # only replace token if prompt has already been generated
                    next_token = torch.where(
                        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                    )
                    tokens[:, cur_pos] = next_token
                    if logprobs:
                        token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                            input=logits.transpose(1, 2),
                            target=tokens[:, prev_pos + 1 : cur_pos + 1],
                            reduction="none",
                            ignore_index=pad_id,
                        )
                    # Get EOS token ID safely (use the same logic as above)
                    eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (
                        next_token == eos_token_id
                    )
                    prev_pos = cur_pos
                    if all(eos_reached):
                        break
        else:
            cur_pos = min_prompt_len
            while cur_pos < total_len:
                block_size = min(infer_block_size, total_len - cur_pos)
                if effective_block_mode == "direct_block":
                    input_tokens = tokens[:, :cur_pos]
                    ebt_outputs = model.forward(
                        input_tokens,
                        start_pos=0,
                        learning=False,
                        return_raw_logits=True,
                        block_size=block_size,
                    )
                    draft_block_logits = ebt_outputs[0][-1]
                    if draft_block_logits.shape[1] != block_size:
                        raise RuntimeError(
                            f"direct_block logits shape mismatch: expected block dim {block_size}, got {draft_block_logits.shape[1]}"
                        )
                    draft_block_ids = _decode_block_from_logits(draft_block_logits, temperature, top_p)

                    if len(ebt_outputs[1]) > 0:
                        step0 = ebt_outputs[1][0].reshape(bsz, -1).mean(dim=1)
                        stepn = ebt_outputs[1][-1].reshape(bsz, -1).mean(dim=1)
                        block_energy_before_accum += step0
                        block_energy_after_accum += stepn
                        block_energy_count += 1.0
                else:
                    draft_block_ids, draft_block_logits = _draft_block_tokens(
                        hparams=hparams,
                        model=model,
                        tokens=tokens,
                        cur_pos=cur_pos,
                        block_size=block_size,
                        input_text_mask=input_text_mask,
                        temperature=temperature,
                        top_p=top_p,
                        bsz=bsz,
                    )

                commit_block_ids = draft_block_ids
                commit_block_logits = draft_block_logits
                if (
                    hparams.model_name == "ebt"
                    and infer_block_use_refine
                    and infer_block_refine_steps > 0
                    and block_size > 0
                ):
                    # Dispatch refine entry point on the attention-semantic
                    # block_mode so training and inference stay in sync. The
                    # explicit-block-latent modes use a C+K layout; mtp_mcmc /
                    # dense_token use the legacy symmetric prefix trick.
                    if attention_block_mode in EXPLICIT_BLOCK_LATENT_MODES_INFER:
                        if not hasattr(model, "ebt_refine_block_explicit_block_latent"):
                            raise NotImplementedError(
                                f"Model lacks ebt_refine_block_explicit_block_latent; "
                                f"refine for block_mode={attention_block_mode!r} unavailable."
                            )
                        refined_block_logits, refined_block_ids = model.ebt_refine_block_explicit_block_latent(
                            context_ids=tokens[:, :cur_pos],
                            draft_block_ids=draft_block_ids,
                            refine_steps=infer_block_refine_steps,
                            init_logit_scale=infer_block_init_logit_scale,
                            start_pos=0,
                            learning=False,
                        )
                    elif hasattr(model, "ebt_refine_block_fast"):
                        refined_block_logits, refined_block_ids = model.ebt_refine_block_fast(
                            context_ids=tokens[:, :cur_pos],
                            draft_block_ids=draft_block_ids,
                            refine_steps=infer_block_refine_steps,
                            init_logit_scale=infer_block_init_logit_scale,
                            start_pos=0,
                            learning=False,
                        )
                    else:
                        refined_block_logits = draft_block_logits
                        refined_block_ids = draft_block_ids
                    commit_block_ids = refined_block_ids
                    commit_block_logits = refined_block_logits

                eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))
                for offset in range(block_size):
                    pos = cur_pos + offset
                    next_token = commit_block_ids[:, offset].reshape(-1)
                    next_token = torch.where(input_text_mask[:, pos], tokens[:, pos], next_token)
                    tokens[:, pos] = next_token

                    if logprobs:
                        token_logprobs[:, pos] = -F.cross_entropy(
                            input=commit_block_logits[:, offset, :],
                            target=tokens[:, pos],
                            reduction="none",
                            ignore_index=pad_id,
                        )

                    eos_reached |= (~input_text_mask[:, pos]) & (next_token == eos_token_id)
                    prev_pos = pos

                if all(eos_reached):
                    break
                cur_pos += block_size
    gen_elapsed = time.perf_counter() - gen_start

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))
        if eos_token_id in toks:
            eos_idx = toks.index(eos_token_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)

    if logprobs:
        return [
            {
                "generation": tokenizer.decode(t, skip_special_tokens=True),
                "tokens": [tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
                "target": tokenizer.decode(gt_ans, skip_special_tokens=True),
                "prompt": tokenizer.decode(question, skip_special_tokens=True),
                "decode_time_sec": gen_elapsed / max(1, bsz),
                "infer_block_mode": effective_block_mode,
                "infer_block_size": infer_block_size,
            }
            for t, logprobs_i, gt_ans, question in zip(out_tokens, out_logprobs, answers['input_ids'], questions['input_ids'])
        ]
    outputs = []
    for i, (t, gt_ans, question) in enumerate(zip(out_tokens, answers['input_ids'], questions['input_ids'])):
        item = {
            "generation": tokenizer.decode(t, skip_special_tokens=True),
            "target": tokenizer.decode(gt_ans, skip_special_tokens=True),
            "prompt": tokenizer.decode(question, skip_special_tokens=True),
            "decode_time_sec": gen_elapsed / max(1, bsz),
            "infer_block_mode": effective_block_mode,
            "infer_block_size": infer_block_size,
        }
        if block_energy_count[i].item() > 0:
            item["block_energy_before_refine"] = (block_energy_before_accum[i] / block_energy_count[i]).item()
            item["block_energy_after_refine"] = (block_energy_after_accum[i] / block_energy_count[i]).item()
        outputs.append(item)
    return outputs


def get_ppl(model, batch, hparams, token_bytes=None): # computes teacher-forced metrics
    batch_size = batch['input_ids'].shape[0]
    infer_block_size = max(1, int(getattr(hparams, "infer_block_size", 1)))
    effective_block_mode = _resolve_inference_strategy(hparams, infer_block_size=infer_block_size)
    attention_block_mode = _resolve_attention_block_mode(hparams, model=model)
    _check_inference_block_mode_compat(
        attention_block_mode=attention_block_mode,
        inference_strategy=effective_block_mode,
        infer_block_size=infer_block_size,
        infer_block_use_refine=False,
        infer_block_refine_steps=0,
    )

    with torch.no_grad(): # by default no grad, although ebt will enable grad
        full_ids = batch['input_ids'].squeeze(dim=1)
        if hparams.model_name == "ebt" and effective_block_mode == "direct_block" and infer_block_size > 1:
            # Blockwise teacher-forced:
            # for each chunk starting at cur_pos, condition on real prefix full_ids[:, :cur_pos]
            # and predict the next K tokens in one shot.
            pad_token_id = model.tokenizer_pad_token_id
            if pad_token_id is None:
                pad_token_id = -100

            all_losses = []
            for cur_pos in range(1, full_ids.shape[1], infer_block_size):
                block_len = min(infer_block_size, full_ids.shape[1] - cur_pos)
                if block_len <= 0:
                    continue
                input_tokens = full_ids[:, :cur_pos]
                ebt_outputs = model.forward(
                    input_tokens,
                    start_pos=0,
                    learning=False,
                    return_raw_logits=True,
                    block_size=block_len,
                )
                block_logits = ebt_outputs[0][-1]  # [B, block_len, V]
                block_targets = full_ids[:, cur_pos:cur_pos + block_len]  # [B, block_len]
                block_losses = F.cross_entropy(
                    block_logits.reshape(-1, block_logits.shape[-1]),
                    block_targets.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="none",
                )
                all_losses.append(block_losses)

            if len(all_losses) == 0:
                per_token_loss = torch.tensor([0.0], device=full_ids.device)
            else:
                per_token_loss = torch.cat(all_losses, dim=0)
            energies = None
        else:
            input_ids = full_ids[:, :-1]
            if hparams.model_name == "ebt":
                logits, energies = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size)
            else:
                logits, _ = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size)

            next_token_indices = full_ids[:, 1:].reshape(-1) # BS * S; reshape since targets are supposed to be 1D

            # Get pad token id safely - nanochat doesn't have pad token, so use -100 (standard ignore index)
            pad_token_id = model.tokenizer_pad_token_id
            if pad_token_id is None:
                pad_token_id = -100  # Standard ignore index that won't match any actual token

            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                next_token_indices,
                ignore_index=pad_token_id,
                reduction="none",
            )
    cce_loss = per_token_loss.mean()
    perplexity = torch.exp(cce_loss).detach()

    outputs = {
        'loss': cce_loss,  # legacy
        'perplexity': perplexity,  # legacy
        'teacher_forced_loss': cce_loss,
        'teacher_forced_ppl': perplexity,
    }

    if token_bytes is not None:
        if hparams.model_name == "ebt" and effective_block_mode == "direct_block" and infer_block_size > 1:
            # Next-token targets are all tokens after BOS-equivalent first token in full_ids.
            next_token_indices = full_ids[:, 1:].reshape(-1)
            # blockwise all_losses is built in the same left-to-right order; trim to target length if needed
            if per_token_loss.numel() != next_token_indices.numel():
                min_len = min(per_token_loss.numel(), next_token_indices.numel())
                per_token_loss = per_token_loss[:min_len]
                next_token_indices = next_token_indices[:min_len]
        if token_bytes.device != next_token_indices.device:
            token_bytes = token_bytes.to(next_token_indices.device)

        if (next_token_indices.int() < 0).any():
            valid = next_token_indices >= 0
            y_safe = torch.where(valid, next_token_indices, torch.zeros_like(next_token_indices))
            num_bytes = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(next_token_indices, dtype=token_bytes.dtype),
            )
        else:
            num_bytes = token_bytes[next_token_indices]

        total_nats = (per_token_loss * (num_bytes > 0)).sum()
        total_bytes = num_bytes.sum().to(torch.int64)
        bpb = total_nats.item() / (math.log(2) * total_bytes.item()) if total_bytes.item() > 0 else float('inf')
        outputs["bpb"] = bpb  # legacy
        outputs["teacher_forced_bpb"] = bpb

    if hparams.model_name == "ebt" and energies is not None:
        energy_tensors = []
        for step_energies in zip(*[energies]):
            step_tensor = torch.stack(step_energies)
            avg_step_energy = torch.mean(step_tensor, dim=0)
            energy_tensors.append(avg_step_energy)

        # Convert energy tensors to scalars for logging
        for step_idx, energy in enumerate(energy_tensors):
            # Take mean if energy is not a scalar
            if energy.numel() > 1:
                energy_scalar = energy.mean().item()
            else:
                energy_scalar = energy.item()
            outputs[f"mcmc_step_{step_idx}_energy"] = energy_scalar

    if hparams.infer_plot_energy_landscape:
        assert hparams.model_name == "ebt", "Energy landscape plotting only works with EBT models"
        
        _, energies_list, predicted_tokens_list = model.ebt_advanced_inference(input_ids, start_pos=0, learning=False)
        
        # Create separate plots for different MCMC steps
        image_tensors = {}
        
        # Select steps to plot (first, middle, and last step)
        num_steps = len(predicted_tokens_list)
        steps_to_plot = list(range(num_steps))
        
        for step_idx in steps_to_plot:
            token_losses = []
            step_energies = []
            
            # Get target tokens (actual next tokens in sequence)
            target_tokens = batch['input_ids'].squeeze(dim=1)[:, 1:]
            
            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    # Skip padded positions
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue
                    
                    target_token = target_tokens[batch_idx, pos_idx]
                    
                    # Get predicted token distribution at this position for this step
                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]
                    
                    # Calculate cross entropy loss for this prediction vs ground truth
                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0), 
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()
                    
                    # Get energy value for this position at this step
                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()
                    
                    token_losses.append(token_loss)
                    step_energies.append(token_energy)
            
            # Create the scatter plot for this step
            plt.figure(figsize=(10, 6))
            plt.scatter(step_energies, token_losses, alpha=0.5)
            plt.xlabel('Predicted Energy')
            plt.ylabel('Ground Truth Cross-Entropy Loss')
            plt.title(f'Energy Landscape vs Ground Truth Loss (MCMC Step {step_idx})')
            
            # Add trend line if there are enough points
            if len(step_energies) > 5:
                z = np.polyfit(step_energies, token_losses, 1)
                p = np.poly1d(z)
                plt.plot(sorted(step_energies), p(sorted(step_energies)), "r--", alpha=0.8)
                
                # Add correlation coefficient
                from scipy.stats import pearsonr
                corr, _ = pearsonr(step_energies, token_losses)
                plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
            
            # Save plot to buffer and convert to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # # Convert to PIL Image and then to tensor
            # pil_img = Image.open(buf).convert('RGB')
            # # Convert PIL image to tensor (channels, height, width) with values in [0, 1]
            # img_tensor = torch.FloatTensor(np.array(pil_img)).permute(2, 0, 1) / 255.0
            
            # image_tensors[f"image_energy_landscape_step_{step_idx}"] = img_tensor
        
        # Also create a combined plot showing the evolution
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        
        for i, step_idx in enumerate(steps_to_plot):
            token_losses = []
            step_energies = []
            
            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue
                    
                    target_token = target_tokens[batch_idx, pos_idx]
                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]
                    
                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0), 
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()
                    
                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()
                    
                    token_losses.append(token_loss)
                    step_energies.append(token_energy)
            
            color_idx = i % len(colors)
            plt.scatter(step_energies, token_losses, alpha=0.5, color=colors[color_idx], 
                        label=f'Step {step_idx}')
        
        plt.xlabel('Predicted Energy')
        plt.ylabel('Ground Truth Cross-Entropy Loss')
        plt.title('Energy Landscape Evolution During MCMC Steps')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # pil_img = Image.open(buf).convert('RGB')
        # img_tensor = torch.FloatTensor(np.array(pil_img)).permute(2, 0, 1) / 255.0
        
        # image_tensors["image_energy_landscape_combined"] = img_tensor
        
        # # Add all images to outputs
        # for key, tensor in image_tensors.items():
        #     outputs[key] = tensor

    return outputs