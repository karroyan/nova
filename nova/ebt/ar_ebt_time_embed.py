#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import (
    init_whole_model_weights,
    EBTModelArgs,
    BLOCK_MODE_CHOICES,
    EXPLICIT_BLOCK_LATENT_MODES,
    build_explicit_block_latent_mask,
    build_explicit_block_latent_freq_indices,
    build_explicit_block_latent_inference_mask,
    build_explicit_block_latent_inference_freq_indices,
)


# ---------------------------------------------------------------------------
# KV cache containers (dense_token / mtp_mcmc inference only).
# Only the context portion (positions 1..S) of the EBT attention layout is
# cached. The time-embedding (position 0) varies per MCMC step and the
# predicted latents (positions S+1..S+K) vary per MCMC update, so neither is
# cached. K/V are stored AFTER RoPE has been applied, since RoPE depends on
# absolute position which is stable for context tokens across decode steps.
# ---------------------------------------------------------------------------

class KVCacheLayer:
    """Per-layer cache of context K/V (RoPE-applied)."""

    def __init__(self, max_seqlen: int, bsz: int, n_kv_heads: int, head_dim: int,
                 dtype: torch.dtype, device: torch.device):
        self.max_seqlen = max_seqlen
        self.cached_k = torch.zeros(bsz, max_seqlen, n_kv_heads, head_dim,
                                    dtype=dtype, device=device)
        self.cached_v = torch.zeros(bsz, max_seqlen, n_kv_heads, head_dim,
                                    dtype=dtype, device=device)
        self.cached_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """k_new/v_new: (B, T_new, n_kv_heads, head_dim). RoPE already applied."""
        T = k_new.shape[1]
        if self.cached_len + T > self.max_seqlen:
            raise RuntimeError(
                f"KVCacheLayer overflow: cached_len={self.cached_len}, T_new={T}, max_seqlen={self.max_seqlen}"
            )
        # detach so the cache never participates in autograd; MCMC backward
        # only flows through the freshly computed K/V each call.
        self.cached_k[:, self.cached_len:self.cached_len + T] = k_new.detach()
        self.cached_v[:, self.cached_len:self.cached_len + T] = v_new.detach()
        self.cached_len += T

    def get(self):
        return self.cached_k[:, :self.cached_len], self.cached_v[:, :self.cached_len]


class KVCache:
    """Whole-model KV cache with PER-MCMC-STEP sub-caches.

    Rationale: in the EBT time_embed layout the time-embedding token (which
    encodes the MCMC step) is prepended to the sequence and is attended by the
    context tokens. Therefore the context K/V at layers >= 1 are a function of
    the MCMC step, and a single cache cannot be reused across MCMC steps. We
    keep one independent sub-cache per MCMC step value (0 .. n_steps-1).

    All sub-caches advance their cached_len independently but in practice stay
    in lockstep (each is appended once per outer decode step).
    """

    def __init__(self, n_layers: int, n_steps: int, max_seqlen: int, bsz: int,
                 n_kv_heads: int, head_dim: int,
                 dtype: torch.dtype, device: torch.device):
        self.n_steps = max(1, int(n_steps))
        self.step_caches = [
            [KVCacheLayer(max_seqlen, bsz, n_kv_heads, head_dim, dtype, device)
             for _ in range(n_layers)]
            for _ in range(self.n_steps)
        ]

    def layers_for_step(self, step: int):
        return self.step_caches[int(step)]

    def cached_len_for_step(self, step: int) -> int:
        return self.step_caches[int(step)][0].cached_len

    @property
    def cached_len(self) -> int:
        # Representative value (sub-cache 0). All sub-caches are in lockstep at
        # outer-step boundaries, which is where this is read.
        return self.step_caches[0][0].cached_len

    def reset(self):
        for sc in self.step_caches:
            for layer in sc:
                layer.cached_len = 0

    def reset_step_to(self, step: int, target_len: int):
        """Roll back one MCMC step's sub-cache to target_len in every layer.
        Used to make the MCMC inner loop idempotent: each MCMC iteration
        rebuilds its new-ctx K/V afresh on top of the prior outer steps'
        cached prefix, without disturbing other steps' sub-caches."""
        for layer in self.step_caches[int(step)]:
            if target_len > layer.cached_len:
                raise RuntimeError(
                    f"reset_step_to(step={step}, {target_len}) cannot extend cached_len={layer.cached_len}"
                )
            layer.cached_len = target_len


def make_kv_cache(transformer, bsz: int, max_seqlen: int,
                  n_steps: Optional[int] = None,
                  dtype: torch.dtype = torch.float32,
                  device: Optional[torch.device] = None) -> KVCache:
    """Build a KVCache sized for the given EBTTimeConcat transformer.

    n_steps defaults to the transformer's number of time-embedding slots
    (= mcmc_num_steps), giving one sub-cache per MCMC step.
    """
    params = transformer.params
    n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
    head_dim = params.dim // params.n_heads
    if device is None:
        device = next(transformer.parameters()).device
    if n_steps is None:
        # EBTTimeConcat stores time_embeddings = nn.Embedding(max_mcmc_steps, dim)
        n_steps = getattr(getattr(transformer, "time_embeddings", None), "num_embeddings", 1)
    return KVCache(
        n_layers=params.n_layers,
        n_steps=n_steps,
        max_seqlen=max_seqlen,
        bsz=bsz,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )


def _resolve_block_mode(explicit: Optional[str], fallback: str) -> str:
    """Resolve a block_mode string, preferring the explicit argument.

    Any call site that owns a trained model must pass its block_mode
    explicitly. We only fall back to the model-configured default when
    no override is provided (e.g. from legacy call sites that will be
    updated progressively).
    """
    mode = explicit if explicit is not None else fallback
    if mode not in BLOCK_MODE_CHOICES:
        raise ValueError(
            f"Unknown block_mode={mode!r}; must be one of {BLOCK_MODE_CHOICES}"
        )
    return mode


class BackwardRMSNormFunction(torch.autograd.Function):
    """
    Forward: identity
    Backward: apply RMSNorm on grad_output, with its own weight parameter.
    """

    @staticmethod
    def forward(ctx, input_, weight, eps):
        # Save for backward
        ctx.save_for_backward(weight)
        ctx.eps = eps
        return input_  # Identity forward

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output -> RMSNorm(grad_output) using 'weight'.
        """
        # print("grad_output", grad_output)
        (weight,) = ctx.saved_tensors
        eps = ctx.eps

        # Compute RMS of grad_output over last dim
        rms = torch.rsqrt(grad_output.float().pow(2).mean(dim=-1, keepdim=True) + eps)

        # Normalized gradient wrt the input
        grad_input = grad_output * rms * weight  # shape matches grad_output

        # Gradient wrt 'weight'
        # The transform is: out_grad = grad_output * (rms * weight).
        # So partial wrt weight is grad_output * rms.
        # We sum over all but the last dimension.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_weight = (grad_output * rms).sum(dim=sum_dims)

        return grad_input, grad_weight, None


class BackwardRMSNorm(nn.Module):
    """
    nn.Module that applies identity in forward, RMSNorm in backward,
    with its own learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 0.01))
        # self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps

    def forward(self, x: torch.Tensor):
        return BackwardRMSNormFunction.apply(x, self.weight, self.eps)
    
class BackwardLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, gamma, beta, eps):
        # Save for backward
        ctx.save_for_backward(gamma, beta)
        ctx.eps = eps
        # Identity forward pass
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        gamma, beta = ctx.saved_tensors
        eps = ctx.eps
        
        # Compute mean and variance of grad_output over the last dimension
        mu = grad_output.mean(dim=-1, keepdim=True)
        var = grad_output.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)

        # "LayerNorm" of the incoming gradient
        normed_grad = (grad_output - mu) / std
        
        # Output gradient for the input
        #    out_grad = gamma * normed_grad + beta
        grad_input = gamma * normed_grad + beta
        
        # For the chain rule:
        # partial(out_grad) / partial(gamma) = normed_grad
        # partial(out_grad) / partial(beta)  = 1
        # Then multiply each by grad_output for the chain rule, i.e. (grad_output_from_next * partial).
        # But in this scenario, the "local" gradient transform is the final. By convention (matching the RMSNorm example),
        # we multiply normed_grad by grad_output to find derivative wrt gamma, etc. 
        # However, the code snippet below just sums across the relevant dims, 
        # assuming grad_output is the final gradient from the next operation.

        sum_dims = list(range(grad_output.dim() - 1))
        grad_gamma = (grad_output * normed_grad).sum(dim=sum_dims)
        grad_beta = grad_output.sum(dim=sum_dims)

        return grad_input, grad_gamma, grad_beta, None


class BackwardLayerNorm(nn.Module):
    """
    Identity on the forward pass, LayerNorm on the backward pass,
    with learnable gamma and beta. Normalizes over the last dimension.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # self.gamma = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.full((dim,), 0.01))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return BackwardLayerNormFunction.apply(x, self.gamma, self.beta, self.eps)


class EBMBackwardsRMSNorm(nn.Module): # NOTE none of this worked well, could make loss lower initially (i.e. equal to -log(len(vocab))) but then it diverged or converged slower
    """
    Applies:
      1) A standard (forward) RMSNorm.
      2) A backward-only RMSNorm (identity forward) with its own parameter.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Forward RMSNorm has its own weight_fwd
        self.forward_rms = RMSNorm(dim, eps)
        # Backward RMSNorm has a separate weight_bwd
        self.backward_rms = BackwardRMSNorm(dim, eps)
        # self.backward_ln = BackwardLayerNorm(dim, eps)

    def forward(self, x: torch.Tensor):
        """
        1) Apply standard RMSNorm in forward.
        2) Then apply backward-only RMSNorm (which is identity in forward).
        """
        x = self.forward_rms(x)
        x = self.backward_rms(x)
        # x = self.backward_ln(x)
        return x

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5, bias_learnable = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=bias_learnable)
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    if xq.dtype == torch.float32:
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    if xk.dtype == torch.float32:
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    else:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: EBTModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 #NOTE this is hardcoded since we are using DDP
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.register_buffer('superdiag_rows', torch.arange(args.max_seq_len - 1))
        self.register_buffer('superdiag_cols', torch.arange(2, args.max_seq_len + 1))
        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        context_len: int,
        pred_len: int,
        block_mode: str,
        kv_cache_layer: Optional["KVCacheLayer"] = None,
    ):
        """Forward pass of the attention module.

        block_mode dispatch:
          - ``dense_token`` / ``mtp_mcmc``: legacy symmetric EBT attention
            (matches ``main`` branch). Requires ``pred_len == context_len``
            as a shape invariant, not as a semantic selector. **The code
            path for these two modes is preserved verbatim from the
            previous version — do not refactor it for the new modes.**
          - ``future_latent_non_causal`` / ``blockwise``: dispatches to
            ``_forward_explicit_block_latent`` (a separate, plain SDPA
            implementation that uses the additive ``mask`` to encode
            visibility for ``S * K`` independent future latents).

        KV cache (dense_token / mtp_mcmc only):
          When ``kv_cache_layer`` is provided, the layer caches the context
          K/V (positions 1..S, RoPE applied). Two cache modes:
            * prefill (cached_len == 0 entering, n_new_ctx >= 1): runs the
              full legacy attention and writes context K/V to cache at the
              end. Output shape and math are identical to the no-cache path.
            * decode (cached_len > 0, n_new_ctx == 1, pred_len == 1): uses
              a specialized short-input path. ``x`` must be
              ``[time, c_new, z_new]`` of length 3.

        Args:
            x: Input tensor of shape ``(B, 1 + context_len + pred_len, D)``.
                For the explicit-block-latent modes the layout is instead
                ``(B, 1 + S + S * K, D)`` with ``pred_len = S * K``.
            start_pos: Starting position for rotary caching (unused here but
                kept for symmetry with other EBT variants).
            freqs_cis: Precomputed rotary frequency tensor. For the
                explicit-block-latent modes this must already be of length
                ``full_seqlen`` (the trunk gathers per-token positions).
            mask: Additive attention mask of shape
                ``(context_len + 2, context_len + 2)`` for the symmetric
                modes. For the explicit-block-latent modes this is the
                ``(full_seqlen, full_seqlen)`` mask produced by
                ``build_explicit_block_latent_mask``.
            context_len: Number of real context tokens (pre-time-embed).
            pred_len: Number of predicted-block tokens. For the explicit
                modes this equals ``context_len * block_size``.
            block_mode: Explicit attention semantic. See module docstring.
        """
        bsz, full_seqlen, _ = x.shape
        block_mode = _resolve_block_mode(block_mode, "dense_token")

        if block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            # Plain full-sequence SDPA path with an explicit (S, K) mask.
            # Lives in its own method so that mtp_mcmc / dense_token logic
            # below is *byte-identical* to the previous version.
            if kv_cache_layer is not None:
                raise NotImplementedError(
                    "KV cache is currently only supported for dense_token / mtp_mcmc."
                )
            return self._forward_explicit_block_latent(
                x=x,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")

        # Cache-aware DECODE: cache already has context, input carries only
        # [time, 1 new ctx, 1 pred]. Specialized short path.
        if kv_cache_layer is not None and kv_cache_layer.cached_len > 0:
            return self._forward_dense_decode(
                x=x,
                freqs_cis=freqs_cis,
                kv_cache_layer=kv_cache_layer,
            )

        # dense_token and mtp_mcmc share the legacy symmetric attention math.
        # pred_len == context_len is a shape invariant under these modes; it
        # is NOT used to pick between algorithms.
        if pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires a symmetric layout with "
                f"pred_len == context_len; got context_len={context_len}, "
                f"pred_len={pred_len}. Non-symmetric layouts are reserved for "
                f"the 'blockwise' mode which is not implemented yet."
            )
        if 1 + context_len + pred_len != full_seqlen:
            raise ValueError(
                f"Invalid layout: expected 1 + context_len + pred_len == "
                f"full_seqlen, got 1 + {context_len} + {pred_len} != {full_seqlen}"
            )

        context_with_time_len = context_len + 1
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)

        # Context path includes the prepended time embedding.
        xq_o = xq[:, :context_with_time_len, :, :]
        xk_o = xk[:, :context_with_time_len, :, :]
        xv_o = xv[:, :context_with_time_len, :, :]

        # Predicted block path has exactly pred_len positions.
        xq_p = xq[:, context_with_time_len:, :, :]
        xk_p = xk[:, context_with_time_len:, :, :]
        xv_p = xv[:, context_with_time_len:, :, :]

        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:context_with_time_len])
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[2:context_with_time_len + 1])

        xq_o = xq_o.transpose(1, 2)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores_o = scores_o + mask[:-1, :-1]
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, context_with_time_len, -1)

        xq_p = xq_p.transpose(1, 2)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
        temp_append = torch.zeros(
            (scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1),
            dtype=scores_p.dtype,
            device=scores_p.device,
        )
        scores_p = torch.cat((scores_p, temp_append), dim=-1)

        insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)

        seq_len_minus_1 = scores_p.shape[2]
        superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
        superdiag_cols = self.superdiag_cols[:seq_len_minus_1]

        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask

        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask

        if mask is not None:
            scores_p = scores_p + mask[2:, :]
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

        scores_p_superdiagonal = scores_p.diagonal(offset=2, dim1=2, dim2=3).clone()
        scores_p = scores_p * diagonal_removal_mask
        scores_p = scores_p[:, :, :, :-1]
        output_p = torch.matmul(scores_p, values_o)
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim=-1)
        output_p = output_p + next_pred_self_attention
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, pred_len, -1)
        output = torch.cat((output_o, output_p), dim=1)  # B, (S+1)+K, D

        # Cache write (prefill): store context K/V (positions 1..context_len,
        # RoPE applied). Skip the time embedding (index 0) which is per-MCMC.
        if kv_cache_layer is not None:
            n_to_write = context_len - kv_cache_layer.cached_len
            if n_to_write > 0:
                ctx_xk = xk_o[:, 1 + kv_cache_layer.cached_len:1 + context_len]
                ctx_xv = xv_o[:, 1 + kv_cache_layer.cached_len:1 + context_len]
                kv_cache_layer.append(ctx_xk, ctx_xv)

        return self.wo(output)

    def _forward_dense_decode(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache_layer: "KVCacheLayer",
    ):
        """Specialized dense_token / mtp_mcmc decode-mode attention.

        Layout: ``x = [time, c_new, z_new]`` of length 3. Cached context
        ``c_1..c_{cached_len}`` lives in ``kv_cache_layer``. After this call
        ``cached_len`` grows by 1.

        ``freqs_cis`` indexes positions for the new tokens in the layout
        ``[time(pos 0), c_new(pos cached_len+1), z_new(pos cached_len+2)]``
        and has length 3 (caller builds it via ``index_select``).
        """
        bsz, T_new, _ = x.shape
        if T_new != 3:
            raise NotImplementedError(
                f"_forward_dense_decode currently supports only T_new=3 "
                f"(time + 1 new ctx + 1 pred); got T_new={T_new}."
            )

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, T_new, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, T_new, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, T_new, self.n_local_kv_heads, self.head_dim)

        # Split: new_o = [time, c_new] (positions 0 and cached_len+1);
        #        new_p = [z_new] (position cached_len+2).
        new_o_xq = xq[:, :2]
        new_o_xk = xk[:, :2]
        new_o_xv = xv[:, :2]
        new_p_xq = xq[:, 2:]
        new_p_xk = xk[:, 2:]
        new_p_xv = xv[:, 2:]

        new_o_xq, new_o_xk = apply_rotary_emb(new_o_xq, new_o_xk, freqs_cis=freqs_cis[:2])
        new_p_xq, new_p_xk = apply_rotary_emb(new_p_xq, new_p_xk, freqs_cis=freqs_cis[2:3])

        # Append the new context (c_new at index 1 within new_o) to cache.
        new_ctx_xk = new_o_xk[:, 1:2]
        new_ctx_xv = new_o_xv[:, 1:2]
        kv_cache_layer.append(new_ctx_xk, new_ctx_xv)
        S_total = kv_cache_layer.cached_len  # = cached_len_pre + 1
        cached_ctx_k, cached_ctx_v = kv_cache_layer.get()  # (B, S_total, H_kv, D)

        # Build full keys_o = [time_K_new, cached_ctx_K (RoPE already applied)]
        # of length 1 + S_total. time uses the fresh K from this call.
        time_xk = new_o_xk[:, 0:1]
        time_xv = new_o_xv[:, 0:1]
        keys_o = torch.cat([time_xk, cached_ctx_k], dim=1)   # (B, 1 + S_total, H_kv, D)
        values_o = torch.cat([time_xv, cached_ctx_v], dim=1)

        # --- Context (o) path output for the new positions only. ---
        # query time (idx 0) can only attend the time key (causal). query
        # c_new (idx 1) can attend the full context+time (it's the latest).
        new_o_xq_t = new_o_xq.transpose(1, 2)            # (B, H, 2, D)
        keys_o_t = keys_o.transpose(1, 2)                # (B, H_kv, 1+S, D)
        values_o_t = values_o.transpose(1, 2)            # (B, H_kv, 1+S, D)
        scores_o = torch.matmul(new_o_xq_t, keys_o_t.transpose(2, 3)) / math.sqrt(self.head_dim)
        # additive causal mask of shape (2, 1+S_total): row 0 (time) sees only key 0; row 1 (c_new) sees all.
        causal_o = torch.full(
            (2, 1 + S_total), float("-inf"), device=x.device, dtype=scores_o.dtype
        )
        causal_o[0, 0] = 0.0
        causal_o[1, :] = 0.0
        scores_o = scores_o + causal_o
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(new_o_xq_t)
        output_o = torch.matmul(scores_o, values_o_t)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, 2, -1)

        # --- Pred (p) path: z_new attends [time, c_1..c_{S_total}] + self. ---
        new_p_xq_t = new_p_xq.transpose(1, 2)            # (B, H, 1, D)
        scores_p = torch.matmul(new_p_xq_t, keys_o_t.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Append a single placeholder column for the z_new self-attention.
        temp_append = torch.zeros(
            (scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1),
            dtype=scores_p.dtype, device=scores_p.device,
        )
        scores_p = torch.cat((scores_p, temp_append), dim=-1)  # (B, H, 1, 1+S_total+1)

        # Self-attention insertion: z_new attends itself via (q · k) at the
        # last column (the placeholder we just appended). No causal mask
        # needed: K_new=1 means there is no "future pred" to hide.
        insertion_self = (new_p_xq * new_p_xk).sum(dim=3) / math.sqrt(self.head_dim)
        insertion_self = insertion_self.to(scores_p.dtype)  # (B, 1, H)
        # place at scores_p[:, :, 0, -1]
        insertion_full = torch.zeros_like(scores_p)
        # insertion_self shape: (B, K_new=1, H); we need (B, H, 1, 1)
        insertion_full[:, :, 0, -1] = insertion_self.transpose(1, 2).squeeze(-1)
        scores_p = scores_p + insertion_full

        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(new_p_xq_t)
        # Pull the self weight out, then mix values_o for context part and values_p for self.
        self_weight = scores_p[:, :, :, -1:]  # (B, H, 1, 1)
        scores_p_ctx = scores_p[:, :, :, :-1]  # (B, H, 1, 1+S_total)
        output_p_ctx = torch.matmul(scores_p_ctx, values_o_t)  # (B, H, 1, D)
        new_p_xv_t = new_p_xv.transpose(1, 2)                  # (B, H_kv, 1, D)
        output_p = output_p_ctx + self_weight * new_p_xv_t
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, 1, -1)

        output = torch.cat((output_o, output_p), dim=1)        # (B, 3, D)
        return self.wo(output)

    def _forward_explicit_block_latent(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Plain full-sequence SDPA for the future_latent / blockwise modes.

        This is intentionally a *separate* implementation from the legacy
        symmetric path used by ``dense_token`` / ``mtp_mcmc``. It does NOT
        use the superdiagonal trick — that trick was specific to the
        symmetric "one latent per source position" layout and would not
        encode the per-(t, j) visibility rules of the new modes.

        ``mask`` must be the full ``(T, T)`` additive mask produced by
        ``build_explicit_block_latent_mask`` and ``freqs_cis`` must already
        be a ``(T, head_dim/2)`` slice with the right rotary position per
        token (built by the caller via
        ``build_explicit_block_latent_freq_indices``).
        """
        bsz, T, _ = x.shape
        if mask is None:
            raise ValueError(
                "explicit-block-latent attention requires a non-None mask "
                "(use utils.build_explicit_block_latent_mask)."
            )
        if freqs_cis.shape[0] != T:
            raise ValueError(
                "freqs_cis length mismatch for explicit-block-latent attention: "
                f"got freqs_cis length {freqs_cis.shape[0]}, expected {T}."
            )

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, T, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, T, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, T, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)            # [B, H, T, D]
        keys = xk.transpose(1, 2)          # [B, H_kv, T, D]
        values = xv.transpose(1, 2)        # [B, H_kv, T, D]
        # The legacy code in this file assumes n_kv_heads == n_heads (no
        # repeat_kv), so we follow the same assumption to keep behavior
        # consistent across modes.

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask  # additive: 0 where allowed, -inf where forbidden
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, T, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        ebt_act_func: str = "silu",
        weight_initialization_gain: float = 1.0
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # # custom dim factor multiplier
        # if ffn_dim_multiplier is not None:
        #     hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        hidden_dim = dim if ffn_dim_multiplier is None else int(dim*ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.act_func = {
            "silu": F.silu,
            "relu": F.relu,
            "gelu": F.gelu,
            "elu": F.elu
        }[ebt_act_func]
        
        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )

    def forward(self, x):
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: EBTModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            ebt_act_func=args.ebt_act_func,
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        if args.ebt_norm == "rms":
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        elif args.ebt_norm == "layer":
            self.attention_norm = nn.LayerNorm(args.dim)
            self.ffn_norm = nn.LayerNorm(args.dim)
        elif args.ebt_norm == "none":
            self.attention_norm = nn.Identity()
            self.ffn_norm = nn.Identity()
        elif args.ebt_norm == "dyt":
            self.attention_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
            self.ffn_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
        elif args.ebt_norm == "ebm_backwards_norm":
            self.attention_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps)
        else:
            raise ValueError(f"Invalid ebt_norm value: {args.ebt_norm}")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        context_len: int,
        pred_len: int,
        block_mode: str,
        kv_cache_layer: Optional["KVCacheLayer"] = None,
    ):
        """Perform a forward pass through the TransformerBlock.

        ``block_mode`` is forwarded to ``Attention.forward``.
        """
        h = x + self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cis,
            mask,
            context_len=context_len,
            pred_len=pred_len,
            block_mode=block_mode,
            kv_cache_layer=kv_cache_layer,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.

        Attributes:
            params (EBTModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False) # no learnable bias here since grad cant be computed for a final bias term in EBT
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps)
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step = 0,
        context_len = None,
        pred_len = None,
        return_pred_hidden: bool = False,
        return_context_hidden: bool = False,
        block_mode: Optional[str] = None,
        block_size: Optional[int] = None,
        kv_cache: Optional["KVCache"] = None,
    ):
        """Perform a forward pass through the Transformer model.

        Attention / mask semantics are selected explicitly via ``block_mode``.
        Shape validation (``pred_len == context_len``) is treated as a shape
        invariant for the symmetric modes, not as a semantic branch.

        For the explicit-block-latent modes (``future_latent_non_causal`` /
        ``blockwise``), this dispatches to a separate forward path
        (``_forward_explicit_block_latent``) that uses a different sequence
        layout (``[time, c_1..c_S, z_{1,1}..z_{S,K}]``) and a per-(t, j)
        attention mask. The dense_token / mtp_mcmc code below is kept
        verbatim from the previous version.
        """
        block_mode = _resolve_block_mode(block_mode, self.params.block_mode)
        if block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            return self._forward_explicit_block_latent(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=context_len,
                pred_len=pred_len,
                block_size=block_size,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")

        if context_len is None or pred_len is None:
            full_len = embeddings.shape[1]
            if full_len % 2 != 0:
                raise ValueError(
                    f"Unable to infer context_len/pred_len from odd embeddings length {full_len}; pass them explicitly."
                )
            context_len = full_len // 2
            pred_len = full_len - context_len
        if context_len + pred_len != embeddings.shape[1]:
            raise ValueError(
                f"context_len + pred_len must equal embeddings length, got {context_len}+{pred_len}!={embeddings.shape[1]}"
            )

        # Decide cache path. In decode mode the caller already trimmed input
        # to just the *new* context + new pred latents (typically 1 + 1 for
        # sequential decoding); the cached_len lives in kv_cache.
        # Per-MCMC-step: select this step's sub-cache (context K/V depends on
        # the MCMC step because context attends the time-embedding token).
        step_layers = None
        if kv_cache is not None:
            step_layers = kv_cache.layers_for_step(mcmc_step)
        is_decode = (step_layers is not None and step_layers[0].cached_len > 0)

        # Shape invariant for dense_token / mtp_mcmc (prefill / no-cache only).
        if not is_decode and pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires symmetric context/pred layout, "
                f"got context_len={context_len}, pred_len={pred_len}. "
                f"Non-symmetric block prediction will be enabled by block_mode='blockwise' "
                f"once implemented."
            )
        if is_decode and (context_len != 1 or pred_len != 1):
            raise NotImplementedError(
                "KV cache decode currently supports only n_new_ctx=1 and K_new=1 "
                f"(sequential decoding); got context_len={context_len}, pred_len={pred_len}."
            )

        _bsz = embeddings.shape[0]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device=embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1)  # (B, 1, D)
        embeddings = torch.cat((time_embeddings, embeddings), dim=1)  # (B, 1+S+K, D) or (B, 3, D) decode

        self.freqs_cis = self.freqs_cis.to(embeddings.device)

        if is_decode:
            # Effective context size after this call appends the new ctx token.
            cached_len_pre = step_layers[0].cached_len
            S_total = cached_len_pre + context_len  # = cached_len_pre + 1
            max_pos_needed = S_total + pred_len     # = S_total + 1
            required_length = max_pos_needed + 1
            if required_length > self.freqs_cis.shape[0]:
                self.freqs_cis = precompute_freqs_cis(
                    self.params.dim // self.params.n_heads,
                    required_length,
                ).to(embeddings.device)

            # RoPE positions for the *new* tokens in this call's input:
            #   index 0 (time)     -> position 0
            #   index 1 (new ctx)  -> position S_total      (= cached_len_pre + 1)
            #   index 2 (new pred) -> position S_total + 1  (= cached_len_pre + 2)
            pos_indices = torch.tensor(
                [0, S_total, S_total + 1], dtype=torch.long, device=embeddings.device
            )
            freqs_cis_new = self.freqs_cis.index_select(0, pos_indices)
            mask_for_decode = None  # decode path builds its own causal mask

            for i, layer in enumerate(self.layers):
                embeddings = layer(
                    embeddings,
                    start_pos,
                    freqs_cis_new,
                    mask_for_decode,
                    context_len=context_len,
                    pred_len=pred_len,
                    block_mode=block_mode,
                    kv_cache_layer=step_layers[i],
                )
        else:
            # Prefill / no-cache path: original code, optionally passing the
            # cache layer (Attention writes context K/V at the end).
            legacy_seqlen = context_len + 2
            required_length = start_pos + legacy_seqlen
            if required_length > self.freqs_cis.shape[0]:
                new_freqs_cis = precompute_freqs_cis(
                    self.params.dim // self.params.n_heads,
                    required_length,
                ).to(embeddings.device)
                self.freqs_cis = new_freqs_cis

            freqs_cis = self.freqs_cis[start_pos : start_pos + legacy_seqlen]

            mask = None
            if legacy_seqlen > 1:
                mask = torch.full((legacy_seqlen, legacy_seqlen), float("-inf"), device=embeddings.device)
                mask = torch.triu(mask, diagonal=1)
                mask = mask.type_as(embeddings)

            for i, layer in enumerate(self.layers):
                kv_cache_layer = step_layers[i] if step_layers is not None else None
                embeddings = layer(
                    embeddings,
                    start_pos,
                    freqs_cis,
                    mask,
                    context_len=context_len,
                    pred_len=pred_len,
                    block_mode=block_mode,
                    kv_cache_layer=kv_cache_layer,
                )
        embeddings = self.norm(embeddings)
        embeddings = embeddings[:, 1:]  # remove time embedding
        context_hidden = embeddings[:, :context_len]
        pred_hidden = embeddings[:, context_len:]
        energies = self.final_layer(embeddings)
        energies = energies[:, context_len:]
        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies

    def _forward_explicit_block_latent(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: Optional[int],
        pred_len: Optional[int],
        block_size: Optional[int],
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Dispatch between training-layout and inference-layout for the
        explicit-block-latent modes (``future_latent_non_causal`` /
        ``blockwise``).

        The two layouts are distinguished by ``pred_len``:

          * ``pred_len == context_len * block_size`` → **training layout**:
            S source positions, each with K future latents (S*K pred tokens
            in position-major order).
          * ``pred_len == block_size`` → **inference layout**:
            C context tokens followed by exactly K pred latents anchored at
            the end of context.

        Both layouts use the same attention semantics, just at different
        scales. Inference layout is mathematically equivalent to extracting
        the last source position's K latents from the training layout, but
        avoids paying O(S^2) attention cost.
        """
        if context_len is None or pred_len is None or block_size is None:
            raise ValueError(
                "EBTTimeConcat explicit-block-latent path requires "
                "context_len, pred_len and block_size to be set explicitly."
            )
        S_or_C = int(context_len)
        K = int(block_size)
        if S_or_C <= 0 or K <= 0:
            raise ValueError(
                f"context_len/block_size must be > 0; got context_len={S_or_C}, K={K}"
            )

        if int(pred_len) == S_or_C * K:
            return self._forward_explicit_block_latent_training(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=S_or_C,
                block_size=K,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        if int(pred_len) == K:
            return self._forward_explicit_block_latent_inference(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=S_or_C,
                block_size=K,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        raise ValueError(
            "explicit-block-latent path expects pred_len in "
            f"{{context_len * block_size, block_size}}; got pred_len={pred_len}, "
            f"context_len={S_or_C}, block_size={K}."
        )

    def _forward_explicit_block_latent_training(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: int,
        block_size: int,
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Training-layout forward.

        Sequence layout (1-indexed semantic, 0-indexed positions in the mask):
            ``[time, c_1, ..., c_S, z_{1,1}, ..., z_{1,K}, z_{2,1}, ...,``
            ``z_{2,K}, ..., z_{S,1}, ..., z_{S,K}]``

        Pred latents are stored in **position-major** order (outer ``t``,
        inner ``j``), matching ``build_explicit_block_latent_mask``.
        Returns energies of shape ``(B, S * K, 1)`` in the same order.
        """
        S = int(context_len)
        K = int(block_size)
        full_len = embeddings.shape[1]
        if full_len != S + S * K:
            raise ValueError(
                "explicit-block-latent (training) expects embeddings of length "
                f"S + S * K = {S + S * K}, got {full_len}."
            )

        # ------ time embedding (prepended as a single token like legacy) ------
        bsz = embeddings.shape[0]
        mcmc_step_t = torch.full(
            size=(bsz,),
            fill_value=mcmc_step,
            device=embeddings.device,
            dtype=torch.long,
        )
        time_embeddings = self.time_embeddings(mcmc_step_t).unsqueeze(dim=1)  # (B, 1, D)
        embeddings = torch.cat((time_embeddings, embeddings), dim=1)
        T_time = 1
        T = T_time + S + S * K
        assert embeddings.shape[1] == T

        # ------ rotary frequencies (gather per-token positions) ------
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        max_pos_needed = start_pos + S + K
        required_length = max_pos_needed + 1
        if required_length > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            ).to(embeddings.device)

        pos_indices = build_explicit_block_latent_freq_indices(
            context_len=S,
            block_size=K,
            time_len=T_time,
            start_pos=start_pos,
        )
        pos_index_tensor = torch.tensor(
            pos_indices, dtype=torch.long, device=embeddings.device
        )
        freqs_cis = self.freqs_cis.index_select(0, pos_index_tensor)

        mask = build_explicit_block_latent_mask(
            context_len=S,
            block_size=K,
            time_len=T_time,
            block_mode=block_mode,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                start_pos,
                freqs_cis,
                mask,
                context_len=S,
                pred_len=S * K,
                block_mode=block_mode,
            )
        embeddings = self.norm(embeddings)
        embeddings = embeddings[:, T_time:]  # drop the time token
        context_hidden = embeddings[:, :S]
        pred_hidden = embeddings[:, S:]      # [B, S*K, D] in position-major order
        energies = self.final_layer(embeddings)
        energies = energies[:, S:]           # [B, S*K, 1]

        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies

    def _forward_explicit_block_latent_inference(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: int,
        block_size: int,
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Inference-layout forward: C context + K pred latents at the end.

        Sequence layout: ``[time, c_1, ..., c_C, z_1, ..., z_K]``
        with ``z_j`` semantically equal to ``z_{C, j}`` from training. The
        attention mask (built by ``build_explicit_block_latent_inference_mask``)
        encodes the same per-mode visibility rules as training.

        Returns energies of shape ``(B, K, 1)`` for the K pred latents.
        """
        C = int(context_len)
        K = int(block_size)
        full_len = embeddings.shape[1]
        if full_len != C + K:
            raise ValueError(
                "explicit-block-latent (inference) expects embeddings of length "
                f"C + K = {C + K}, got {full_len}."
            )

        bsz = embeddings.shape[0]
        mcmc_step_t = torch.full(
            size=(bsz,),
            fill_value=mcmc_step,
            device=embeddings.device,
            dtype=torch.long,
        )
        time_embeddings = self.time_embeddings(mcmc_step_t).unsqueeze(dim=1)
        embeddings = torch.cat((time_embeddings, embeddings), dim=1)
        T_time = 1
        T = T_time + C + K
        assert embeddings.shape[1] == T

        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        max_pos_needed = start_pos + C + K
        required_length = max_pos_needed + 1
        if required_length > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            ).to(embeddings.device)

        pos_indices = build_explicit_block_latent_inference_freq_indices(
            context_len=C,
            block_size=K,
            time_len=T_time,
            start_pos=start_pos,
        )
        pos_index_tensor = torch.tensor(
            pos_indices, dtype=torch.long, device=embeddings.device
        )
        freqs_cis = self.freqs_cis.index_select(0, pos_index_tensor)

        mask = build_explicit_block_latent_inference_mask(
            context_len=C,
            block_size=K,
            time_len=T_time,
            block_mode=block_mode,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                start_pos,
                freqs_cis,
                mask,
                context_len=C,
                pred_len=K,
                block_mode=block_mode,
            )
        embeddings = self.norm(embeddings)
        embeddings = embeddings[:, T_time:]  # drop the time token
        context_hidden = embeddings[:, :C]
        pred_hidden = embeddings[:, C:]      # [B, K, D]
        energies = self.final_layer(embeddings)
        energies = energies[:, C:]           # [B, K, 1]

        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies
