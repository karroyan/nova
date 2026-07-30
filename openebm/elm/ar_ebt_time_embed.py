#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from openebm.elm.utils import init_whole_model_weights, EBTModelArgs
try:
    from openebm.elm.ve import build_layer_ve, build_value_embeds, has_ve
except ImportError:
    has_ve = None
    build_value_embeds = None
    build_layer_ve = None


class BackwardRMSNormFunction(torch.autograd.Function):
    """
    Forward: identity
    Backward: apply RMSNorm on grad_output, with its own weight parameter.
    """

    @staticmethod
    def forward(ctx, input_, weight, eps, use_fp32_norm):
        # Save for backward
        ctx.save_for_backward(weight)
        ctx.eps = eps
        ctx.use_fp32_norm = use_fp32_norm
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
        if ctx.use_fp32_norm:
            rms = torch.rsqrt(grad_output.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        else:
            rms = torch.rsqrt(grad_output.pow(2).mean(dim=-1, keepdim=True) + eps)

        # Normalized gradient wrt the input
        grad_input = grad_output * rms * weight  # shape matches grad_output

        # Gradient wrt 'weight'
        # The transform is: out_grad = grad_output * (rms * weight).
        # So partial wrt weight is grad_output * rms.
        # We sum over all but the last dimension.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_weight = (grad_output * rms).sum(dim=sum_dims)

        return grad_input, grad_weight, None, None


class BackwardRMSNorm(nn.Module):
    """
    nn.Module that applies identity in forward, RMSNorm in backward,
    with its own learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 0.01))
        # self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps
        self.use_fp32_norm = use_fp32_norm

    def forward(self, x: torch.Tensor):
        return BackwardRMSNormFunction.apply(x, self.weight, self.eps, self.use_fp32_norm)
    
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
    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True):
        super().__init__()
        # Forward RMSNorm has its own weight_fwd
        self.forward_rms = RMSNorm(dim, eps, use_fp32_norm=use_fp32_norm)
        # Backward RMSNorm has a separate weight_bwd
        self.backward_rms = BackwardRMSNorm(dim, eps, use_fp32_norm=use_fp32_norm)
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
    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            use_fp32_norm (bool): If True, compute normalization in FP32 then cast back.
                Should be False for bf16-true to avoid FP32 graph nodes.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_fp32_norm = use_fp32_norm

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
        if self.use_fp32_norm:
            output = self._norm(x.float()).type_as(x)
        else:
            output = self._norm(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the cosine and sine frequency tensors for rotary embeddings.

    Returns a tuple of (cos, sin) tensors with shape (end, dim//2), stored as
    float32 but applied via real-valued arithmetic to avoid dtype casting in
    the forward/backward graph.

    Args:
        dim (int): Head dimension.
        end (int): Maximum sequence length.
        theta (float, optional): RoPE base. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) each of shape (end, dim//2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    return freqs.cos(), freqs.sin()  # both float32, shape (end, dim//2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor of shape (seq, head_dim//2) complex.
        x (torch.Tensor): Target tensor of shape (bs, seq, n_heads, head_dim//2) complex.

    Returns:
        torch.Tensor: Reshaped frequency tensor for broadcast.
    """
    ndim = x.ndim
    # RoPE freqs are indexed by sequence position (x.shape[1]) and rotary
    # complex pair dimension (x.shape[-1]), so x must expose both axes.
    assert ndim >= 2, (
        "reshape_for_broadcast expects x to have at least two dimensions "
        f"with a sequence axis at dim=1; got x.ndim={ndim}, shape={tuple(x.shape)}"
    )
    expected_freqs_shape = (x.shape[1], x.shape[-1])
    assert freqs_cis.shape == expected_freqs_shape, (
        "freqs_cis shape must match x sequence and rotary dimensions: "
        f"expected {expected_freqs_shape}, got {tuple(freqs_cis.shape)}"
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_use_complex_fp32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Supports two paths:
    - rope_use_complex_fp32=True: complex-float32 arithmetic (for bf16-mixed / fp32 modes).
    - rope_use_complex_fp32=False: real-valued arithmetic in native dtype (for bf16-true mode).

    The two paths implement the same rotary transform. The complex path treats
    each even/odd hidden-dim pair as the real/imaginary part of a complex value
    and multiplies by cos + i sin; the real path expands the same multiplication
    into the equivalent 2D rotation formula. Numeric differences should only
    come from dtype and rounding behavior.

    Args:
        xq (torch.Tensor): Query tensor, shape (..., seq, n_heads, head_dim).
        xk (torch.Tensor): Key tensor, shape (..., seq, n_kv_heads, head_dim).
        freqs_cis (Tuple[torch.Tensor, torch.Tensor]): (cos, sin) each of
            shape (seq, head_dim//2), precomputed by precompute_freqs_cis.
        rope_use_complex_fp32 (bool): If True, use complex-float32 arithmetic.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated xq and xk in original dtype.
    """
    cos, sin = freqs_cis

    if rope_use_complex_fp32:
        # Complex-float32 path: stable for bf16-mixed/fp32 while preserving the
        # same rotary transform as the explicit real-valued branch below.
        freqs_cis_complex = cos + 1j * sin
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_complex = reshape_for_broadcast(freqs_cis_complex, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis_complex).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_complex).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    else:
        # Real-valued path: equivalent 2D rotation, kept in the input dtype for
        # bf16-true runs that should avoid fp32 casts in the forward graph.
        cos = cos.to(dtype=xq.dtype)
        sin = sin.to(dtype=xq.dtype)
        xq_r = xq.reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.reshape(*xk.shape[:-1], -1, 2)
        xq0, xq1 = xq_r[..., 0], xq_r[..., 1]
        xk0, xk1 = xk_r[..., 0], xk_r[..., 1]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        xq_out = torch.stack([xq0 * cos - xq1 * sin,
                              xq0 * sin + xq1 * cos], dim=-1).flatten(-2)
        xk_out = torch.stack([xk0 * cos - xk1 * sin,
                              xk0 * sin + xk1 * cos], dim=-1).flatten(-2)
        return xq_out, xk_out


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
    def __init__(self, layer_id: int, args: EBTModelArgs):
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
        self.use_ve = args.use_ve and has_ve is not None and has_ve(layer_id, args.n_layers)
        if self.use_ve:
            self.ve_gate_channels = 12
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_local_kv_heads, bias=False)
            nn.init.zeros_(self.ve_gate.weight)
        else:
            self.ve_gate = None
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        
        self.time_offset = 2 if args.use_mcmc_time_embed else 1
        self.rope_use_complex_fp32 = getattr(args, 'float_precision', '') != "bf16-true"
        self.use_sdpa_attention = args.use_sdpa_attention
        self.register_buffer('superdiag_rows', torch.arange(args.max_seq_len - 1))
        self.register_buffer('superdiag_cols', torch.arange(self.time_offset, args.max_seq_len + self.time_offset - 1))
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
        ve: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        # """
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        bsz, full_seqlen, _ = x.shape # full_seqlen includes real embeds and pred embeds
        original_seqlen = (full_seqlen + 1)//2 # this is just the condition plus all original tokens, +1 is bc of condition
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            xv = xv + gate.unsqueeze(-1) * ve
        
        # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]
        
        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]
        
        
        cos, sin = freqs_cis
        xq_o, xk_o = apply_rotary_emb(
            xq_o,
            xk_o,
            freqs_cis=(cos[:original_seqlen], sin[:original_seqlen]),
            rope_use_complex_fp32=self.rope_use_complex_fp32,
        )

        xq_p, xk_p = apply_rotary_emb(
            xq_p,
            xk_p,
            freqs_cis=(
                cos[self.time_offset:original_seqlen+1],
                sin[self.time_offset:original_seqlen+1],
            ),
            rope_use_complex_fp32=self.rope_use_complex_fp32,
        )
        # I tested this compared to prepending row on S dimension and the tensors were the same

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads # this does nothing since self.n_rep = 1
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        
        #original attn calc############################################

        # seqlen here is S-1 which = original_seqlen
        # Use F.scaled_dot_product_attention to avoid materialising the full [B, N, S, S]
        # attention score matrix, which OOMs at context=2048.
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)

        if self.use_sdpa_attention:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
                output_o = F.scaled_dot_product_attention(xq_o, keys_o, values_o, is_causal=True)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
        else:
            scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S-1, S-1
            if mask is not None:
                #this mask needs to be seqlen, seqlen, was S, S
                o_mask = mask[:-1, :-1] #set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc
                scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
            output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1) # has B, S-1, D after
        
        #pred sequence attn calc is for energy-based transformer ########################################################################################
        
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, pred_seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2) # (bs, n_local_heads, pred_seqlen, head_dim)
        values_p = xv_p.transpose(1, 2) # (bs, n_local_heads, pred_seqlen, head_dim)
        n_pred = full_seqlen - original_seqlen

        if self.use_sdpa_attention:
            K = original_seqlen           # number of orig tokens
            pred_seqlen = n_pred          # number of pred tokens

            keys_all   = torch.cat([keys_o,   keys_p],   dim=2)
            values_all = torch.cat([values_o, values_p], dim=2)
            orig_rows = torch.arange(pred_seqlen, device=x.device).unsqueeze(1)  # [pred_seqlen, 1]
            orig_cols = torch.arange(K,           device=x.device).unsqueeze(0)  # [1, K]
            orig_mask_part = torch.where(
                orig_cols <= orig_rows + (self.time_offset - 1),
                x.new_zeros(1),
                x.new_full((1,), float('-inf'))
            )  # [pred_seqlen, K]

            self_mask_part = x.new_full((pred_seqlen, pred_seqlen), float('-inf'))
            self_mask_part.fill_diagonal_(0.0)  # each pred token attends only to itself

            mask_p = torch.cat([orig_mask_part, self_mask_part], dim=1)  # [pred_seqlen, K + pred_seqlen]

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
                output_p = F.scaled_dot_product_attention(xq_p, keys_all, values_all, attn_mask=mask_p)
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)
        else:
            scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S-1, S; this uses xq_p and keys_o since for every next pred calcs similarity to all prev words; right S is because have extra condition

            temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device) # B, N, S-1, 1; is used since context_length = original_length +1, superdiag needs this
            scores_p = torch.cat((scores_p, temp_append), dim = -1)# is B, N, S-1, S; represents for each next pred (S-1 row) attending to all previous words (S-1) and then itself +1
            
            insertion_superdiagonal = (xq_p * keys_p).sum(dim = 3) / math.sqrt(self.head_dim)
            insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype) # for if using non 32 precision
            # bs, n, s-1 ; this calcs attn score of next preds with themselves, is like grabbing diag of matmul
            
            seq_len_minus_1 = scores_p.shape[2]
            superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
            superdiag_cols = self.superdiag_cols[:seq_len_minus_1]
      
            # first remove superdiagonal values so doesnt use attention to future tokens--prevents leakage of probability mass
            zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device) # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
            diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
            scores_p = scores_p * diagonal_removal_mask
            
            # then set diagonal to next pred self attention scores in differentiable way
            diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
            scores_p = scores_p + diagonal_addition_mask
            
            if mask is not None:
                p_mask = mask[self.time_offset:, :]  #S-1, S+1 like 0 0 0 -inf -inf -inf; 0 0 0 0 -inf -inf; etc
                scores_p = scores_p + p_mask
            if scores_p.dtype != torch.float32:
                scores_p = scores_p.float()
            scores_p = F.softmax(scores_p, dim=-1)
            if scores_p.dtype != xq_p.dtype:
                scores_p = scores_p.to(xq_p.dtype)
            
            #Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
            scores_p_superdiagonal = scores_p.diagonal(offset=self.time_offset, dim1=2, dim2=3) # is B, N, S; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this
            
            scores_p = scores_p * diagonal_removal_mask # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o
            
            scores_p = scores_p[:, :, :, :-1] # B, N, S-1, S now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
            output_p = torch.matmul(scores_p, values_o) # B, N, S-1, H; is how next preds attend to all original previous tokens;
            
            #next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
            next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim = -1) # B, N, S-1, H this is for weighted sum of each next pred to its final embed rep.
            
            output_p = output_p + next_pred_self_attention # B, N, S-1, H adding this is adding the aspect of each next pred embedding attending to itself
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1) # after this is B, S-1, D
        
        #return linear projection of concatted outputs ########################################################################################
        
        output = torch.cat((output_o, output_p), dim = 1) # B, 2(S-1)+1, D
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
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            ebt_act_func=args.ebt_act_func,
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        use_fp32 = getattr(args, 'float_precision', '') != "bf16-true"
        if args.ebt_norm == "rms":
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
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
            self.attention_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
            self.ffn_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
        else:
            raise ValueError(f"Invalid ebt_norm value: {args.ebt_norm}")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # x has shape B, 2*(S-1), D?
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, ve
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps, gradient_checkpointing=False, use_mcmc_time_embed=False):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.
            use_mcmc_time_embed (bool): If True, use per-step time embeddings (original behavior).
                If False, all MCMC steps share the same transition kernel, enabling arbitrary inference steps.

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
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mcmc_time_embed = use_mcmc_time_embed
        self.use_ve = params.use_ve and build_value_embeds is not None
        if self.use_ve:
            n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
            self.kv_dim = n_kv_heads * (params.dim // params.n_heads)
            self.value_embeds = build_value_embeds(params.n_layers, params.vocab_size, self.kv_dim)
            ve_init_bound = math.sqrt(3.0) * (params.dim ** -0.5)
            for ve in self.value_embeds.values():
                nn.init.uniform_(ve.weight, -ve_init_bound, ve_init_bound)
                ve.to(dtype=torch.bfloat16)
        else:
            self.value_embeds = None
            self.kv_dim = None

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        use_fp32 = getattr(params, 'float_precision', '') != "bf16-true"
        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps, use_fp32_norm=use_fp32)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False) # no learnable bias here since grad cant be computed for a final bias term in EBT
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps, use_fp32_norm=use_fp32)
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        if self.use_mcmc_time_embed:
            self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step=0,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
        return_pred_hidden: bool = False,
        return_energy: bool = True,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            start_pos (int): Starting position for attention caching.
            mcmc_step (int): Current MCMC step index, used for time embeddings.
            return_pred_hidden (bool): When True, also return the post-norm hidden
                state at candidate positions ([B, S, D]) used by the TF head.
            return_energy (bool): When False, skip the final scalar energy head.
                This is used by strict K=0 CE-only ablations that need trunk hidden
                states but must not run energy-related computation.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.
            (energies, pred_hidden) tuple when return_pred_hidden=True.

        """
        _bsz = embeddings.shape[0]

        if self.use_mcmc_time_embed:
            mcmc_step_val = mcmc_step
            mcmc_step = torch.empty((_bsz,), device=embeddings.device, dtype=torch.long).fill_(mcmc_step_val)
            time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1) # needs to be expanded to B, 1, D
            embeddings = torch.cat((time_embeddings, embeddings), dim = 1) # B, 2S - 1 + 1, D

        _bsz, seqlen = embeddings.shape[:2]
        if self.use_mcmc_time_embed:
            seqlen = (seqlen+3) // 2 # passed in seqlen is 2(S-1)+1+1(time) so add 3 div 2 = S+1
        else:
            seqlen = (seqlen+2) // 2 # passed in seqlen is 2(S-1) so add 2 div 2 = S
        # 动态扩展 freqs_cos/freqs_sin 如果需要的长度超过预计算的长度
        required_length = start_pos + seqlen
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length
            )
            self.freqs_cos = new_cos.to(embeddings.device)
            self.freqs_sin = new_sin.to(embeddings.device)

        freqs_cis = (
            self.freqs_cos[start_pos : start_pos + seqlen],
            self.freqs_sin[start_pos : start_pos + seqlen],
        )

        mask = None
        if seqlen > 1:
            mask = torch.empty((seqlen, seqlen), device=embeddings.device).fill_(float("-inf"))

            mask.triu_(diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # causal mask is like this by default 0, -inf, -inf
            #                         0, 0,    -inf
            #                         0, 0,    0

            for i, layer in enumerate(self.layers):
                ve = None
                if self.use_ve and real_token_ids is not None and predicted_tokens is not None:
                    extra_prefix_tokens = 1 if self.use_mcmc_time_embed else 0
                    ve = build_layer_ve(
                        self.value_embeds,
                        i,
                        real_token_ids,
                        predicted_tokens,
                        extra_prefix_tokens=extra_prefix_tokens,
                    )
                embeddings = layer(embeddings, start_pos, freqs_cis, mask, ve=ve)
            embeddings = self.norm(embeddings)
            if self.use_mcmc_time_embed:
                embeddings = embeddings[:, 1:] # remove temporal embed
            split_idx = embeddings.shape[1] // 2
            energies = None
            if return_energy:
                energies = self.final_layer(embeddings)
                energies = energies[:, split_idx:]
            if return_pred_hidden:
                # Candidate-position hidden state, post-norm, [B, S, D] — for TF head.
                pred_hidden = embeddings[:, split_idx:]
                return energies, pred_hidden
            return energies
