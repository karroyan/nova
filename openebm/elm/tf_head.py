"""TF (teacher-forced, Gemma-drafter-style) head modules for EBT.

Decoupled from the blockwise/MTP machinery in `nova/ebt/`. Each head takes
`(pred_hidden_or_state, prev_token_embed) -> logits` and is used in plain
(K=1) NTP on top of EBT's MCMC output. See `TF_HEAD_ARCHITECTURE.md` §9.

Variants exposed via `build_tf_head(hparams)`:
  - "linear":         Linear(2D, D) -> Linear(D, V)
  - "concat_direct_unembed": Linear(2D, V) over concat(pred_hidden, prev_embed)
  - "direct_unembed": Linear(D, V) over post-update trunk pred_hidden
  - "pre_update_hidden_unembed": Linear(D, V) over energy-forward trunk pred_hidden
  - "post_update_state_unembed": Linear over post-update free-embedding state
  - "transformer":    Linear(2D, D) -> L x causal AR block -> RMSNorm -> Linear(D, V)

The transformer variant matches Gemma-drafter's "concat-then-project + L tiny
attention blocks" pattern. L=1 is the empirical sweet spot in xxs experiments.

Init goes through `init_whole_model_weights(...)` (same as embeddings/trunk),
which respects `--weight_initialization_method` / `--weight_initialization_gain`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TFRMSNorm(nn.Module):
    """Standalone RMSNorm. Keeps an fp32 inner accumulation for bf16-mixed safety."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.type_as(x)) * self.weight


class _TFHeadBlock(nn.Module):
    """Pre-norm causal SDPA + SwiGLU FFN. Minimal AR transformer block."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0, norm_eps: float = 1e-5):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"tf_head: dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.attn_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = ((int(dim * ffn_mult) + 63) // 64) * 64
        self.ffn_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if S > 1:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = v
        x = x + self.wo(attn.transpose(1, 2).reshape(B, S, D))
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class TFLinearHead(nn.Module):
    """forward(pred_hidden[B,S,D], prev_embed[B,S,D]) -> logits[B,S,V]."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        return self.out_proj(self.down_proj(x))


class TFConcatDirectUnembedHead(nn.Module):
    """Directly project concat(pred_hidden, prev_embed) to vocab logits."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        return self.proj(x)


class TFDirectUnembedHead(nn.Module):
    """Project a D-dim TF signal to vocab logits.

    `prev_token_embed` is the first-layer embedding from self.embeddings(input_ids)
    and is intentionally ignored here. It stays in the signature only to match the
    shared TF-head interface used by the linear/transformer heads. The D-dim input
    can be a trunk last-layer hidden or a free-embedding MCMC state, depending on
    the selected tf_head_type.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        del prev_token_embed
        return self.proj(pred_hidden)


class TFPostUpdateStateUnembedHead(nn.Module):
    """Decode the post-update free-embedding MCMC state.

    Optional modes:
      - RMSNorm: z_{i+1} -> RMSNorm -> Linear(D,V)
      - concat zi: concat(z_i, z_{i+1}) -> Linear(2D,V)
      - concat prev embed: concat(z_{i+1}, embed(input_ids)) -> Linear(2D,V)
      - both: concat(z_i, RMSNorm(z_{i+1})) -> Linear(2D,V)
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        use_rmsnorm: bool = False,
        concat_zi: bool = False,
        concat_prev_embed: bool = False,
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.concat_zi = concat_zi
        self.concat_prev_embed = concat_prev_embed
        if concat_zi and concat_prev_embed:
            raise ValueError("post-update state head supports only one concat source at a time.")
        self.state_norm = _TFRMSNorm(dim) if use_rmsnorm else None
        in_dim = 2 * dim if (concat_zi or concat_prev_embed) else dim
        self.proj = nn.Linear(in_dim, vocab_size, bias=False)

    def forward(self, state_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor], prev_token_embed: torch.Tensor) -> torch.Tensor:
        if self.concat_zi:
            if not isinstance(state_input, tuple) or len(state_input) != 2:
                raise ValueError("post_update_state_concat_zi expects (z_next, z_i).")
            z_next, z_i = state_input
        else:
            if isinstance(state_input, tuple):
                z_next = state_input[0]
            else:
                z_next = state_input
            z_i = None

        if self.state_norm is not None:
            z_next = self.state_norm(z_next)

        if self.concat_zi:
            z_next = torch.cat([z_i, z_next], dim=-1)
        elif self.concat_prev_embed:
            z_next = torch.cat([z_next, prev_token_embed], dim=-1)

        return self.proj(z_next)


class TFTransformerHead(nn.Module):
    """forward(pred_hidden[B,S,D], prev_embed[B,S,D]) -> logits[B,S,V].

    `num_layers` causal transformer blocks between the concat-projection and the
    final V-projection. Gemma-drafter shape.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_layers: int = 1,
        n_heads: int = 8,
        ffn_mult: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.blocks = nn.ModuleList(
            [_TFHeadBlock(dim, n_heads, ffn_mult, norm_eps=norm_eps) for _ in range(num_layers)]
        )
        self.final_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        x = self.down_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.final_norm(x))


def build_tf_head(hparams) -> nn.Module:
    """Factory that consumes hparams namespace from train.py CLI flags.

    Reads:
      hparams.tf_head_type       in {"linear", "concat_direct_unembed", "direct_unembed",
                                     "pre_update_hidden_unembed", "post_update_state_unembed",
                                     "transformer"}
      hparams.tf_head_layers     int (transformer only)
      hparams.tf_head_n_heads    int, 0 -> inherit hparams.multiheaded_attention_heads
      hparams.tf_head_ffn_mult   float
      hparams.embedding_dim, hparams.vocab_size
    """
    dim = int(hparams.embedding_dim)
    vocab_size = int(hparams.vocab_size)
    head_type = getattr(hparams, "tf_head_type", "transformer")

    if head_type == "linear":
        return TFLinearHead(dim=dim, vocab_size=vocab_size)

    if head_type == "concat_direct_unembed":
        return TFConcatDirectUnembedHead(dim=dim, vocab_size=vocab_size)

    if head_type in {"direct_unembed", "pre_update_hidden_unembed"}:
        return TFDirectUnembedHead(dim=dim, vocab_size=vocab_size)

    if head_type == "post_update_state_unembed":
        return TFPostUpdateStateUnembedHead(
            dim=dim,
            vocab_size=vocab_size,
            use_rmsnorm=bool(getattr(hparams, "post_update_state_use_rmsnorm", False)),
            concat_zi=bool(getattr(hparams, "post_update_state_concat_zi", False)),
            concat_prev_embed=bool(getattr(hparams, "post_update_state_concat_prev_embed", False)),
        )

    if head_type == "transformer":
        n_heads_override = int(getattr(hparams, "tf_head_n_heads", 0))
        n_heads_default = int(getattr(hparams, "multiheaded_attention_heads", 8))
        n_heads = n_heads_override if n_heads_override > 0 else n_heads_default
        return TFTransformerHead(
            dim=dim,
            vocab_size=vocab_size,
            num_layers=int(getattr(hparams, "tf_head_layers", 1)),
            n_heads=n_heads,
            ffn_mult=float(getattr(hparams, "tf_head_ffn_mult", 4.0)),
        )

    raise ValueError(f"Unknown tf_head_type: {head_type!r}")
