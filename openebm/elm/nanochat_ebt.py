import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nanochat.gpt import GPT, GPTConfig

class NanoChatEBT(GPT):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.config.n_layer):
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
            self.params.dim // self.config.n_head, self.config.sequence_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(self, embeddings: torch.Tensor, start_pos: int, mcmc_step = 0):
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.

        """
        _bsz = embeddings.shape[0]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device = embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1) # needs to be expanded to B, 1, D
        embeddings = torch.cat((time_embeddings, embeddings), dim = 1) # B, 2S - 1, D

        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen+3) // 2 # do this since passed in seqlen is 2(S-1)+1 so add 3 div 2 = S+1 which corresponds to concatting time embed
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

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
                
            for i, layer in enumerate(self.transformer):
                embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings)
            embeddings = embeddings[:, 1:] # remove temporal embed
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies