import torch
from torch import nn
from torch.nn import functional as F
from nanolightning.torchlightning_module import LightningModule
# import torch.optim as optim
# from torchmetrics import Accuracy
# from transformers import AutoTokenizer

import math
import random
import os
import inspect
from utils import setup_ebt, init_whole_model_weights
from utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from utils import EXPLICIT_BLOCK_LATENT_MODES
from replay_buffer import CausalReplayBuffer
from metrics import calculate_bpb_score

import ipdb

class EBT_NLP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        # tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        # Use tokenizer_obj if available (set by ModelTrainer), otherwise use tokenizer directly
        self.tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        self.tokenizer_pad_token_id = None # Nanochat doesn't have <|pad|> or <|eos|> # self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.get_vocab_size() # len(self.tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        max_blockwise_offsets = max(1, int(getattr(self.hparams, "train_block_size", 1)))
        # Dense blockwise joint head:
        # shared trunk features [B, S, D] -> [B, S, K_max * V], then reshape to [B, K, S, V].
        self.blockwise_joint_head = nn.Linear(
            self.hparams.embedding_dim,
            max_blockwise_offsets * self.vocab_size,
            bias=False,
        )
        init_whole_model_weights(
            self.blockwise_joint_head,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )

        # Single-token head used by the explicit-block-latent modes
        # (future_latent_non_causal / blockwise). Each future latent's hidden
        # state is mapped through this head to produce a single token's logits;
        # there is no joint multi-offset projection here. Kept independent of
        # `blockwise_joint_head` so mtp_mcmc behavior is byte-identical.
        self.block_latent_token_head = nn.Linear(
            self.hparams.embedding_dim,
            self.vocab_size,
            bias=False,
        )
        init_whole_model_weights(
            self.block_latent_token_head,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        
        if not self.hparams.vocab_to_embed_uses_prob_dist: # if are not using the prob dist * embed as vocab to embed
            if 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory and self.hparams.process_memory_type != None:
                self.vocab_to_embed = Memory_Gating_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.process_memory_type, self.hparams.process_memory_linear_layer)
            elif 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory:
                assert self.hparams.num_modality_processing_mlp_layers > 1, "must set self.hparams.num_modality_processing_mlp_layers > 1 if not using self.hparams.process_memory_type"
                self.vocab_to_embed = Memory_Augmented_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers)
            elif self.hparams.num_modality_processing_mlp_layers != 1:
                self.vocab_to_embed = MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers - 2)
            else:
                self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias = False, device = self.device) #NOTE this is ebt special, since we want to input a prob dist and pred this prob dist but the transformer needs an embedding as input
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        self.transformer = setup_ebt(self.hparams)
        self._transformer_accepts_return_pred_hidden = (
            "return_pred_hidden" in inspect.signature(self.transformer.forward).parameters
        )
        self._transformer_accepts_return_context_hidden = (
            "return_context_hidden" in inspect.signature(self.transformer.forward).parameters
        )
        # Single source of truth for the attention / training semantic the
        # trunk and all downstream paths (train, MCMC, blockwise dense, block
        # refine, inference) must consistently use. If not set on hparams
        # (e.g. a legacy checkpoint), we default to dense_token which matches
        # original main-branch EBT semantics.
        self._block_mode = getattr(self.hparams, "block_mode", "dense_token")
        
        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        # DEBUGGING CODE ################################################################################################################################################
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update
        
    @torch.compiler.disable
    def _mcmc_step_excluded(self, predicted_tokens, real_embeddings_input, mcmc_step, i, num_mcmc_steps, 
                      langevin_dynamics_noise_std, alpha, start_pos, learning, return_raw_logits):
        batch_size = predicted_tokens.shape[0]
        seq_length = predicted_tokens.shape[1]
        
        if self.hparams.no_mcmc_detach:
            predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
        else: # default, do detach
            predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

        if self.hparams.langevin_dynamics_noise != 0:
            ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
            predicted_tokens = predicted_tokens + ld_noise

        if self.hparams.normalize_initial_condition:
            if self.hparams.normalize_initial_condition_only_first_step:
                if mcmc_step == 0:
                    predicted_tokens = self.softmax(predicted_tokens)
            else:
                predicted_tokens = self.softmax(predicted_tokens)
                
            if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
            else:
                predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        else:
            predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        
        all_embeddings = torch.cat((real_embeddings_input.detach(), predicted_embeddings), dim = 1) # B, 2*S, D
        
        energy_preds = self.transformer(all_embeddings, start_pos = start_pos, mcmc_step=mcmc_step, block_mode=self._block_mode) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
        energy_preds = energy_preds.reshape(-1, 1)
        
        if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
            if i == (num_mcmc_steps - 1):
                predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
            else:
                predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=False)[0]
        else:
            predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
        # predicted_tokens_grad has shape B, S, V
        
        if self.hparams.clamp_futures_grad:
            min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha) # use self.alpha and not random alpha to clamp
            # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
            predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
            
        if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
            raise ValueError("NaN or Inf gradients detected during MCMC.")
        
        predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after  
        
        if self.hparams.absolute_clamp != 0.0:
            predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
        
        if self.hparams.sharpen_predicted_distribution != 0.0:
            predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

        if return_raw_logits:
            predicted_tokens_for_loss = predicted_tokens # BS, S, V
        else:
            predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
            
        return predicted_tokens, energy_preds, predicted_tokens_for_loss

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True, block_size = None, kv_cache = None): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        # KV cache (dense_token / mtp_mcmc inference). When the cache already
        # has context (decode mode), the input x carries only the *new*
        # context token(s); the symmetric (S=K) layout no longer applies and
        # we force block_size=1 for sequential decoding.
        is_decode_with_cache = (
            kv_cache is not None
            and kv_cache.cached_len > 0
            and self._block_mode in ("dense_token", "mtp_mcmc")
        )
        if block_size is None:
            # Default block_size differs by block_mode:
            #   * dense_token / mtp_mcmc: legacy symmetric layout, defaults
            #     to seq_length (S=K, the historical contract).
            #   * future_latent_non_causal / blockwise: this entry point is
            #     the inference C+K layout (training uses
            #     forward_explicit_block_latent_logits directly), so the
            #     natural default is K=1 (sequential decoding).
            if is_decode_with_cache:
                block_size = 1
            elif self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
                block_size = getattr(self.hparams, "block_size", 1)
            else:
                block_size = getattr(self.hparams, "block_size", seq_length)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        # block_mode dispatch:
        #   * dense_token / mtp_mcmc: legacy symmetric path. Requires
        #     block_size == seq_length inside the trunk. This path is byte-
        #     identical to the previous mtp_mcmc behavior.
        #   * future_latent_non_causal / blockwise: new modes. We always use
        #     the inference layout here (C context tokens + K = block_size
        #     latents) so this entry point can be used directly for sequential
        #     (K=1) and direct_block (K>1) inference. Training does NOT call
        #     forward(); it goes through forward_explicit_block_latent_logits.
        if self._block_mode in ("dense_token", "mtp_mcmc"):
            if (not is_decode_with_cache) and block_size != seq_length:
                raise NotImplementedError(
                    f"EBT_NLP.forward with block_size != seq_length is not supported under "
                    f"block_mode={self._block_mode!r}; this path requires the 'blockwise' "
                    f"block_mode which is not implemented yet. Use sequential inference, "
                    f"or train a blockwise-mode checkpoint to enable non-symmetric block "
                    f"prediction. Got block_size={block_size}, seq_length={seq_length}."
                )
            if is_decode_with_cache and (seq_length != 1 or block_size != 1):
                raise NotImplementedError(
                    "KV cache decode currently expects seq_length=1 and block_size=1 "
                    f"(sequential decoding); got seq_length={seq_length}, block_size={block_size}."
                )
        elif self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            ebt_type = getattr(self.hparams, "ebt_type", "default")
            if ebt_type not in ("default", "time_embed"):
                raise NotImplementedError(
                    f"block_mode={self._block_mode!r} is currently only supported for "
                    f"ebt_type in [default, time_embed]; got ebt_type={ebt_type}."
                )
            return self._forward_explicit_block_latent_inference(
                real_embeddings_input=real_embeddings_input,
                block_size=block_size,
                start_pos=start_pos,
                learning=learning,
                return_raw_logits=return_raw_logits,
                no_randomness=no_randomness,
            )
        else:
            raise ValueError(f"Unknown block_mode={self._block_mode!r} on EBT_NLP")

        if getattr(self.hparams, "ebt_type", "default") not in ("default", "time_embed") and block_size != seq_length:
            raise NotImplementedError(
                f"block_size != seq_length is currently only supported for ebt_type in [default, time_embed]; got ebt_type={self.hparams.ebt_type}, block_size={block_size}, seq_length={seq_length}"
            )

        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, block_size, 1)
            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        if is_decode_with_cache:
            # RNG-compat: in decode mode `block_size == 1`, but the matching
            # no-cache call at the same absolute position would corrupt over
            # `(B, cached_len + 1, V)` entries. To keep generation byte-equal
            # to no-cache (so cache vs no-cache outputs are identical given the
            # same seed), allocate the *full* noise tensor and then slice the
            # last row. This trades a small amount of wasted compute for
            # end-to-end determinism w.r.t. the no-cache baseline.
            full_target = kv_cache.cached_len + block_size  # = effective_S_total
            full_noise = self.corrupt_embeddings(real_embeddings_input, target_length=full_target)  # (B, S_total, V)
            predicted_tokens = full_noise[:, -block_size:, :].contiguous()  # (B, 1, V)
        else:
            predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=block_size) # B, K, V
        if replay_buffer_logits is not None: # using replay buffer, use the logits instead of corruption
            if block_size != seq_length:
                raise NotImplementedError("replay_buffer_logits path only supports block_size == seq_length")
            if replay_buffer_logits.shape[1] != block_size:
                raise ValueError(
                    f"replay_buffer_logits shape mismatch: expected second dim {block_size}, got {replay_buffer_logits.shape[1]}"
                )
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits # NOTE this assumes the fresh data is concatted first

        _, predicted_distributions, predicted_energies = self._run_mcmc_on_given_pred_tokens(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            start_pos=start_pos,
            learning=learning,
            return_raw_logits=return_raw_logits,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            kv_cache=kv_cache,
        )
        return predicted_distributions, predicted_energies

    def _build_mcmc_steps(self, no_randomness=True):
        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.mcmc_num_steps):
            if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
                if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
                    if step == (self.hparams.mcmc_num_steps - 1):
                        min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                        repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                        mcmc_steps.extend([step] * repeats)
                    else:
                        mcmc_steps.append(step)
                else:
                    min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                    repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                    mcmc_steps.extend([step] * repeats)
            elif no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0: # use max steps
                if step == (self.hparams.mcmc_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)
        return mcmc_steps

    def _run_mcmc_on_given_pred_tokens(
        self,
        real_embeddings_input,
        predicted_tokens,
        start_pos=0,
        learning=True,
        return_raw_logits=False,
        no_randomness=True,
        alpha=None,
        langevin_dynamics_noise_std=None,
        optimize_mask=None,
        mcmc_steps=None,
        return_pred_hidden=False,
        return_context_hidden=False,
        kv_cache=None,
    ):
        predicted_distributions = []
        predicted_energies = []
        predicted_hiddens = []
        context_hiddens = []
        batch_size, seq_length, _ = predicted_tokens.shape
        if alpha is None:
            alpha = torch.clamp(self.alpha, min=0.0001)
        if langevin_dynamics_noise_std is None:
            langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        if mcmc_steps is None:
            mcmc_steps = self._build_mcmc_steps(no_randomness=no_randomness)

        need_post_update_hidden = return_context_hidden or return_pred_hidden
        if return_context_hidden and not self._transformer_accepts_return_context_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_context_hidden=True"
            )
        if return_pred_hidden and not self._transformer_accepts_return_pred_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_pred_hidden=True"
            )

        optimize_mask_bool = None
        optimize_mask_float = None
        fixed_logits = None
        if optimize_mask is not None:
            optimize_mask_bool = optimize_mask.to(dtype=torch.bool)
            optimize_mask_float = optimize_mask.to(dtype=predicted_tokens.dtype)
            fixed_logits = predicted_tokens.detach()

        # Snapshot each MCMC step's sub-cache length at the start of this outer
        # step so every MCMC inner iteration sees the same baseline (the prior
        # outer steps' cached context). Each iteration rebuilds its own step's
        # new-ctx K/V idempotently on top of the snapshot. Sub-caches are
        # per-MCMC-step because context K/V depends on the time-embedding token.
        kv_cache_snapshot = None
        if kv_cache is not None:
            kv_cache_snapshot = {s: kv_cache.cached_len_for_step(s) for s in set(mcmc_steps)}

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if kv_cache is not None:
                    kv_cache.reset_step_to(mcmc_step, kv_cache_snapshot[mcmc_step])
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
                else: # default, do detach
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
                    if optimize_mask_float is not None:
                        ld_noise = ld_noise * optimize_mask_float
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)

                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D

                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, S+K, D
                base_transformer_kwargs = dict(
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=predicted_embeddings.shape[1],
                    block_mode=self._block_mode,
                )
                if kv_cache is not None:
                    base_transformer_kwargs["kv_cache"] = kv_cache
                energy_preds = self.transformer(
                    all_embeddings,
                    **base_transformer_kwargs,
                ) # checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)

                if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                    if i == (len(mcmc_steps) - 1):
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                    else:
                        predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=False)[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                # predicted_tokens_grad has shape B, S, V

                if optimize_mask_float is not None:
                    predicted_tokens_grad = predicted_tokens_grad * optimize_mask_float

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha) # use self.alpha and not random alpha to clamp
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)

                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                if optimize_mask_bool is not None:
                    predicted_tokens = torch.where(optimize_mask_bool, predicted_tokens, fixed_logits)

                if need_post_update_hidden:
                    if kv_cache is not None:
                        kv_cache.reset_step_to(mcmc_step, kv_cache_snapshot[mcmc_step])
                    post_update_pred_embeddings = self._logits_to_pred_embeddings(predicted_tokens, mcmc_step)
                    post_update_all_embeddings = torch.cat((real_embeddings_input, post_update_pred_embeddings), dim=1)
                    post_transformer_kwargs = dict(base_transformer_kwargs)
                    post_transformer_kwargs["return_context_hidden"] = return_context_hidden
                    post_transformer_kwargs["return_pred_hidden"] = return_pred_hidden
                    post_transformer_out = self.transformer(
                        post_update_all_embeddings,
                        **post_transformer_kwargs,
                    )
                    if return_context_hidden and return_pred_hidden:
                        _, context_hidden, pred_hidden = post_transformer_out
                        context_hiddens.append(context_hidden)
                        predicted_hiddens.append(pred_hidden)
                    elif return_context_hidden:
                        _, context_hidden = post_transformer_out
                        context_hiddens.append(context_hidden)
                    elif return_pred_hidden:
                        _, pred_hidden = post_transformer_out
                        predicted_hiddens.append(pred_hidden)

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens # BS, S, V
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
                predicted_distributions.append(predicted_tokens_for_loss)

        if return_context_hidden and return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, context_hiddens, predicted_hiddens
        if return_context_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, context_hiddens
        if return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, predicted_hiddens
        return predicted_tokens, predicted_distributions, predicted_energies

    def _discrete_token_ids_to_initial_logits(self, token_ids, init_logit_scale=8.0):
        logits = torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            self.vocab_size,
            device=token_ids.device,
            dtype=self.embeddings.weight.dtype,
        )
        return logits.scatter_(-1, token_ids.unsqueeze(-1), float(init_logit_scale))

    def _draft_block_ids_to_initial_logits(self, draft_block_ids, init_logit_scale=8.0):
        return self._discrete_token_ids_to_initial_logits(draft_block_ids, init_logit_scale=init_logit_scale)

    def _logits_to_pred_embeddings(self, logits, step_idx):
        if self.hparams.normalize_initial_condition:
            if self.hparams.normalize_initial_condition_only_first_step:
                if step_idx == 0:
                    logits = self.softmax(logits)
            else:
                logits = self.softmax(logits)

            if self.hparams.vocab_to_embed_uses_prob_dist:
                return torch.matmul(logits, self.embeddings.weight)
            return self.vocab_to_embed(logits)
        return self.vocab_to_embed(logits)

    def ebt_refine_block_fast(self, context_ids, draft_block_ids, refine_steps=None, init_logit_scale=8.0, start_pos=0, learning=False):
        """
        Fast block refinement: only block logits require grad.
        Returns:
            refined_block_logits: [B, K, V]
            refined_block_ids: [B, K]
        """
        if draft_block_ids.shape[1] == 0:
            empty_logits = torch.empty(
                draft_block_ids.shape[0], 0, self.vocab_size, device=draft_block_ids.device, dtype=self.embeddings.weight.dtype
            )
            return empty_logits, draft_block_ids

        if context_ids.shape[1] == 0:
            raise ValueError("ebt_refine_block_fast requires at least one context token")

        # EBT-compatible pair:
        # real_input_ids: [context, draft[:-1]] length = C + K - 1
        # predicted targets: [context[1:], draft] where only draft logits are optimized
        real_input_ids = torch.cat([context_ids, draft_block_ids[:, :-1]], dim=1)
        block_len = draft_block_ids.shape[1]
        real_embeddings_input = self.embeddings(real_input_ids)
        context_target_ids = context_ids[:, 1:]
        fixed_prefix_logits = self._discrete_token_ids_to_initial_logits(
            context_target_ids, init_logit_scale=init_logit_scale
        ).detach()
        block_logits = self._draft_block_ids_to_initial_logits(
            draft_block_ids, init_logit_scale=init_logit_scale
        ).detach()
        init_block_logits = block_logits.detach().clone()

        alpha = torch.clamp(self.alpha, min=0.0001)
        noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        requested_steps = int(refine_steps) if refine_steps is not None else int(self.hparams.mcmc_num_steps)
        if requested_steps <= 0:
            refined_block_ids = torch.argmax(block_logits, dim=-1)
            return block_logits, refined_block_ids
        effective_steps = min(requested_steps, int(self.hparams.mcmc_num_steps))
        mcmc_steps = list(range(effective_steps))
        diagnose = bool(getattr(self.hparams, "infer_block_diagnose", False))

        def _block_energy_mean(logits, step_idx):
            with torch.no_grad():
                prefix_embeds = self._logits_to_pred_embeddings(fixed_prefix_logits, step_idx)
                block_embeds = self._logits_to_pred_embeddings(logits, step_idx)
                pred_embeddings = torch.cat([prefix_embeds, block_embeds], dim=1)
                all_embeddings = torch.cat((real_embeddings_input, pred_embeddings), dim=1)
                energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
                return energy_preds[:, -block_len:].mean().item()

        initial_energy_mean = _block_energy_mean(init_block_logits, mcmc_steps[0])
        last_grad_norm = 0.0

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                block_logits = block_logits.detach().requires_grad_()
                cur_block_logits = block_logits
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(cur_block_logits.detach()) * noise_std
                    cur_block_logits = cur_block_logits + ld_noise

                prefix_embeds = self._logits_to_pred_embeddings(fixed_prefix_logits, mcmc_step)
                block_embeds = self._logits_to_pred_embeddings(cur_block_logits, mcmc_step)
                pred_embeddings = torch.cat([prefix_embeds, block_embeds], dim=1)
                all_embeddings = torch.cat((real_embeddings_input, pred_embeddings), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=pred_embeddings.shape[1],
                    block_mode=self._block_mode,
                )
                energy_block = energy_preds[:, -block_len:].reshape(-1, 1)

                block_grad = torch.autograd.grad([energy_block.sum()], [cur_block_logits], create_graph=learning)[0]
                last_grad_norm = block_grad.norm().item()
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    block_grad = torch.clamp(block_grad, min=-min_and_max, max=min_and_max)
                if torch.isnan(block_grad).any() or torch.isinf(block_grad).any():
                    raise ValueError("NaN or Inf gradients detected during block refinement.")

                block_logits = cur_block_logits - alpha * block_grad
                if self.hparams.absolute_clamp != 0.0:
                    block_logits = torch.clamp(block_logits, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp)
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    block_logits = block_logits / self.hparams.sharpen_predicted_distribution
                block_logits = block_logits.detach()

        final_energy_mean = _block_energy_mean(block_logits, mcmc_steps[-1])
        refined_block_logits = block_logits
        refined_block_ids = torch.argmax(refined_block_logits, dim=-1)

        if diagnose:
            delta_norm = (refined_block_logits - init_block_logits).norm().item()
            max_show = min(2, draft_block_ids.shape[0])
            for b in range(max_show):
                print(f"draft_block_ids[{b}]: {draft_block_ids[b].tolist()}", flush=True)
                print(f"refined_block_ids[{b}]: {refined_block_ids[b].tolist()}", flush=True)
            print(f"||refined_logits - init_logits||: {delta_norm:.6f}", flush=True)
            print(f"block 初始 energy: {initial_energy_mean:.6f}", flush=True)
            print(f"block 最终 energy: {final_energy_mean:.6f}", flush=True)
            print(f"block logits 梯度范数 grad.norm(): {last_grad_norm:.6f}", flush=True)

        return refined_block_logits, refined_block_ids

    def ebt_refine_block(self, context_ids, draft_block_ids, refine_steps=None, init_logit_scale=8.0, start_pos=0, learning=False):
        # Backward-compatible wrapper for older callsites.
        return self.ebt_refine_block_fast(
            context_ids=context_ids,
            draft_block_ids=draft_block_ids,
            refine_steps=refine_steps,
            init_logit_scale=init_logit_scale,
            start_pos=start_pos,
            learning=learning,
        )

    def forward_blockwise_dense_hidden(self, input_ids, no_randomness):
        """
        Shared-trunk blockwise dense hidden extraction for the dense blockwise head.
        IMPORTANT: use post-update pred_hidden, not context_hidden.
        In the current attention layout, context_hidden does not attend to the pred branch,
        so even post-update context_hidden stays independent of alpha. post-update pred_hidden
        does depend on x' = x - alpha * grad(E), which restores a differentiable alpha path.
        Returns:
            pred_hiddens_per_step: list[[B, S_eff, D]]
            predicted_energies: list[[B*S_eff, 1]]
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S_eff], got shape {tuple(input_ids.shape)}")

        real_embeddings_input = self.embeddings(input_ids)
        seq_eff = input_ids.shape[1]
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=seq_eff)  # [B, S_eff, V]

        _, _, predicted_energies, pred_hiddens_per_step = self._run_mcmc_on_given_pred_tokens(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            start_pos=0,
            learning=True,
            return_raw_logits=True,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )
        return pred_hiddens_per_step, predicted_energies

    def forward_blockwise_dense_logits(self, input_ids, num_offsets, no_randomness, return_hidden=False):
        """
        Shared-trunk blockwise dense prediction.
        Returns:
            multi_offset_logits_per_step: list[[B, K, S_eff, V]]
            predicted_energies: list[[B*S_eff, 1]]
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S_eff], got shape {tuple(input_ids.shape)}")
        if num_offsets <= 0:
            raise ValueError(f"num_offsets must be > 0, got {num_offsets}")
        max_offsets = self.blockwise_joint_head.out_features // self.vocab_size
        if num_offsets > max_offsets:
            raise ValueError(
                f"Requested K={num_offsets} offsets but model only initialized K_max={max_offsets}. "
                "Increase --train_block_size and reinitialize model."
            )

        batch_size = input_ids.shape[0]
        seq_eff = input_ids.shape[1]
        pred_hiddens_per_step, predicted_energies = self.forward_blockwise_dense_hidden(
            input_ids=input_ids,
            no_randomness=no_randomness,
        )

        multi_offset_logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            projected = self.blockwise_joint_head(pred_hidden)  # [B, S_eff, K_max*V]
            projected = projected.reshape(batch_size, seq_eff, max_offsets, self.vocab_size)
            projected = projected[:, :, :num_offsets, :]  # [B, S_eff, K, V]
            projected = projected.permute(0, 2, 1, 3).contiguous()  # [B, K, S_eff, V]
            multi_offset_logits_per_step.append(projected)

        if return_hidden:
            return multi_offset_logits_per_step, predicted_energies, pred_hiddens_per_step
        return multi_offset_logits_per_step, predicted_energies

    # ------------------------------------------------------------------
    # Explicit-block-latent paths (future_latent_non_causal / blockwise).
    # ------------------------------------------------------------------
    # These paths are intentionally kept independent of the dense_token /
    # mtp_mcmc paths above so that any change here is guaranteed to leave
    # mtp_mcmc behavior byte-identical. New helpers, new MCMC loop, new
    # head, new inference entry points.
    #
    # Latent shape convention (logical):
    #     predicted_tokens / pred_hidden : [B, S, K, V] / [B, S, K, D]
    # Internally we flatten to position-major [B, S*K, *] for the trunk:
    #     flatten order: outer source position t (0..S-1), inner block
    #     offset j (0..K-1). i.e. for position-major index p, t = p // K,
    #     j = p % K. This MUST match `build_explicit_block_latent_mask`
    #     and `build_explicit_block_latent_freq_indices` in utils.py.
    #
    # Each future latent z_{t,j} predicts a single token corresponding to
    # block_targets[:, j, t] (using the dataset convention that
    # block_targets has shape [B, K, S]).
    # ------------------------------------------------------------------

    def _explicit_block_latent_pred_hidden_to_logits(self, pred_hidden, S, K):
        """Map a position-major pred_hidden ``[B, S*K, D]`` to logits
        ``[B, K, S, V]`` aligned with ``block_targets [B, K, S]``.

        Steps:
          1. ``self.block_latent_token_head`` : ``[B, S*K, V]`` (single-token
             head; not the joint head used by mtp_mcmc).
          2. Reshape to ``[B, S, K, V]`` (position-major: outer t, inner j).
          3. Permute to ``[B, K, S, V]`` to match ``block_targets``.
        """
        B, P, _ = pred_hidden.shape
        if P != S * K:
            raise ValueError(
                f"pred_hidden length mismatch: expected S*K={S*K}, got {P}"
            )
        logits = self.block_latent_token_head(pred_hidden)  # [B, S*K, V]
        logits = logits.reshape(B, S, K, self.vocab_size)   # [B, S, K, V]
        logits = logits.permute(0, 2, 1, 3).contiguous()    # [B, K, S, V]
        return logits

    def _run_explicit_block_latent_mcmc(
        self,
        real_embeddings_input,
        predicted_tokens,
        block_size,
        start_pos=0,
        learning=True,
        return_raw_logits=False,
        no_randomness=True,
        alpha=None,
        langevin_dynamics_noise_std=None,
        mcmc_steps=None,
        return_pred_hidden=False,
    ):
        """MCMC loop dedicated to the explicit-block-latent modes.

        Operates on a flattened [B, P, V] latent where ``P`` is either
        ``S * K`` (training layout, with S = real_embeddings_input.shape[1])
        or ``K`` (inference layout). The trunk picks the correct sequence
        layout by inspecting ``pred_len``; see
        ``EBTTimeConcat._forward_explicit_block_latent`` /
        ``EBTDefault._forward_explicit_block_latent``.

        Energies of all P pred latents are summed before back-propagation,
        consistent with EBT/MCMC where all pred-token energies contribute
        to the gradient on the latents.

        IMPORTANT: this function is NOT used by mtp_mcmc; mtp_mcmc still
        uses ``_run_mcmc_on_given_pred_tokens`` which is left untouched.
        """
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "_run_explicit_block_latent_mcmc must only be called under "
                f"block_mode in {EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        if predicted_tokens.dim() != 3 or predicted_tokens.shape[-1] != self.vocab_size:
            raise ValueError(
                "predicted_tokens must be [B, P, V] with V == vocab_size; "
                f"got shape={tuple(predicted_tokens.shape)}, vocab_size={self.vocab_size}"
            )
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        B, P, V = predicted_tokens.shape
        S = real_embeddings_input.shape[1]
        if P == S * K:
            layout = "training"
        elif P == K:
            layout = "inference"
        else:
            raise ValueError(
                "explicit-block-latent MCMC requires pred_len in {S*K, K}; "
                f"got pred_len={P}, S={S}, K={K}."
            )

        if alpha is None:
            alpha = torch.clamp(self.alpha, min=0.0001)
        if langevin_dynamics_noise_std is None:
            langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        if mcmc_steps is None:
            mcmc_steps = self._build_mcmc_steps(no_randomness=no_randomness)

        if return_pred_hidden and not self._transformer_accepts_return_pred_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_pred_hidden=True"
            )

        predicted_distributions = []
        predicted_energies = []
        predicted_hiddens = []

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_().reshape(B, P, V)
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(B, P, V)

                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)

                    if self.hparams.vocab_to_embed_uses_prob_dist:
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight)
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens)
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)

                # all_embeddings: [B, S + P, D]
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim=1)
                trunk_kwargs = dict(
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=S,
                    pred_len=P,
                    block_size=K,
                    block_mode=self._block_mode,
                )
                energy_preds = self.transformer(all_embeddings, **trunk_kwargs)
                # energy_preds: [B, P, 1]
                energy_preds_flat = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds_flat)

                if self.hparams.truncate_mcmc:
                    if i == (len(mcmc_steps) - 1):
                        predicted_tokens_grad = torch.autograd.grad(
                            [energy_preds_flat.sum()], [predicted_tokens], create_graph=learning
                        )[0]
                    else:
                        predicted_tokens_grad = torch.autograd.grad(
                            [energy_preds_flat.sum()], [predicted_tokens], create_graph=False
                        )[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad(
                        [energy_preds_flat.sum()], [predicted_tokens], create_graph=learning
                    )[0]

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min=-min_and_max, max=min_and_max)

                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during explicit-block-latent MCMC.")

                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(
                        predicted_tokens, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp
                    )
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                if return_pred_hidden:
                    # Re-run trunk on the post-update latent so pred_hidden
                    # depends on alpha (mirrors the dense-blockwise pattern
                    # used by mtp_mcmc).
                    post_pred_embeddings = self._logits_to_pred_embeddings(predicted_tokens, mcmc_step)
                    post_all = torch.cat((real_embeddings_input, post_pred_embeddings), dim=1)
                    post_kwargs = dict(trunk_kwargs)
                    post_kwargs["return_pred_hidden"] = True
                    post_out = self.transformer(post_all, **post_kwargs)
                    if isinstance(post_out, tuple) and len(post_out) == 2:
                        _, pred_hidden = post_out
                    else:
                        raise RuntimeError(
                            "Trunk did not return (energies, pred_hidden) for explicit-block-latent path"
                        )
                    predicted_hiddens.append(pred_hidden)

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size)
                predicted_distributions.append(predicted_tokens_for_loss)

        if return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, predicted_hiddens
        return predicted_tokens, predicted_distributions, predicted_energies

    def forward_explicit_block_latent_hidden(self, input_ids, block_size, no_randomness):
        """Training-time pred-hidden extraction for the new modes.

        Returns:
            pred_hiddens_per_step: list[Tensor] each [B, S*K, D] in
                position-major order.
            predicted_energies: list[Tensor] each [B*S*K, 1].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S], got shape {tuple(input_ids.shape)}")
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "forward_explicit_block_latent_hidden only valid for block_mode in "
                f"{EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        S = int(input_ids.shape[1])

        real_embeddings_input = self.embeddings(input_ids)  # [B, S, D]
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        # Initialize S*K latents. Position-major flattening matches the trunk
        # mask / freq-index helpers in utils.py.
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=S * K)

        _, _, predicted_energies, pred_hiddens_per_step = self._run_explicit_block_latent_mcmc(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            block_size=K,
            start_pos=0,
            learning=True,
            return_raw_logits=True,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )
        return pred_hiddens_per_step, predicted_energies

    def forward_explicit_block_latent_logits(self, input_ids, block_size, no_randomness, return_hidden=False):
        """Training-time per-MCMC-step logits for the new modes.

        Returns:
            logits_per_step: list[Tensor] each [B, K, S, V] aligned with
                ``block_targets [B, K, S]`` (so the last axis is the vocab
                axis). Each latent z_{t,j} is mapped through
                ``block_latent_token_head`` independently.
            predicted_energies: list[Tensor] each [B*S*K, 1].
            pred_hiddens_per_step (only if return_hidden=True): list of
                [B, S*K, D].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S], got shape {tuple(input_ids.shape)}")
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        S = int(input_ids.shape[1])

        pred_hiddens_per_step, predicted_energies = self.forward_explicit_block_latent_hidden(
            input_ids=input_ids,
            block_size=K,
            no_randomness=no_randomness,
        )

        logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            logits = self._explicit_block_latent_pred_hidden_to_logits(pred_hidden, S=S, K=K)
            logits_per_step.append(logits)

        if return_hidden:
            return logits_per_step, predicted_energies, pred_hiddens_per_step
        return logits_per_step, predicted_energies

    def _forward_explicit_block_latent_inference(
        self,
        real_embeddings_input,
        block_size,
        start_pos=0,
        learning=False,
        return_raw_logits=False,
        no_randomness=True,
    ):
        """Inference-time forward for new modes via the C+K trunk layout.

        Builds K future latents anchored at the end of context, runs MCMC,
        and produces per-step predicted distributions of shape ``[B, K, V]``
        (one prediction per latent), matching the legacy ``(predicted_distributions,
        predicted_energies)`` return shape used by ``call_model_forward_decode``.

        After MCMC, the K latent hidden states are mapped through the single-
        token head ``block_latent_token_head`` to yield the *real* per-latent
        logits (since MCMC operates on the prob-dist surrogate, the trunk's
        pred_hidden is the natural output for downstream sampling).
        """
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        if real_embeddings_input.dim() != 3:
            raise ValueError(
                f"Expected real_embeddings_input [B, C, D], got shape {tuple(real_embeddings_input.shape)}"
            )

        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        # Initialize exactly K latents (one per future block offset).
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=K)

        _, _, predicted_energies, pred_hiddens_per_step = self._run_explicit_block_latent_mcmc(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            block_size=K,
            start_pos=start_pos,
            learning=learning,
            return_raw_logits=return_raw_logits,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )

        # Map each step's [B, K, D] pred_hidden through the single-token head
        # to produce [B, K, V] logits. These are the inference logits used
        # by sampling code; they are consistent with training where each
        # latent is supervised by exactly one target token.
        logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            if pred_hidden.shape[1] != K:
                raise RuntimeError(
                    f"pred_hidden length mismatch in inference: expected K={K}, got {pred_hidden.shape[1]}"
                )
            logits_per_step.append(self.block_latent_token_head(pred_hidden))  # [B, K, V]

        return logits_per_step, predicted_energies

    def ebt_refine_block_explicit_block_latent(
        self,
        context_ids,
        draft_block_ids,
        refine_steps=None,
        init_logit_scale=8.0,
        start_pos=0,
        learning=False,
    ):
        """Block refinement entry point for the new explicit-block-latent
        modes. Mirrors :meth:`ebt_refine_block_fast` but uses the C+K trunk
        layout instead of the symmetric ``[ctx, draft[:-1]]`` prefix trick.

        Inputs:
            context_ids:     [B, C]  context token ids (real, not optimized)
            draft_block_ids: [B, K]  draft token ids used to seed the K
                                     future latents.

        Returns:
            refined_block_logits: [B, K, V]
            refined_block_ids:    [B, K]
        """
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "ebt_refine_block_explicit_block_latent only valid for block_mode in "
                f"{EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        if draft_block_ids.shape[1] == 0:
            empty_logits = torch.empty(
                draft_block_ids.shape[0], 0, self.vocab_size,
                device=draft_block_ids.device, dtype=self.embeddings.weight.dtype,
            )
            return empty_logits, draft_block_ids
        if context_ids.shape[1] == 0:
            raise ValueError("ebt_refine_block_explicit_block_latent requires at least one context token")

        K = int(draft_block_ids.shape[1])
        real_embeddings_input = self.embeddings(context_ids)  # [B, C, D]
        # Seed K latents from the draft tokens: peaked logits then run MCMC.
        block_logits = self._draft_block_ids_to_initial_logits(
            draft_block_ids, init_logit_scale=init_logit_scale,
        ).detach()
        init_block_logits = block_logits.detach().clone()

        alpha = torch.clamp(self.alpha, min=0.0001)
        noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        requested_steps = int(refine_steps) if refine_steps is not None else int(self.hparams.mcmc_num_steps)
        if requested_steps <= 0:
            refined_block_ids = torch.argmax(block_logits, dim=-1)
            return block_logits, refined_block_ids
        effective_steps = min(requested_steps, int(self.hparams.mcmc_num_steps))
        mcmc_steps = list(range(effective_steps))
        diagnose = bool(getattr(self.hparams, "infer_block_diagnose", False))

        def _block_energy_mean(logits, step_idx):
            with torch.no_grad():
                block_embeds = self._logits_to_pred_embeddings(logits, step_idx)
                all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=block_embeds.shape[1],
                    block_size=K,
                    block_mode=self._block_mode,
                )
                return energy_preds.mean().item()

        initial_energy_mean = _block_energy_mean(init_block_logits, mcmc_steps[0])
        last_grad_norm = 0.0

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                block_logits = block_logits.detach().requires_grad_()
                cur_block_logits = block_logits
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(cur_block_logits.detach()) * noise_std
                    cur_block_logits = cur_block_logits + ld_noise

                block_embeds = self._logits_to_pred_embeddings(cur_block_logits, mcmc_step)
                all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=block_embeds.shape[1],
                    block_size=K,
                    block_mode=self._block_mode,
                )
                energy_block = energy_preds.reshape(-1, 1)

                block_grad = torch.autograd.grad(
                    [energy_block.sum()], [cur_block_logits], create_graph=learning
                )[0]
                last_grad_norm = block_grad.norm().item()
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    block_grad = torch.clamp(block_grad, min=-min_and_max, max=min_and_max)
                if torch.isnan(block_grad).any() or torch.isinf(block_grad).any():
                    raise ValueError("NaN or Inf gradients detected during explicit-block-latent block refinement.")

                block_logits = cur_block_logits - alpha * block_grad
                if self.hparams.absolute_clamp != 0.0:
                    block_logits = torch.clamp(
                        block_logits, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp,
                    )
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    block_logits = block_logits / self.hparams.sharpen_predicted_distribution
                block_logits = block_logits.detach()

        # After MCMC, reconstruct the *real* per-latent token logits from
        # the post-update pred_hidden (consistent with training where each
        # latent is decoded by `block_latent_token_head`).
        with torch.no_grad():
            block_embeds = self._logits_to_pred_embeddings(block_logits, mcmc_steps[-1])
            all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
            _, pred_hidden = self.transformer(
                all_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_steps[-1],
                context_len=real_embeddings_input.shape[1],
                pred_len=block_embeds.shape[1],
                block_size=K,
                block_mode=self._block_mode,
                return_pred_hidden=True,
            )
            refined_block_logits = self.block_latent_token_head(pred_hidden)  # [B, K, V]
            refined_block_ids = torch.argmax(refined_block_logits, dim=-1)

        if diagnose:
            final_energy_mean = _block_energy_mean(block_logits, mcmc_steps[-1])
            delta_norm = (block_logits - init_block_logits).norm().item()
            max_show = min(2, draft_block_ids.shape[0])
            for b in range(max_show):
                print(f"draft_block_ids[{b}]: {draft_block_ids[b].tolist()}", flush=True)
                print(f"refined_block_ids[{b}]: {refined_block_ids[b].tolist()}", flush=True)
            print(f"||refined_logits - init_logits||: {delta_norm:.6f}", flush=True)
            print(f"block initial energy: {initial_energy_mean:.6f}", flush=True)
            print(f"block final energy: {final_energy_mean:.6f}", flush=True)
            print(f"block logits grad.norm(): {last_grad_norm:.6f}", flush=True)

        return refined_block_logits, refined_block_ids

    def _forward_loss_wrapper_explicit_block_latent(self, x, phase, token_bytes, no_randomness):
        """Blockwise loss path for the new explicit-block-latent modes.

        Constraints versus mtp_mcmc:
          * Each future latent z_{t,j} corresponds to exactly one target
            token: ``block_targets[:, j-1, t-1]`` (1-indexed t,j; equivalently
            the (j, t) entry of the [B, K, S] tensor).
          * Logits come from ``block_latent_token_head`` applied to each
            latent independently — there is no joint multi-offset projection.
          * Attention semantics inside the trunk are governed by ``self._block_mode``
            (future_latent_non_causal vs blockwise) via the new mask helpers.

        The mtp_mcmc loss path below is left completely untouched.
        """
        if not isinstance(x, dict):
            raise ValueError("Expected dense blockwise batch to be a dict with keys: input_ids and block_targets")

        def _maybe_squeeze_loader_dim(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
                return tensor.squeeze(dim=0)
            return tensor

        input_ids = _maybe_squeeze_loader_dim(x["input_ids"])
        block_targets = _maybe_squeeze_loader_dim(x["block_targets"])
        target_offsets = _maybe_squeeze_loader_dim(x.get("target_offsets"))

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be 2D [B, S_eff], got shape {tuple(input_ids.shape)}")
        if block_targets.dim() != 3:
            raise ValueError(f"Expected block_targets to be 3D [B, K, S_eff], got shape {tuple(block_targets.shape)}")
        if block_targets.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"Batch mismatch between input_ids and block_targets: {tuple(input_ids.shape)} vs {tuple(block_targets.shape)}"
            )
        if block_targets.shape[2] != input_ids.shape[1]:
            raise ValueError(
                f"S_eff mismatch between input_ids and block_targets: input_ids.shape[1]={input_ids.shape[1]}, "
                f"block_targets.shape[2]={block_targets.shape[2]}"
            )

        num_offsets = int(block_targets.shape[1])  # = K
        S_eff = int(input_ids.shape[1])
        if target_offsets is None:
            target_offsets = torch.arange(1, num_offsets + 1, device=input_ids.device, dtype=torch.long)

        logits_per_step, predicted_energies, pred_hiddens_per_step = self.forward_explicit_block_latent_logits(
            input_ids=input_ids,
            block_size=num_offsets,
            no_randomness=no_randomness,
            return_hidden=True,
        )
        # Flatten targets to match the [B, K, S, V] -> [B*K*S, V] layout.
        next_token_indices = block_targets.reshape(-1)
        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies)
        final_cce_loss = None

        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(logits_per_step, predicted_energies)):
            predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)

            if self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                cce_loss = F.cross_entropy(
                    predicted_distribution,
                    next_token_indices,
                    label_smoothing=label_smoothing,
                    ignore_index=-1,
                )
            else:
                predicted_distribution = self.log_softmax(predicted_distribution)
                cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)

            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = cce_loss
                    final_reconstruction_loss = cce_loss.detach()
                    final_cce_loss = cce_loss.detach()
            else:
                reconstruction_loss += cce_loss
                if mcmc_step == (total_mcmc_steps - 1):
                    final_reconstruction_loss = cce_loss.detach()
                    final_cce_loss = cce_loss.detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps

            if mcmc_step == 0:
                initial_loss = cce_loss.detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()

        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        ppl_loss = torch.exp(final_reconstruction_loss).detach()
        total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
        contrastive_loss = 0.0

        if token_bytes is not None:
            bpb_loss = calculate_bpb_score(next_token_indices, final_cce_loss, token_bytes)
        else:
            bpb_loss = 0

        # Per-offset loss logging, computed on the final MCMC step's logits
        # in the canonical [B, K, S, V] layout. This is the same logging
        # format mtp_mcmc uses, so downstream dashboards keep working.
        offset_loss_log_dict = {}
        final_step_logits = logits_per_step[-1].detach()       # [B, K, S, V]
        final_step_targets = block_targets.detach()            # [B, K, S]
        for offset_idx in range(num_offsets):
            offset_value = int(target_offsets[offset_idx].item())
            offset_logits = final_step_logits[:, offset_idx, :, :].reshape(-1, self.vocab_size)
            offset_targets = final_step_targets[:, offset_idx, :].reshape(-1)
            offset_loss = F.cross_entropy(
                offset_logits,
                offset_targets,
                ignore_index=-1,
            )
            offset_loss_log_dict[f"offset_{offset_value}_loss"] = offset_loss

        if getattr(self.hparams, "debug_blockwise_shapes", False):
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"explicit_block_latent=True, input_ids.shape={tuple(input_ids.shape)}, "
                f"block_targets.shape={tuple(block_targets.shape)}, "
                f"offsets={target_offsets.tolist()}, "
                f"logits_last_step.shape={tuple(logits_per_step[-1].shape)}, "
                f"alpha={self.alpha.detach().item():.6f}",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"post_update_pred_hidden_last_step.shape={tuple(pred_hiddens_per_step[-1].shape)} "
                f"(position-major flatten of [B, S_eff={S_eff}, K={num_offsets}, D])",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"aggregated_loss={total_loss.detach().item():.6f}",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] offset_losses="
                f"{ {k: round(v.item(), 6) for k, v in offset_loss_log_dict.items()} }",
                flush=True,
            )

        log_dict = {
            'loss': total_loss,
            'initial_loss': initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss': contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss,
        }
        log_dict.update(offset_loss_log_dict)
        return log_dict

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None):
        no_randomness = False if phase == "train" else True
        training_objective = getattr(self.hparams, "training_objective", "dense_next_token")

        if training_objective == "blockwise":
            if self.mcmc_replay_buffer:
                raise NotImplementedError("mcmc_replay_buffer is not supported for training_objective=blockwise")
            if self.hparams.contrastive_loss:
                raise NotImplementedError("contrastive_loss is not supported for training_objective=blockwise")
            if self.hparams.execution_mode == "finetune":
                raise NotImplementedError("execution_mode=finetune is not supported for training_objective=blockwise yet")

            # block_mode dispatch within the blockwise training_objective:
            #   * mtp_mcmc        -> existing dense-blockwise + joint head path
            #     (kept byte-identical below).
            #   * future_latent_non_causal / blockwise -> dedicated path that
            #     uses block_latent_token_head and the K-future-latents-per-
            #     source-position semantic (see
            #     forward_explicit_block_latent_logits above).
            if self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
                return self._forward_loss_wrapper_explicit_block_latent(
                    x=x, phase=phase, token_bytes=token_bytes, no_randomness=no_randomness,
                )

            if not isinstance(x, dict):
                raise ValueError("Expected dense blockwise batch to be a dict with keys: input_ids and block_targets")

            def _maybe_squeeze_loader_dim(tensor):
                if tensor is None:
                    return None
                if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
                    return tensor.squeeze(dim=0)
                return tensor

            input_ids = _maybe_squeeze_loader_dim(x["input_ids"])
            block_targets = _maybe_squeeze_loader_dim(x["block_targets"])
            target_offsets = _maybe_squeeze_loader_dim(x.get("target_offsets"))

            if input_ids.dim() != 2:
                raise ValueError(f"Expected input_ids to be 2D [B, S_eff], got shape {tuple(input_ids.shape)}")
            if block_targets.dim() != 3:
                raise ValueError(f"Expected block_targets to be 3D [B, K, S_eff], got shape {tuple(block_targets.shape)}")
            if block_targets.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    f"Batch mismatch between input_ids and block_targets: {tuple(input_ids.shape)} vs {tuple(block_targets.shape)}"
                )
            if block_targets.shape[2] != input_ids.shape[1]:
                raise ValueError(
                    f"S_eff mismatch between input_ids and block_targets: input_ids.shape[1]={input_ids.shape[1]}, block_targets.shape[2]={block_targets.shape[2]}"
                )

            num_offsets = int(block_targets.shape[1])
            seq_eff = int(input_ids.shape[1])
            if target_offsets is None:
                target_offsets = torch.arange(1, num_offsets + 1, device=input_ids.device, dtype=torch.long)

            multi_offset_logits_per_step, predicted_energies, pred_hiddens_per_step = self.forward_blockwise_dense_logits(
                input_ids,
                num_offsets=num_offsets,
                no_randomness=no_randomness,
                return_hidden=True,
            )
            next_token_indices = block_targets.reshape(-1)
            reconstruction_loss = 0
            total_mcmc_steps = len(predicted_energies)
            final_cce_loss = None

            for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(multi_offset_logits_per_step, predicted_energies)):
                predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)

                if self.hparams.soften_target_prob_dist != 0.0:
                    if total_mcmc_steps <= 1:
                        label_smoothing = 0.0
                    else:
                        label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                    cce_loss = F.cross_entropy(
                        predicted_distribution,
                        next_token_indices,
                        label_smoothing=label_smoothing,
                        ignore_index=-1,
                    )
                else:
                    predicted_distribution = self.log_softmax(predicted_distribution)
                    cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)

                if self.hparams.truncate_mcmc:
                    if mcmc_step == (total_mcmc_steps - 1):
                        reconstruction_loss = cce_loss
                        final_reconstruction_loss = cce_loss.detach()
                        final_cce_loss = cce_loss.detach()
                else:
                    reconstruction_loss += cce_loss
                    if mcmc_step == (total_mcmc_steps - 1):
                        final_reconstruction_loss = cce_loss.detach()
                        final_cce_loss = cce_loss.detach()
                        reconstruction_loss = reconstruction_loss / total_mcmc_steps

                if mcmc_step == 0:
                    initial_loss = cce_loss.detach()
                    initial_pred_energies = predicted_energy.squeeze().mean().detach()
                if mcmc_step == (total_mcmc_steps - 1):
                    final_pred_energies = predicted_energy.squeeze().mean().detach()

            initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
            ppl_loss = torch.exp(final_reconstruction_loss).detach()
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0

            if token_bytes is not None:
                bpb_loss = calculate_bpb_score(next_token_indices, final_cce_loss, token_bytes)
            else:
                bpb_loss = 0

            offset_loss_log_dict = {}
            final_step_logits = multi_offset_logits_per_step[-1].detach()  # [B, K, S_eff, V]
            final_step_targets = block_targets.detach()  # [B, K, S_eff]
            for offset_idx in range(num_offsets):
                offset_value = int(target_offsets[offset_idx].item())
                offset_logits = final_step_logits[:, offset_idx, :, :].reshape(-1, self.vocab_size)
                offset_targets = final_step_targets[:, offset_idx, :].reshape(-1)
                offset_loss = F.cross_entropy(
                    offset_logits,
                    offset_targets,
                    ignore_index=-1,
                )
                offset_loss_log_dict[f"offset_{offset_value}_loss"] = offset_loss

            if getattr(self.hparams, "debug_blockwise_shapes", False):
                print(
                    f"[blockwise-debug][phase={phase}] dense_mode=True, "
                    f"input_ids.shape={tuple(input_ids.shape)}, block_targets.shape={tuple(block_targets.shape)}, "
                    f"offsets={target_offsets.tolist()}, logits_last_step.shape={tuple(multi_offset_logits_per_step[-1].shape)}, "
                    f"alpha={self.alpha.detach().item():.6f}",
                    flush=True,
                )
                print(
                    f"[blockwise-debug][phase={phase}] post_update_pred_hidden_last_step.shape={tuple(pred_hiddens_per_step[-1].shape)}",
                    flush=True,
                )
                if num_offsets >= 1:
                    print(
                        f"[blockwise-debug][phase={phase}] offset_1_logits.shape={tuple(multi_offset_logits_per_step[-1][:, 0, :, :].shape)}",
                        flush=True,
                    )
                if num_offsets >= 2:
                    print(
                        f"[blockwise-debug][phase={phase}] offset_2_logits.shape={tuple(multi_offset_logits_per_step[-1][:, 1, :, :].shape)}",
                        flush=True,
                    )
                print(
                    f"[blockwise-debug][phase={phase}] aggregated_loss={total_loss.detach().item():.6f}",
                    flush=True,
                )
                print(
                    f"[blockwise-debug][phase={phase}] offset_losses="
                    f"{ {k: round(v.item(), 6) for k, v in offset_loss_log_dict.items()} }",
                    flush=True,
                )

            log_dict = {
                'loss': total_loss,
                'initial_loss' : initial_loss,
                'final_step_loss': final_reconstruction_loss,
                'contrastive_loss' : contrastive_loss,
                'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
                'perplexity': ppl_loss,
                'bpb': bpb_loss
            }
            log_dict.update(offset_loss_log_dict)
            return log_dict
        else:
            if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
                # all_tokens = x['input_ids'].squeeze(dim=1)
                all_tokens = x[0].squeeze(dim=0)
                input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens) # this automatically does indexing for input ids and next token indices while also passing back the logits
                predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, replay_buffer_logits = replay_buffer_logits, no_randomness = no_randomness)
                self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach()) # update using the final predicted distributions
            else:
                input_ids = x[0].squeeze(dim=0)
                next_token_indices = x[1].squeeze(dim=0)
                predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)

                # input_ids = x['input_ids'].squeeze(dim=1)[:, :-1]
                # predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)
                # next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # squeeze was to remove 1 on 2nd dim

            if self.hparams.execution_mode == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
                next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
            next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies) # in general this equals self.hparams.mcmc_num_steps, isnt in case of rand number
        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, predicted_energies)):
            if self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)
                cce_loss = F.cross_entropy(predicted_distribution, next_token_indices, label_smoothing=label_smoothing, ignore_index=-1)
            else:
                predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
                cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)
            
            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = cce_loss
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
            else:
                reconstruction_loss += cce_loss
                if mcmc_step == (total_mcmc_steps - 1):
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps # normalize so is indifferent to number of mcmc steps
                
            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_loss = cce_loss.detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()
        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies

        if self.hparams.contrastive_loss: # works by pushing up on energies model predicted and pushing down on energy of true samples
            contrastive_loss = self.calculate_contrastive_loss(predicted_energies, input_ids, next_token_indices)
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.contrastive_loss_coeff * contrastive_loss
            contrastive_loss = contrastive_loss.detach()
        else:
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0
        
        if token_bytes is not None:
            bpb_loss = calculate_bpb_score(next_token_indices, cce_loss.detach(), token_bytes)
        else:
            bpb_loss = 0

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss' : contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss  # Renamed from 'bpb_loss' for cleaner logging
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings, target_length=None):
        if target_length is None:
            target_length = embeddings.shape[1]
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(size=(embeddings.shape[0], target_length, self.vocab_size), device = self.device) * self.hparams.gaussian_random_noise_scaling
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(size=(embeddings.shape[0], target_length, self.vocab_size), device = self.device)
        else:
            raise NotImplementedError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        
        return predicted_tokens
    
    def calculate_contrastive_loss(self, predicted_energies, input_ids, next_token_indices):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        real_embeddings_input = self.embeddings(input_ids)
        
        next_token_indices_2d = next_token_indices.reshape(batch_size, seq_length)
        
        if self.hparams.discrete_contrastive_loss_true_logit_val != 0: # NOTE from experience this doesnt work very well and it not recommended compared to just one hot encoding
            true_logit_value = self.hparams.discrete_contrastive_loss_true_logit_val
            false_logit_value = -1 * true_logit_value
            true_token_logits = torch.full((batch_size, seq_length, self.vocab_size), false_logit_value, device=next_token_indices.device)
            
            batch_idx = torch.arange(batch_size, device=next_token_indices.device).view(-1, 1).expand(-1, seq_length)
            seq_idx = torch.arange(seq_length, device=next_token_indices.device).view(1, -1).expand(batch_size, -1)
            true_token_logits[batch_idx, seq_idx, next_token_indices_2d] = true_logit_value
            
            if self.hparams.normalize_initial_condition:
                true_token_logits = self.softmax(true_token_logits)
                    
                if self.hparams.vocab_to_embed_uses_prob_dist:
                    true_embeddings = torch.matmul(true_token_logits, self.embeddings.weight)
                else:
                    true_embeddings = self.vocab_to_embed(true_token_logits)
            else:
                true_embeddings = self.vocab_to_embed(true_token_logits)
        else:
            assert self.hparams.normalize_initial_condition, "if not using normalize initial condition must set logit val"
            true_token_one_hot = torch.zeros((batch_size, seq_length, self.vocab_size), device=next_token_indices.device)
            batch_idx = torch.arange(batch_size, device=next_token_indices.device).view(-1, 1).expand(-1, seq_length)
            seq_idx = torch.arange(seq_length, device=next_token_indices.device).view(1, -1).expand(batch_size, -1)
            true_token_one_hot[batch_idx, seq_idx, next_token_indices_2d] = 1.0
            
            if self.hparams.vocab_to_embed_uses_prob_dist:
                true_embeddings = torch.matmul(true_token_one_hot, self.embeddings.weight)
            else:
                true_embeddings = self.vocab_to_embed(true_token_one_hot)

        all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
        
        real_energies = self.transformer(all_true_embeddings, start_pos=0, mcmc_step=self.hparams.mcmc_num_steps - 1, block_mode=self._block_mode) # NOTE if want to use this maybe check in better detail what ired does
        real_energies = real_energies.reshape(-1, 1) # BS, 1
        fake_energies = predicted_energies[-1] # B*S, 1
        energy_stack = torch.cat([real_energies, fake_energies], dim=1)
        energy_targets = torch.zeros(real_energies.shape[0], dtype=torch.long, device=fake_energies.device)
        padding_positions = None # (next_token_indices == self.tokenizer_pad_token_id).reshape(-1)
        energy_targets[padding_positions] = -100 # prevents nans instead of using self.tokenizer_pad_token_id, as setting this to 0 leads to issues
        contrastive_loss = F.cross_entropy(-1 * energy_stack, energy_targets, ignore_index=-100)
        return contrastive_loss
    
    def warm_up_finished(self):
        if self.hparams.clamp_max_after_warm_up != 0.0:
            print(f"changing clamp value after warming up from {self.hparams.clamp_futures_grad_max_change} (see next line)")
            self.hparams.clamp_futures_grad_max_change = self.hparams.clamp_max_after_warm_up
            print(f"to the value {self.hparams.clamp_futures_grad_max_change}")
        self.finished_warming_up = True
        self.langevin_dynamics_noise_std.requires_grad = self.hparams.langevin_dynamics_noise_learnable


    def ebt_advanced_inference(self, original_real_input_ids, start_pos=0, learning=True): # code was written with help from AI
        real_embeddings_input = self.embeddings(original_real_input_ids)  # (B, S, D)
        original_predicted_tokens = self.corrupt_embeddings(real_embeddings_input)  # (B, S, V)

        alpha = self.alpha * self.hparams.infer_ebt_override_alpha if 0 < self.hparams.infer_ebt_override_alpha < 1 else (
            torch.tensor(self.hparams.infer_ebt_override_alpha, device=self.device) if self.hparams.infer_ebt_override_alpha >= 1 else self.alpha
        )

        noise = (torch.tensor(
            self.hparams.infer_langevin_dynamics_noise,
            dtype=self.langevin_dynamics_noise_std.dtype,
            device=self.langevin_dynamics_noise_std.device
        ) if self.hparams.infer_langevin_dynamics_noise != 0 else self.langevin_dynamics_noise_std)

        B, S, V = original_predicted_tokens.shape
        G = self.hparams.infer_generated_samples

        if G > 1:
            repeated_pred = original_predicted_tokens.repeat_interleave(G, dim=0)
            # Optionally corrupt again so each copy starts differently
            repeated_pred = self.corrupt_embeddings(real_embeddings_input.repeat_interleave(G, dim=0))
            repeated_real_embeds = real_embeddings_input.repeat_interleave(G, dim=0)
            repeated_bs = B * G
        else:
            repeated_pred = original_predicted_tokens
            repeated_real_embeds = real_embeddings_input
            repeated_bs = B

        all_final_pred = torch.zeros_like(repeated_pred)
        energies_list_accum = None
        predicted_distributions_accum = None

        chunk_size = B  # or another chunk size if you prefer
        for start in range(0, repeated_bs, chunk_size):
            end = min(start + chunk_size, repeated_bs)

            chunk_pred = repeated_pred[start:end]           # shape: (chunk_size, S, V)
            chunk_real_embeds = repeated_real_embeds[start:end]  # shape: (chunk_size, S, D)

            final_pred_chunk, energies_list_chunk, predicted_distributions_chunk = self._run_ebt_inference_steps(
                chunk_pred, chunk_real_embeds,
                alpha, noise, start_pos, learning
            )
            all_final_pred[start:end] = final_pred_chunk


            energies_list_chunk = [
                e.reshape(chunk_size, -1) for e in energies_list_chunk
            ]

            if energies_list_accum is None:
                energies_list_accum = [e for e in energies_list_chunk]
                predicted_distributions_accum = [p.detach() for p in predicted_distributions_chunk]
            else:
                for i in range(len(energies_list_accum)):
                    energies_list_accum[i] = torch.cat(
                        [energies_list_accum[i], energies_list_chunk[i]], dim=0
                    )
                for i in range(len(predicted_distributions_accum)):
                    if i < len(predicted_distributions_chunk):
                        predicted_distributions_accum[i] = torch.cat(
                            [predicted_distributions_accum[i], predicted_distributions_chunk[i].detach()], dim=0
                        )
        # energies_list_accum is a list of length total_mcmc_steps, each shape (B*G, S)

        if G > 1:
            final_energies_3d = energies_list_accum[-1].reshape(B, G, S)
            
            if self.hparams.infer_debug_sample_distances: # to print the distances between samples if are generating many. good to know if model's samples are diverse or if should add more noise to initial condition
                all_final_pred_4d = all_final_pred.reshape(B, G, S, V)
                softmaxed_preds = self.softmax(all_final_pred_4d)
                for b in range(min(B, 2)):  # Only show first 2 batches to avoid excessive output
                    for s in range(min(S, 5)):  # Only show first 5 sequence positions
                        print(f"Batch {b}, Seq pos {s} - Sample distances:")
                        for i in range(G):
                            for j in range(i+1, G):
                                p_i = softmaxed_preds[b, i, s]
                                p_j = softmaxed_preds[b, j, s]
                                # KL divergence
                                # Add small value to avoid log(0)
                                kl_div = F.kl_div(
                                    (p_i + 1e-10).log(), 
                                    p_j + 1e-10, 
                                    reduction='sum'
                                )
                                # L2 distance
                                l2_dist = torch.norm(p_i - p_j, p=2)
                                print(f"  Sample {i} vs {j}: KL={kl_div.item():.4f}, L2={l2_dist.item():.4f}")
            if self.hparams.infer_energy_sampling_technique == "min":
                best_indices_2d = final_energies_3d.argmin(dim=1)  # shape: (B, S)
            elif self.hparams.infer_energy_sampling_technique == "max":
                best_indices_2d = final_energies_3d.argmax(dim=1)  # shape: (B, S)
            elif self.hparams.infer_energy_sampling_technique == "max_gap":
                initial_energies_3d = energies_list_accum[0].reshape(B, G, S)
                gap_3d = initial_energies_3d - final_energies_3d
                best_indices_2d = gap_3d.argmax(dim=1)             # shape: (B, S)
            else:
                raise ValueError(f"Unknown infer_energy_sampling_technique: {self.hparams.infer_energy_sampling_technique}")

            all_final_pred_4d = all_final_pred.reshape(B, G, S, V)

            b_arange = torch.arange(B, device=all_final_pred.device).unsqueeze(-1)  # shape: (B, 1)
            s_arange = torch.arange(S, device=all_final_pred.device).unsqueeze(0)   # shape: (1, S)

            
            final_output = all_final_pred_4d[b_arange, best_indices_2d, s_arange, :]
        else:
            final_output = all_final_pred

        # final_output shape (B, S, V), energies_list_accum (at each index for original num_mcmc_steps len) shape (B*G, S)
        return final_output, energies_list_accum, predicted_distributions_accum

    def _run_ebt_inference_steps(
        self,
        initial_pred_tokens,
        real_embeds,
        adjusted_alpha,
        noise,
        start_pos,
        learning
    ):
        energies_list = []
        pred_states_list = []
        pred_states_list.append(initial_pred_tokens)

        def do_mcmc_step(step_idx, cur_pred_tokens, alpha):
            with torch.set_grad_enabled(True):
                cur_pred_tokens = cur_pred_tokens.detach().requires_grad_()

                # Add noise if set
                if not self.hparams.infer_langevin_first_step: # default
                    cur_pred_tokens = cur_pred_tokens + noise * torch.randn_like(cur_pred_tokens)
                else:
                    if step_idx == 0: # only do langevin on first step
                        cur_pred_tokens = cur_pred_tokens + noise * torch.randn_like(cur_pred_tokens)

                # Convert logits -> embeddings
                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if step_idx == 0:
                            cur_pred_tokens = self.softmax(cur_pred_tokens)
                    else:
                        cur_pred_tokens = self.softmax(cur_pred_tokens)
                            
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        pred_embeds = torch.matmul(cur_pred_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        pred_embeds = self.vocab_to_embed(cur_pred_tokens) #BS, S, D
                else:
                    pred_embeds = self.vocab_to_embed(cur_pred_tokens)

                combined_embeddings = torch.cat([real_embeds, pred_embeds], dim=1)  # (chunk_size, 2S, D)
                energies = self.transformer(combined_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
                energies = energies.reshape(-1)
                energies_list.append(energies.detach())

                grad = torch.autograd.grad(energies.sum(), [cur_pred_tokens], create_graph=learning)[0]

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (alpha)
                    grad = torch.clamp(grad, -min_and_max, min_and_max)

                if self.hparams.infer_accept_lower_energies: # have to get energy to determine if should decrease
                    old_energies = energies.reshape(cur_pred_tokens.shape[:2])
                    proposed_tokens = cur_pred_tokens - alpha * grad
                    new_energies = get_energy(step_idx, proposed_tokens).reshape(cur_pred_tokens.shape[:2])
                    accept_mask = (new_energies < old_energies).float().unsqueeze(-1)
                    updated_tokens = accept_mask * proposed_tokens + (1 - accept_mask) * cur_pred_tokens

                else:
                    updated_tokens = cur_pred_tokens - alpha * grad
                return updated_tokens.detach()
            
        def get_energy(step_idx, cur_pred_tokens): # for if just want to get energy of currently predicted tokens
            with torch.no_grad():
                cur_pred_tokens = cur_pred_tokens.detach().requires_grad_()

                # Convert logits -> embeddings
                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if step_idx == 0:
                            cur_pred_tokens = self.softmax(cur_pred_tokens)
                    else:
                        cur_pred_tokens = self.softmax(cur_pred_tokens)
                            
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        pred_embeds = torch.matmul(cur_pred_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        pred_embeds = self.vocab_to_embed(cur_pred_tokens) #BS, S, D
                else:
                    pred_embeds = self.vocab_to_embed(cur_pred_tokens)
                combined_embeddings = torch.cat([real_embeds, pred_embeds], dim=1)  # (chunk_size, 2S, D)
                energies = self.transformer(combined_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
                energies = energies.reshape(-1)
                return energies

        # ebt_type
        if self.hparams.ebt_type == "default":
            total_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps > 1 else self.hparams.mcmc_num_steps
            pred_state = initial_pred_tokens
            for step_idx in range(total_steps):
                pred_state = do_mcmc_step(step_idx, pred_state, adjusted_alpha)
                pred_states_list.append(pred_state)
        else:
            # alternative ebt_type i.e. adaln or time embed
            pred_state = initial_pred_tokens
            for step_idx in range(self.hparams.mcmc_num_steps):
                if self.hparams.infer_steps_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1):
                    alpha = self.alpha if self.hparams.infer_alpha_final_landscape else adjusted_alpha
                    pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                    pred_states_list.append(pred_state)
                else:
                    inner_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps != 1 else (self.hparams.randomize_mcmc_num_steps_min if self.hparams.randomize_mcmc_num_steps_min != 0 else 1)
                    for _ in range(inner_steps):
                        alpha = self.alpha if (self.hparams.infer_alpha_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1)) else adjusted_alpha
                        pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                        pred_states_list.append(pred_state)


        final_pred_state_energies = get_energy((self.hparams.mcmc_num_steps - 1), pred_state)
        energies_list.append(final_pred_state_energies)
        return pred_state, energies_list, pred_states_list
        return pred_state, energies_list, pred_states_list


