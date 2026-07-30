import torch
from torch import nn
from torch.nn import functional as F
from openebm.elm.nanolightning.torchlightning_module import LightningModule
# import torch.optim as optim
# from torchmetrics import Accuracy
# from transformers import AutoTokenizer

import math
import random
import os
from openebm.elm.utils import setup_ebt, init_whole_model_weights
from openebm.elm.utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from openebm.elm.replay_buffer import CausalReplayBuffer
from openebm.elm.metrics import calculate_bpb_score
from openebm.elm.tf_head import build_tf_head

import ipdb

PRECISION_TO_MCMC_DTYPE = {
    "bf16-true": torch.bfloat16,
    "bf16-mixed": torch.float32,
    "16-mixed": torch.float32,
    "32-true": torch.float32,
}

class EBT_NLP(LightningModule):
    @staticmethod
    def _parse_per_step_float_values(raw_value, num_steps, default_value, flag_name):
        if raw_value is None or str(raw_value).strip() == "":
            return [float(default_value)] * num_steps
        values = [item.strip() for item in str(raw_value).split(",")]
        if len(values) != num_steps:
            raise ValueError(f"{flag_name} must have exactly {num_steps} comma-separated values.")
        try:
            return [float(item) for item in values]
        except ValueError as exc:
            raise ValueError(f"{flag_name} must contain only float values.") from exc

    @staticmethod
    def _parse_per_step_bool_mask(raw_value, num_steps, default_value, flag_name):
        if raw_value is None or str(raw_value).strip() == "":
            return [bool(default_value)] * num_steps
        values = [item.strip().lower() for item in str(raw_value).split(",")]
        if len(values) != num_steps:
            raise ValueError(f"{flag_name} must have exactly {num_steps} comma-separated values.")
        mask = []
        for item in values:
            if item in ("1", "true", "t", "yes", "y"):
                mask.append(True)
            elif item in ("0", "false", "f", "no", "n"):
                mask.append(False)
            else:
                raise ValueError(f"{flag_name} entries must be 0/1 or true/false.")
        return mask

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.zero_mcmc_ce_only = bool(getattr(self.hparams, "zero_mcmc_ce_only", False))
        if self.hparams.mcmc_num_steps < 0:
            raise ValueError("--mcmc_num_steps must be >= 0.")
        if self.hparams.mcmc_num_steps == 0 and not self.zero_mcmc_ce_only:
            raise ValueError("--mcmc_num_steps 0 requires --zero_mcmc_ce_only.")
        if self.zero_mcmc_ce_only:
            if self.hparams.mcmc_num_steps != 0:
                raise ValueError("--zero_mcmc_ce_only requires --mcmc_num_steps 0.")
            if self.hparams.mcmc_step_size_learnable:
                raise ValueError("--zero_mcmc_ce_only does not use alpha; omit --mcmc_step_size_learnable.")
            if getattr(self.hparams, "randomize_mcmc_num_steps", 0) != 0:
                raise ValueError("--zero_mcmc_ce_only is incompatible with --randomize_mcmc_num_steps.")
        
        # tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        # Use tokenizer_obj if available (set by ModelTrainer), otherwise use tokenizer directly
        self.tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        self.tokenizer_pad_token_id = None # Nanochat doesn't have <|pad|> or <|eos|> # self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.get_vocab_size() # len(self.tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.hparams.vocab_size = self.vocab_size
        
        if getattr(self.hparams, 'mcmc_step_size_per_step', False):
            if not self.hparams.mcmc_step_size_learnable:
                raise ValueError("--mcmc_step_size_per_step requires --mcmc_step_size_learnable.")
            alpha_values = self._parse_per_step_float_values(
                getattr(self.hparams, 'mcmc_step_size_per_step_values', ""),
                self.hparams.mcmc_num_steps,
                self.hparams.mcmc_step_size,
                "--mcmc_step_size_per_step_values",
            )
            alpha_learnable_mask = self._parse_per_step_bool_mask(
                getattr(self.hparams, 'mcmc_step_size_per_step_learnable_mask', ""),
                self.hparams.mcmc_num_steps,
                self.hparams.mcmc_step_size_learnable,
                "--mcmc_step_size_per_step_learnable_mask",
            )
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=learnable)
                for value, learnable in zip(alpha_values, alpha_learnable_mask)
            ])
        else:
            self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size), dtype=torch.float32),
                                      requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
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

        # TF head sits AFTER MCMC and consumes either the trunk's candidate-position
        # pred_hidden (post-norm / last-layer hidden) or the free-embedding MCMC
        # state to produce final CE logits. Some head variants also use
        # embed(input_ids[t]) as a first-layer teacher-forced anchor; direct
        # unembedding variants intentionally ignore that anchor.
        # Default off; toggled by --use_tf_head. Keeps original EBT CE path intact
        # when off.
        self.use_tf_head = bool(getattr(self.hparams, "use_tf_head", False))
        self.use_pre_update_hidden_unembed = getattr(self.hparams, "tf_head_type", "") == "pre_update_hidden_unembed"
        self.use_post_update_state_unembed = getattr(self.hparams, "tf_head_type", "") == "post_update_state_unembed"
        self.post_update_state_use_rmsnorm = bool(getattr(self.hparams, "post_update_state_use_rmsnorm", False))
        self.post_update_state_concat_zi = bool(getattr(self.hparams, "post_update_state_concat_zi", False))
        self.post_update_state_concat_prev_embed = bool(getattr(self.hparams, "post_update_state_concat_prev_embed", False))
        self.post_update_state_detach_prev_embed = bool(getattr(self.hparams, "post_update_state_detach_prev_embed", False))
        self.truncate_mcmc_per_step_ce = bool(getattr(self.hparams, "truncate_mcmc_per_step_ce", False))
        if self.truncate_mcmc_per_step_ce and not self.hparams.truncate_mcmc:
            raise ValueError("--truncate_mcmc_per_step_ce requires --truncate_mcmc.")
        if (
            self.post_update_state_use_rmsnorm
            or self.post_update_state_concat_zi
            or self.post_update_state_concat_prev_embed
            or self.post_update_state_detach_prev_embed
        ) and not self.use_post_update_state_unembed:
            raise ValueError(
                "post_update_state_* options only apply when "
                "--tf_head_type post_update_state_unembed."
            )
        if self.post_update_state_concat_zi and self.post_update_state_concat_prev_embed:
            raise ValueError(
                "--post_update_state_concat_zi and --post_update_state_concat_prev_embed "
                "are mutually exclusive; use one concat ablation at a time."
            )
        if self.post_update_state_detach_prev_embed and not self.post_update_state_concat_prev_embed:
            raise ValueError(
                "--post_update_state_detach_prev_embed requires "
                "--post_update_state_concat_prev_embed."
            )
        if self.use_tf_head:
            self.tf_head = build_tf_head(self.hparams)
            init_whole_model_weights(
                self.tf_head,
                self.hparams.weight_initialization_method,
                weight_initialization_gain=self.hparams.weight_initialization_gain,
            )

        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        # Free embedding MCMC: iterate in D-dim instead of V-dim. Skips softmax/vocab_to_embed
        # at trunk input. TF head is mandatory in this mode (provides discrete supervision).
        # Must come AFTER self.mcmc_replay_buffer is assigned, since the assertion reads it.
        self.use_free_embedding_mcmc = bool(getattr(self.hparams, "free_embedding_mcmc", False))
        if self.use_free_embedding_mcmc:
            if not self.use_tf_head:
                raise ValueError(
                    "--free_embedding_mcmc requires --use_tf_head: D-dim MCMC iterate has no "
                    "intrinsic V-dim decode path; TF head supplies the discrete CE supervision."
                )
            if self.mcmc_replay_buffer:
                raise ValueError(
                    "--free_embedding_mcmc is incompatible with mcmc_replay_buffer "
                    "(buffer stores V-dim logits)."
                )
        if self.use_tf_head and self.use_post_update_state_unembed and not self.use_free_embedding_mcmc:
            raise ValueError(
                "--tf_head_type post_update_state_unembed requires --free_embedding_mcmc: "
                "the post-update state must be D-dim before Linear(D, V) unembedding."
            )
        if self.zero_mcmc_ce_only:
            if not self.use_tf_head:
                raise ValueError("--zero_mcmc_ce_only requires --use_tf_head for CE supervision.")
            if not self.use_free_embedding_mcmc:
                raise ValueError("--zero_mcmc_ce_only currently supports the free-embedding TF-head path only.")
            if self.use_post_update_state_unembed:
                raise ValueError(
                    "--zero_mcmc_ce_only has no post-update state; use --tf_head_type direct_unembed, "
                    "linear, transformer, or pre_update_hidden_unembed."
                )
            if self.hparams.ebt_type != "time_embed":
                raise ValueError("--zero_mcmc_ce_only currently requires --ebt_type time_embed.")
            if getattr(self.hparams, "use_mcmc_time_embed", False):
                raise ValueError("--zero_mcmc_ce_only is incompatible with --use_mcmc_time_embed.")
            if getattr(self.hparams, "contrastive_loss", False):
                raise ValueError("--zero_mcmc_ce_only is incompatible with --contrastive_loss.")
            if hasattr(self.transformer, "final_layer"):
                for param in self.transformer.final_layer.parameters():
                    param.requires_grad = False

        self._alpha_debug_step = 0  # counter for alpha diagnostic prints
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update

    def _apply(self, fn):
        """Override to ensure alpha always stays in float32 regardless of model dtype.

        _apply is the lowest-level method called by all dtype/device conversions
        (to, cuda, float, etc.). Lightning's to() bypasses nn.Module.to() when a
        trainer is present, so overriding to() is insufficient.
        """
        result = super()._apply(fn)
        if isinstance(result.alpha, nn.ParameterList):
            for alpha_param in result.alpha:
                alpha_param.data = alpha_param.data.to(dtype=torch.float32)
        else:
            result.alpha.data = result.alpha.data.to(dtype=torch.float32)
        return result

    def _logits_to_pred_embeddings(self, predicted_tokens):
        """Convert V-dim predicted_tokens (post-update) to D-dim embeddings for a
        second trunk forward. Always softmaxes when normalize_initial_condition=True
        because post-update logits are not a clean prob dist any more. No Langevin noise.
        Used ONLY by the TF-head pred_hidden extraction path.
        """
        if self.hparams.normalize_initial_condition:
            if getattr(self.hparams, 'float_precision', '') == "bf16-true":
                predicted_tokens = self.softmax(predicted_tokens)
            else:
                predicted_tokens = self.softmax(predicted_tokens.float()).to(predicted_tokens.dtype)
            if self.hparams.vocab_to_embed_uses_prob_dist:
                return torch.matmul(predicted_tokens, self.embeddings.weight)
            return self.vocab_to_embed(predicted_tokens)
        return self.vocab_to_embed(predicted_tokens)

    def _zero_mcmc_ce_only_pred_hidden(self, predicted_tokens, real_embeddings_input, start_pos, real_token_ids):
        if not self.use_free_embedding_mcmc:
            raise RuntimeError("--zero_mcmc_ce_only currently expects D-dim free-embedding state.")

        all_embeddings = torch.cat((real_embeddings_input.detach(), predicted_tokens), dim=1)
        energy_preds, pred_hidden_step = self.transformer(
            all_embeddings,
            start_pos=start_pos,
            mcmc_step=0,
            real_token_ids=real_token_ids,
            predicted_tokens=predicted_tokens,
            return_pred_hidden=True,
            return_energy=False,
        )
        if energy_preds is not None:
            raise RuntimeError("--zero_mcmc_ce_only expected the transformer to skip energy output.")
        return pred_hidden_step

    @torch.compiler.disable
    def _mcmc_step_excluded(self, predicted_tokens, real_embeddings_input, mcmc_step, i, num_mcmc_steps,
                      langevin_dynamics_noise_std, alpha, start_pos, learning, return_raw_logits,
                      real_token_ids=None, compute_pred_hidden=False):
        batch_size = predicted_tokens.shape[0]
        seq_length = predicted_tokens.shape[1]
        
        # predicted_tokens state-dim: V (logit mode) or D (free embedding mode). Use -1
        # so the reshape doesn't break in free mode where state_dim == embedding_dim != vocab_size.
        if self.hparams.no_mcmc_detach:
            predicted_tokens.requires_grad_().reshape(batch_size, seq_length, -1)
        else: # default, do detach
            predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, -1)

        if self.hparams.langevin_dynamics_noise != 0:
            ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
            predicted_tokens = predicted_tokens + ld_noise

        if self.use_free_embedding_mcmc:
            # Free embedding mode: predicted_tokens is already [B, S, D]; no softmax/vocab_to_embed
            # conversion. The MCMC state IS the trunk's input embedding.
            predicted_embeddings = predicted_tokens
        elif self.hparams.normalize_initial_condition:
            if getattr(self.hparams, 'float_precision', '') == "bf16-true":
                if self.hparams.normalize_initial_condition_only_first_step:
                    if mcmc_step == 0:
                        predicted_tokens = self.softmax(predicted_tokens)
                else:
                    predicted_tokens = self.softmax(predicted_tokens)
            else:
                if self.hparams.normalize_initial_condition_only_first_step:
                    if mcmc_step == 0:
                        predicted_tokens = self.softmax(predicted_tokens.float()).to(predicted_tokens.dtype)
                else:
                    predicted_tokens = self.softmax(predicted_tokens.float()).to(predicted_tokens.dtype)

            if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
            else:
                predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        else:
            predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D

        all_embeddings = torch.cat((real_embeddings_input.detach(), predicted_embeddings), dim = 1) # B, 2*S, D
        
        # create_graph=True 的 autograd.grad 与 compiled graph 不兼容
        # 所以若 transformer 已被 torch.compile 编译，MCMC 中需要用 eager 版本
        transformer = getattr(self, 'transformer_eager', self.transformer)
        # First trunk forward: always produces energy. pre_update_hidden_unembed
        # also consumes this same forward's candidate hidden for CE. The post-update
        # state variant waits until after the energy-gradient update below.
        pred_hidden_step = None
        pre_update_state_for_head = None
        if compute_pred_hidden and self.use_post_update_state_unembed and self.post_update_state_concat_zi:
            pre_update_state_for_head = predicted_tokens
        use_pre_update_hidden = compute_pred_hidden and self.use_pre_update_hidden_unembed
        if use_pre_update_hidden:
            energy_preds, pred_hidden_step = transformer(
                all_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                real_token_ids=real_token_ids,
                predicted_tokens=predicted_tokens,
                return_pred_hidden=True,
            )
        else:
            energy_preds = transformer(
                all_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                real_token_ids=real_token_ids,
                predicted_tokens=predicted_tokens,
            ) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
        energy_preds = energy_preds.reshape(-1, 1)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            energy_f32 = energy_preds.float()
            if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                if i == (num_mcmc_steps - 1):
                    create_graph_step = learning
                else:
                    create_graph_step = False
            else:
                create_graph_step = learning
            retain_graph_step = create_graph_step or (learning and use_pre_update_hidden)
            predicted_tokens_grad = torch.autograd.grad(
                [energy_f32.sum()],
                [predicted_tokens],
                create_graph=create_graph_step,
                retain_graph=retain_graph_step,
            )[0]
        # predicted_tokens_grad has shape B, S, V
        
        if self.hparams.clamp_futures_grad:
            _alpha_for_clamp = alpha if alpha is not None else (
                self.alpha[mcmc_step] if isinstance(self.alpha, nn.ParameterList) else self.alpha
            )
            min_and_max = self.hparams.clamp_futures_grad_max_change / torch.clamp(_alpha_for_clamp.float(), min=0.0001)
            # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
            predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
            
        if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
            raise ValueError("NaN or Inf gradients detected during MCMC.")
        
        predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after
        
        if self.hparams.absolute_clamp != 0.0:
            predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
        
        if self.hparams.sharpen_predicted_distribution != 0.0:
            predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

        # post_update_state_unembed decodes the updated free-embedding state z_{i+1}
        # directly, so it also skips the post-update second trunk forward.
        if compute_pred_hidden and self.use_post_update_state_unembed:
            if self.post_update_state_concat_zi:
                pred_hidden_step = (predicted_tokens, pre_update_state_for_head)
            else:
                pred_hidden_step = predicted_tokens

        # Second trunk forward — POST-update — to extract pred_hidden for the
        # default TF-head path. pre_update_hidden_unembed uses pred_hidden_step from
        # the energy forward above; post_update_state_unembed uses z_{i+1} directly.
        if compute_pred_hidden and pred_hidden_step is None:
            if self.use_free_embedding_mcmc:
                # post-update predicted_tokens IS already D-dim — feed straight to trunk.
                post_pred_embeddings = predicted_tokens
            else:
                post_pred_embeddings = self._logits_to_pred_embeddings(predicted_tokens)
            post_all_embeddings = torch.cat((real_embeddings_input.detach(), post_pred_embeddings), dim=1)
            _, pred_hidden_step = transformer(
                post_all_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                real_token_ids=real_token_ids,
                predicted_tokens=predicted_tokens,
                return_pred_hidden=True,
            )

        if self.use_free_embedding_mcmc:
            # No V-dim baseline CE in free mode; TF head supplies the supervised logits
            # in forward_loss_wrapper. Returning None here is fine — the list slot gets
            # replaced wholesale before the CE loop.
            predicted_tokens_for_loss = None
        elif return_raw_logits:
            predicted_tokens_for_loss = predicted_tokens # BS, S, V
        else:
            predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V

        return predicted_tokens, energy_preds, predicted_tokens_for_loss, pred_hidden_step

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True, return_pred_hiddens = False): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        predicted_distributions = []
        predicted_energies = []
        predicted_pred_hiddens = []  # populated only if return_pred_hiddens=True (TF head path)

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        model_dtype = self.embeddings.weight.dtype
        if not isinstance(self.alpha, nn.ParameterList):
            alpha = torch.clamp(self.alpha, min=0.0001).float()
            if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
                expanded_alpha = alpha.expand(batch_size, seq_length, 1)

                scale = self.hparams.randomize_mcmc_step_size_scale
                low = alpha / scale
                high = alpha * scale
                alpha = low + torch.rand_like(expanded_alpha) * (high - low)
        else:
            alpha = None  # will be resolved per step below

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V
        if replay_buffer_logits is not None: # using replay buffer, use the logits instead of corruption
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits # NOTE this assumes the fresh data is concatted first

        if self.zero_mcmc_ce_only:
            if replay_buffer_logits is not None:
                raise RuntimeError("--zero_mcmc_ce_only is incompatible with replay_buffer_logits.")
            pred_hidden_step = self._zero_mcmc_ce_only_pred_hidden(
                predicted_tokens,
                real_embeddings_input,
                start_pos,
                real_token_ids=x,
            )
            zero_energy = pred_hidden_step.new_zeros(batch_size * seq_length, 1)
            if return_pred_hiddens:
                return [None], [zero_energy], [pred_hidden_step]
            return [None], [zero_energy]


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

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                compute_pred_hidden = return_pred_hiddens
                if return_pred_hiddens and self.hparams.truncate_mcmc and not self.truncate_mcmc_per_step_ce:
                    compute_pred_hidden = i == (len(mcmc_steps) - 1)
                step_alpha = alpha
                if isinstance(self.alpha, nn.ParameterList):
                    step_alpha = torch.clamp(self.alpha[mcmc_step], min=0.0001).float()
                    if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
                        expanded_alpha = step_alpha.expand(batch_size, seq_length, 1)
                        scale = self.hparams.randomize_mcmc_step_size_scale
                        low = step_alpha / scale
                        high = step_alpha * scale
                        step_alpha = low + torch.rand_like(expanded_alpha) * (high - low)

                predicted_tokens, energy_preds, predicted_tokens_for_loss, pred_hidden_step = self._mcmc_step_excluded(
                    predicted_tokens, real_embeddings_input, mcmc_step, i, len(mcmc_steps),
                    langevin_dynamics_noise_std, step_alpha, start_pos, learning, return_raw_logits,
                    real_token_ids=x, compute_pred_hidden=compute_pred_hidden,
                )
                if self.hparams.contrastive_loss:
                    predicted_energies.append(energy_preds)
                else:
                    predicted_energies.append(energy_preds.detach())
                predicted_distributions.append(predicted_tokens_for_loss)
                if return_pred_hiddens:
                    predicted_pred_hiddens.append(pred_hidden_step)
                del energy_preds, predicted_tokens_for_loss, pred_hidden_step  # release references to help GC

        if return_pred_hiddens:
            return predicted_distributions, predicted_energies, predicted_pred_hiddens
        return predicted_distributions, predicted_energies

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None, global_step=None):
        no_randomness = False if phase == "train" else True
        learning = phase == "train"
        predicted_pred_hiddens = None  # set only on TF-head path
        # v3: optional 3rd batch element = per-token loss weights (B, T) aligned with targets.
        # Used by the sudoku v2 dataset for blank-position loss weighting (P1). When None
        # (or a normal 2-tuple batch), behavior matches v2 byte-for-byte.
        token_loss_weights = None
        if isinstance(x, (list, tuple)) and len(x) >= 3:
            tw = x[2]
            if isinstance(tw, torch.Tensor):
                # squeeze to match input/target shape
                token_loss_weights = tw.squeeze(dim=0) if tw.dim() == 3 else tw
        if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
            # all_tokens = x['input_ids'].squeeze(dim=1)
            all_tokens = x[0].squeeze(dim=0)
            input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens) # this automatically does indexing for input ids and next token indices while also passing back the logits
            if self.use_tf_head:
                predicted_distributions, predicted_energies, predicted_pred_hiddens = self(
                    input_ids, return_raw_logits=True, replay_buffer_logits=replay_buffer_logits,
                    no_randomness=no_randomness, learning=learning, return_pred_hiddens=True,
                )
            else:
                predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, replay_buffer_logits = replay_buffer_logits, no_randomness = no_randomness, learning=learning)
            # Replay buffer is seeded with MCMC's raw distribution (NOT the TF-head logits) so
            # buffer semantics stay consistent across baseline/TF runs.
            self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach()) # update using the final predicted distributions
        else:
            input_ids = x[0].squeeze(dim=0)
            next_token_indices = x[1].squeeze(dim=0)
            if self.use_tf_head:
                predicted_distributions, predicted_energies, predicted_pred_hiddens = self(
                    input_ids, return_raw_logits=True, no_randomness=no_randomness, learning=learning, return_pred_hiddens=True,
                )
            else:
                predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness, learning=learning)

            # input_ids = x['input_ids'].squeeze(dim=1)[:, :-1]
            # predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)
            # next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # squeeze was to remove 1 on 2nd dim

        if self.use_tf_head:
            # Replace per-step distributions with TF-head outputs. Most variants
            # consume trunk post-norm candidate hidden; post_update_state_unembed
            # consumes the post-update free-embedding MCMC state instead.
            # prev_embed is only the first-layer token embedding; direct unembedding
            # variants keep the shared call signature but do not use it.
            prev_embed = self.embeddings(input_ids)  # first-layer anchor, [B, S, D]
            if self.post_update_state_detach_prev_embed:
                prev_embed = prev_embed.detach()
            predicted_distributions = [
                None if pred_hidden_step is None else self.tf_head(pred_hidden_step, prev_embed)  # [B, S, V]
                for pred_hidden_step in predicted_pred_hiddens
            ]

        dataset_name = getattr(self.hparams, 'dataset_name', '')
        if self.hparams.execution_mode == "finetune" and dataset_name != "nanochat_sft":
            # Legacy SFT datasets use the "[[Answer]]:" text marker.
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
        supervised_tokens = (next_token_indices != -1).sum()
        # Empty SFT rows still continue through BPB/DDP reductions; early return
        # can deadlock mixed empty/non-empty ranks.
        supervised_tokens_denom = supervised_tokens.clamp_min(1)
        empty_supervision_batch = (supervised_tokens == 0).to(dtype=torch.int64)

        # v3: flatten weights to (BS*S,) to match next_token_indices.
        flat_weights = None
        if token_loss_weights is not None:
            weight_dtype = next(
                (pred.dtype for pred in predicted_distributions if pred is not None),
                torch.float32,
            )
            flat_weights = token_loss_weights.reshape(-1).to(
                device=next_token_indices.device,
                dtype=weight_dtype,
            )
            # Zero-out positions corresponding to ignored targets so they don't bias the mean.
            flat_weights = flat_weights * (next_token_indices != -1).to(flat_weights.dtype)

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies) # in general this equals self.hparams.mcmc_num_steps, isnt in case of rand number
        initial_loss = None
        initial_pred_energies = None
        final_pred_energies = None
        final_reconstruction_loss = None
        reconstruction_loss_steps = 0
        ppl_loss = None
        final_predicted_distribution = None
        final_label_smoothing = 0.0
        final_step_only_loss = self.hparams.truncate_mcmc and not self.truncate_mcmc_per_step_ce
        for mcmc_step, predicted_energy in enumerate(predicted_energies):
            if mcmc_step == 0:
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()

            predicted_distribution = predicted_distributions[mcmc_step]
            if predicted_distribution is None:
                if final_step_only_loss and mcmc_step != (total_mcmc_steps - 1):
                    continue
                raise RuntimeError("Missing predicted distribution for a supervised MCMC step.")

            label_smoothing = 0.0
            if self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)
                if flat_weights is not None:
                    # v3: weighted CE on flat positions. We compute per-token CE then take a
                    # weight-normalized mean. ignore_index=-1 is preserved by zeroed weights.
                    per_token = F.cross_entropy(
                        predicted_distribution, next_token_indices,
                        label_smoothing=label_smoothing, ignore_index=-1,
                        reduction='none',
                    )
                    w_sum = flat_weights.sum().clamp_min(1.0)
                    cce_loss = (per_token * flat_weights).sum() / w_sum
                else:
                    cce_loss = F.cross_entropy(predicted_distribution, next_token_indices, label_smoothing=label_smoothing, ignore_index=-1)
            else:
                predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
                if flat_weights is not None:
                    per_token = F.nll_loss(
                        predicted_distribution, next_token_indices,
                        ignore_index=-1, reduction='none',
                    )
                    w_sum = flat_weights.sum().clamp_min(1.0)
                    cce_loss = (per_token * flat_weights).sum() / w_sum
                else:
                    cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)
            
            if final_step_only_loss:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = cce_loss
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
                    final_predicted_distribution = predicted_distribution
                    final_label_smoothing = label_smoothing
            else:
                reconstruction_loss += cce_loss
                reconstruction_loss_steps += 1
                if mcmc_step == (total_mcmc_steps - 1):
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
                    final_predicted_distribution = predicted_distribution
                    final_label_smoothing = label_smoothing
                
            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_loss = cce_loss.detach()

        if final_reconstruction_loss is None or ppl_loss is None or final_predicted_distribution is None:
            raise RuntimeError("No final-step reconstruction loss was computed.")
        if not final_step_only_loss:
            if reconstruction_loss_steps == 0:
                raise RuntimeError("No supervised MCMC-step reconstruction loss was computed.")
            reconstruction_loss = reconstruction_loss / reconstruction_loss_steps
        if initial_loss is None:
            initial_loss = final_reconstruction_loss
        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies

        if self.hparams.contrastive_loss: # works by pushing up on energies model predicted and pushing down on energy of true samples
            contrastive_loss = self.calculate_contrastive_loss(predicted_energies, input_ids, next_token_indices)
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.contrastive_loss_coeff * contrastive_loss
            contrastive_loss = contrastive_loss.detach()
        else:
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0
        
        if token_bytes is not None:
            # Compute per-token loss (reduction='none') for accurate BPB.
            # Passing scalar mean loss (cce_loss) is incorrect because
            # calculate_bpb_score multiplies loss by (num_bytes > 0) mask and sums,
            # yielding mean_loss × count rather than sum(per_token_loss[valid]).
            # predicted_distribution from the last MCMC step is already (-1, vocab_size).
            if self.hparams.soften_target_prob_dist != 0.0:
                per_token_ce = F.cross_entropy(final_predicted_distribution, next_token_indices,
                                                label_smoothing=final_label_smoothing, ignore_index=-1, reduction='none')
            else:
                per_token_ce = F.nll_loss(final_predicted_distribution, next_token_indices, ignore_index=-1, reduction='none')
            bpb_loss, bpb_nats, bpb_bytes = calculate_bpb_score(next_token_indices, per_token_ce.detach(), token_bytes)
        else:
            bpb_loss = 0
            bpb_nats = 0
            bpb_bytes = 0

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss' : contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss,
            'bpb_nats': bpb_nats,    # accumulated nats for epoch-level BPB
            'bpb_bytes': bpb_bytes,  # accumulated bytes for epoch-level BPB
            'supervised_tokens': supervised_tokens.detach(),
            'empty_supervision_batch': empty_supervision_batch.detach(),
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings):
        float_precision = getattr(self.hparams, 'float_precision', '')
        mcmc_dtype = PRECISION_TO_MCMC_DTYPE.get(float_precision, torch.float32)

        # State dim depends on MCMC space: V-dim (logit) or D-dim (free embedding).
        if self.use_free_embedding_mcmc:
            state_dim = self.hparams.embedding_dim
            # Doc §4 noise rescaling — extra multiplier for free embedding training noise.
            noise_scale = self.hparams.gaussian_random_noise_scaling * float(
                getattr(self.hparams, "free_embed_noise_scale", 1.0)
            )
        else:
            state_dim = self.vocab_size
            noise_scale = self.hparams.gaussian_random_noise_scaling

        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(size=(embeddings.shape[0], embeddings.shape[1], state_dim), dtype=mcmc_dtype, device=self.device) * noise_scale
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(size=(embeddings.shape[0], embeddings.shape[1], state_dim), dtype=mcmc_dtype, device = self.device)
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

        true_pred_tokens = true_token_logits if self.hparams.discrete_contrastive_loss_true_logit_val != 0 else true_token_one_hot
        real_energies = self.transformer(
            all_true_embeddings,
            start_pos=0,
            mcmc_step=self.hparams.mcmc_num_steps - 1,
            real_token_ids=input_ids,
            predicted_tokens=true_pred_tokens,
        ) # NOTE if want to use this maybe check in better detail what ired does
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

        alpha = None if isinstance(self.alpha, nn.ParameterList) else (
            self.alpha * self.hparams.infer_ebt_override_alpha if 0 < self.hparams.infer_ebt_override_alpha < 1 else (
                torch.tensor(self.hparams.infer_ebt_override_alpha, device=self.device) if self.hparams.infer_ebt_override_alpha >= 1 else self.alpha
            )
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
            chunk_real_ids = original_real_input_ids.repeat_interleave(G, dim=0)[start:end] if G > 1 else original_real_input_ids[start:end]

            final_pred_chunk, energies_list_chunk, predicted_distributions_chunk = self._run_ebt_inference_steps(
                chunk_pred, chunk_real_embeds,
                alpha, noise, start_pos, learning,
                real_token_ids=chunk_real_ids
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
        learning,
        real_token_ids=None
    ):
        energies_list = []
        pred_states_list = []
        pred_states_list.append(initial_pred_tokens)

        def get_alpha_for_step(step_idx, use_model_alpha=False):
            if isinstance(self.alpha, nn.ParameterList):
                base_alpha = self.alpha[min(step_idx, len(self.alpha) - 1)]
                override_alpha = self.hparams.infer_ebt_override_alpha
                if not use_model_alpha and override_alpha >= 1:
                    step_alpha = torch.tensor(override_alpha, device=self.device)
                elif not use_model_alpha and 0 < override_alpha < 1:
                    step_alpha = base_alpha * override_alpha
                else:
                    step_alpha = base_alpha
            elif use_model_alpha:
                step_alpha = self.alpha
            else:
                step_alpha = adjusted_alpha
            return torch.clamp(step_alpha, min=0.0001).float()

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
                energies = self.transformer(
                    combined_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    real_token_ids=real_token_ids,
                    predicted_tokens=cur_pred_tokens,
                )
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
                energies = self.transformer(
                    combined_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    real_token_ids=real_token_ids,
                    predicted_tokens=cur_pred_tokens,
                )
                energies = energies.reshape(-1)
                return energies

        # ebt_type
        if self.hparams.ebt_type == "default" or (self.hparams.ebt_type == "time_embed" and not getattr(self.hparams, 'use_mcmc_time_embed', False)):
            # default mode or time_embed without time embedding: shared transition kernel, arbitrary step count
            total_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps > 1 else self.hparams.mcmc_num_steps
            pred_state = initial_pred_tokens
            for step_idx in range(total_steps):
                pred_state = do_mcmc_step(step_idx, pred_state, get_alpha_for_step(step_idx))
                pred_states_list.append(pred_state)
        else:
            # alternative ebt_type i.e. adaln or time embed
            pred_state = initial_pred_tokens
            for step_idx in range(self.hparams.mcmc_num_steps):
                if self.hparams.infer_steps_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1):
                    alpha = get_alpha_for_step(step_idx, use_model_alpha=self.hparams.infer_alpha_final_landscape)
                    pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                    pred_states_list.append(pred_state)
                else:
                    inner_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps != 1 else (self.hparams.randomize_mcmc_num_steps_min if self.hparams.randomize_mcmc_num_steps_min != 0 else 1)
                    for _ in range(inner_steps):
                        alpha = get_alpha_for_step(
                            step_idx,
                            use_model_alpha=self.hparams.infer_alpha_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1),
                        )
                        pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                        pred_states_list.append(pred_state)


        final_pred_state_energies = get_energy((self.hparams.mcmc_num_steps - 1), pred_state)
        energies_list.append(final_pred_state_energies)
        return pred_state, energies_list, pred_states_list
