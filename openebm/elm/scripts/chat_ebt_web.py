#!/usr/bin/env python3
"""
EBT Web 对话服务 - 通过网页端与训练好的 EBT 模型进行交互式对话

基于 chat_ebt.py 的 EBTChatEngine 核心引擎，封装为 FastAPI Web 服务。
完整保留原有 /xx 命令功能，并在网页端提供流式对话体验。

启动方法:
    bash runs/chat_ebt_web.sh                           # 默认参数
    bash runs/chat_ebt_web.sh --show-mcmc               # 展示 MCMC 步骤
    bash runs/chat_ebt_web.sh --port 8080               # 指定端口

Endpoints:
    GET  /              - Chat UI (内嵌 HTML)
    POST /chat/completions - 流式对话 API (SSE)
    POST /command       - 执行 /xx 命令
    GET  /health        - 健康检查
    GET  /status        - 当前引擎状态
"""

import argparse
import json
import os
import sys
import time
import asyncio
import logging
import threading
import torch
from contextlib import asynccontextmanager, nullcontext
from typing import Optional, List, Dict, Any, AsyncGenerator

from openebm.elm.generate import call_model_forward_decode, _get_tokenizer, sample_top_p
from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper

# 清除分布式训练环境变量
for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
    if var in os.environ:
        del os.environ[var]

os.environ['NANOCHAT_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"

# ── 参数解析 ──
parser = argparse.ArgumentParser(description='EBT Web Chat Server')
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Checkpoint 路径')
parser.add_argument('--tokenizer', type=str,
                    default="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer",
                    help='Tokenizer 路径')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='默认温度')
parser.add_argument('--top-p', type=float, default=0.9, help='默认 Top-P')
parser.add_argument('--max-tokens', type=int, default=512, help='默认最大 tokens')
parser.add_argument('--show-mcmc', action='store_true', help='展示 MCMC 步骤过程')
parser.add_argument('--verbose', action='store_true', help='详细模式')
parser.add_argument('--show-energy', action='store_true', help='展示能量值变化')
parser.add_argument('--show-distribution', action='store_true', help='展示概率分布变化')
parser.add_argument('--override-mcmc-steps', type=int, default=None)
parser.add_argument('--override-noise-std', type=float, default=None)
parser.add_argument('--override-alpha', type=float, default=None)
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device', type=str, default='cuda', help='设备')
parser.add_argument('--port', type=int, default=8000, help='服务端口')
parser.add_argument('--load-workers', type=int, default=0,
                    help='内置 GPU 压测 worker 数（直接调引擎，不走 HTTP；建议 = GPU数*2）')
parser.add_argument('--load-reserve', type=int, default=1,
                    help='为真实用户保留的 GPU 数（压测最多占用 num_gpus - load_reserve 个引擎，默认 1）')
parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定地址')
parser.add_argument('--num-gpus', type=int, default=1, help='并发模型实例数（每实例占一张 GPU）')
parser.add_argument('--batch-size', type=int, default=4,
                    help='每 GPU 批处理并发请求数（bsz，建议 4-8，越大 GPU 利用率越高）')
parser.add_argument('--batch-wait-ms', type=float, default=50.0,
                    help='批次收集窗口（毫秒），窗口内请求凑满 batch-size 则立即发射')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EBTChatEngine - 直接复用 chat_ebt.py 的核心引擎（内联以避免循环 import）
# ══════════════════════════════════════════════════════════════════════════════

class EBTChatEngine:
    """EBT 对话引擎 (来自 chat_ebt.py)"""

    def __init__(self, checkpoint_path, tokenizer_path, device="cuda", dtype=torch.bfloat16,
                 show_mcmc=False, verbose=False, show_energy=False, show_distribution=False,
                 override_mcmc_steps=None, override_noise_std=None, override_alpha=None):
        self.device = device
        self.dtype = dtype
        self.show_mcmc = show_mcmc
        self.verbose = verbose
        self.show_energy = show_energy
        self.show_distribution = show_distribution
        self.override_mcmc_steps = override_mcmc_steps
        self.override_noise_std = override_noise_std
        self.override_alpha = override_alpha
        self.model = None
        self.tokenizer = None
        self.hparams = None
        self._load_model(checkpoint_path, tokenizer_path)

    def _load_model(self, checkpoint_path, tokenizer_path):
        print(f"[WebServer] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'hyper_parameters' in checkpoint:
            self.hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            self.hparams = checkpoint['hparams']
        else:
            raise ValueError("Cannot find hyperparameters in checkpoint")

        class HParamsNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        if isinstance(self.hparams, dict):
            self.hparams = HParamsNamespace(self.hparams)

        resolved_tokenizer_path = os.path.abspath(tokenizer_path) if tokenizer_path else None
        if resolved_tokenizer_path and os.path.isdir(resolved_tokenizer_path):
            os.environ['NANOCHAT_BASE_DIR'] = os.path.dirname(resolved_tokenizer_path.rstrip('/'))
            self.hparams.tokenizer_path = resolved_tokenizer_path
            self.hparams.tokenizer = resolved_tokenizer_path
            self.tokenizer = NanoChatTokenizerWrapper(tokenizer_dir=resolved_tokenizer_path)
            print(f"  Using NanoChat tokenizer from CLI path: {resolved_tokenizer_path}")
        else:
            if resolved_tokenizer_path:
                print(f"  Tokenizer path not found, falling back to checkpoint hparams: {resolved_tokenizer_path}")
            self.tokenizer = _get_tokenizer(self.hparams)

        vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else len(self.tokenizer)
        print(f"  Raw tokenizer vocab size: {vocab_size}")

        if not hasattr(self.tokenizer, 'get_vocab_size'):
            if hasattr(self.tokenizer, 'tokenizer_obj') and hasattr(self.tokenizer.tokenizer_obj, 'get_vocab_size'):
                self.tokenizer.get_vocab_size = self.tokenizer.tokenizer_obj.get_vocab_size
            else:
                self.tokenizer.get_vocab_size = lambda: len(self.tokenizer)

        self.hparams.tokenizer_obj = self.tokenizer

        # setup_ebt() requires use_ve; older Lightning ckpts may omit it in hyper_parameters.
        if not hasattr(self.hparams, "use_ve"):
            raw_sd = checkpoint.get("state_dict", checkpoint)
            self.hparams.use_ve = any("value_embeds" in k for k in raw_sd.keys())

        from openebm.elm.modeling_ebt import EBT_NLP
        self.model = EBT_NLP(self.hparams)

        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith('model.'):
                new_key = new_key[6:]
            if new_key.startswith('_orig_mod.'):
                new_key = new_key[10:]
            if '._orig_mod.' in new_key:
                new_key = new_key.replace('._orig_mod.', '.')
            if new_key.startswith('transformer_eager.'):
                new_key = 'transformer.' + new_key[len('transformer_eager.'):]
            new_state_dict[new_key] = v

        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print("  ✓ 权重加载成功 (strict=True)")
        except Exception as e:
            print(f"  ⚠ strict 加载失败, 回退 strict=False: {e}")
            self.model.load_state_dict(new_state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        self._apply_overrides()
        print("✓ 模型加载完成")

    def _apply_overrides(self):
        if self.override_mcmc_steps is not None:
            original_steps = getattr(self.hparams, 'mcmc_num_steps', 2)
            if self.override_mcmc_steps > original_steps:
                extra = self.override_mcmc_steps - original_steps
                self.hparams.randomize_mcmc_num_steps = extra
                self.model.hparams.randomize_mcmc_num_steps = extra
                self.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.model.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.hparams.randomize_mcmc_num_steps_min = extra + 1
                self.model.hparams.randomize_mcmc_num_steps_min = extra + 1
                trained_noise = getattr(self.hparams, 'langevin_dynamics_noise', 0.0)
                if self.override_noise_std is None and trained_noise == 0:
                    auto_noise = 0.0
                    self.hparams.langevin_dynamics_noise = auto_noise
                    self.model.hparams.langevin_dynamics_noise = auto_noise
                    if hasattr(self.model, 'langevin_dynamics_noise_std'):
                        self.model.langevin_dynamics_noise_std.data.fill_(auto_noise)
            elif self.override_mcmc_steps < original_steps:
                self.hparams.mcmc_num_steps = self.override_mcmc_steps
                self.model.hparams.mcmc_num_steps = self.override_mcmc_steps

        if self.override_noise_std is not None and hasattr(self.model, 'langevin_dynamics_noise_std'):
            self.model.langevin_dynamics_noise_std.data.fill_(self.override_noise_std)
            self.hparams.langevin_dynamics_noise = self.override_noise_std
            self.model.hparams.langevin_dynamics_noise = self.override_noise_std

        if self.override_alpha is not None and hasattr(self.model, 'alpha'):
            self.model.alpha = torch.tensor(self.override_alpha, dtype=self.model.alpha.dtype, device=self.model.alpha.device)

    def get_model_info(self) -> dict:
        embed_dim = getattr(self.hparams, 'embedding_dim', getattr(self.hparams, 'dim', 'N/A'))
        n_layers = getattr(self.hparams, 'num_layers', getattr(self.hparams, 'n_layers', 'N/A'))
        n_heads = getattr(self.hparams, 'num_heads', getattr(self.hparams, 'n_heads', 'N/A'))
        mcmc_steps = getattr(self.hparams, 'mcmc_num_steps', 'N/A')
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 'N/A'))
        alpha_val = 'N/A'
        if hasattr(self.model, 'alpha'):
            alpha_val = round(self.model.alpha.item(), 6) if isinstance(self.model.alpha, torch.Tensor) else self.model.alpha
        noise_val = 'N/A'
        if hasattr(self.model, 'langevin_dynamics_noise_std'):
            noise_val = round(self.model.langevin_dynamics_noise_std.item(), 6) if isinstance(self.model.langevin_dynamics_noise_std, torch.Tensor) else self.model.langevin_dynamics_noise_std
        return {
            "embedding_dim": embed_dim, "num_layers": n_layers, "num_heads": n_heads,
            "mcmc_steps": mcmc_steps, "alpha": alpha_val, "noise_std": noise_val,
            "context_length": ctx_len, "show_mcmc": self.show_mcmc,
            "verbose": self.verbose, "show_energy": self.show_energy,
        }

    def generate_stream(self, prompt: str, max_tokens: int = 512,
                        temperature: float = 0.8, top_p: float = 0.9):
        """生成器: 逐 token yield, 用于流式输出"""
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)
        if inner_tok is not None and hasattr(inner_tok, 'render_for_completion'):
            conversation = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ]
            }
            prompt_tokens_list = inner_tok.render_for_completion(conversation)
        elif inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            bos_id = inner_tok.get_bos_token_id()
            user_start = inner_tok.encode_special("<|user_start|>")
            user_end = inner_tok.encode_special("<|user_end|>")
            asst_start = inner_tok.encode_special("<|assistant_start|>")
            content_ids = inner_tok.encode(prompt)
            prompt_tokens_list = [bos_id, user_start] + content_ids + [user_end, asst_start]
        else:
            encoded = self.tokenizer.encode(prompt)
            prompt_tokens_list = encoded if isinstance(encoded, list) else encoded.tolist()
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            if bos_id is not None and (not prompt_tokens_list or prompt_tokens_list[0] != bos_id):
                prompt_tokens_list = [bos_id] + prompt_tokens_list

        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            pad_id = self.tokenizer.bos_token_id
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = 0

        bsz = 1
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 2048))
        total_len = min(ctx_len, max_tokens + len(prompt_tokens_list))

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        tokens[0, :len(prompt_tokens_list)] = torch.tensor(prompt_tokens_list, dtype=torch.long, device=self.device)

        input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=self.device)
        input_text_mask[0, :len(prompt_tokens_list)] = True

        # Stop tokens
        # Stop once the assistant ends, or if the model tries to open a new user turn.
        stop_token_ids = set()
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            for special in ("<|assistant_end|>", "<|assistant_start|>", "<|user_start|>", "<|user_end|>"):
                sid = inner_tok.encode_special(special)
                if sid is not None:
                    stop_token_ids.add(sid)
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)

        with torch.no_grad():
            if len(prompt_tokens_list) == total_len:
                call_model_forward_decode(self.hparams, self.model, tokens, prev_pos, bsz)

            for cur_pos in range(len(prompt_tokens_list), total_len):
                input_tokens = tokens[:, :cur_pos]
                logits = call_model_forward_decode(self.hparams, self.model, input_tokens, prev_pos, bsz)

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token

                # EOS check
                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= (next_token == sid)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                if all(eos_reached):
                    break

                # Yield decoded token
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                if token_text:
                    yield token_text

    def generate_stream_multi_turn(self, messages: list, max_tokens: int = 512,
                                   temperature: float = 0.8, top_p: float = 0.9):
        """多轮对话的流式生成"""
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)

        if inner_tok is not None and hasattr(inner_tok, 'render_for_completion'):
            conversation = {"messages": messages + [{"role": "assistant", "content": ""}]}
            prompt_tokens_list = inner_tok.render_for_completion(conversation)
        elif inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            bos_id = inner_tok.get_bos_token_id()
            user_start = inner_tok.encode_special("<|user_start|>")
            user_end = inner_tok.encode_special("<|user_end|>")
            asst_start = inner_tok.encode_special("<|assistant_start|>")
            asst_end = inner_tok.encode_special("<|assistant_end|>")

            prompt_tokens_list = [bos_id]
            for msg in messages:
                if msg["role"] == "user":
                    prompt_tokens_list.append(user_start)
                    prompt_tokens_list.extend(inner_tok.encode(msg["content"]))
                    prompt_tokens_list.append(user_end)
                elif msg["role"] == "assistant":
                    prompt_tokens_list.append(asst_start)
                    prompt_tokens_list.extend(inner_tok.encode(msg["content"]))
                    prompt_tokens_list.append(asst_end)
            prompt_tokens_list.append(asst_start)
        else:
            # Fallback: 只用最后一条 user 消息
            last_user = ""
            for msg in messages:
                if msg["role"] == "user":
                    last_user = msg["content"]
            encoded = self.tokenizer.encode(last_user)
            prompt_tokens_list = encoded if isinstance(encoded, list) else encoded.tolist()
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            if bos_id is not None and (not prompt_tokens_list or prompt_tokens_list[0] != bos_id):
                prompt_tokens_list = [bos_id] + prompt_tokens_list

        # ── 以下与 generate_stream 相同的生成逻辑 ──
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            pad_id = self.tokenizer.bos_token_id
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = 0

        bsz = 1
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 2048))
        total_len = min(ctx_len, max_tokens + len(prompt_tokens_list))

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        tokens[0, :len(prompt_tokens_list)] = torch.tensor(prompt_tokens_list, dtype=torch.long, device=self.device)

        input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=self.device)
        input_text_mask[0, :len(prompt_tokens_list)] = True

        stop_token_ids = set()
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            for special in ("<|assistant_end|>", "<|assistant_start|>", "<|user_start|>", "<|user_end|>"):
                sid = inner_tok.encode_special(special)
                if sid is not None:
                    stop_token_ids.add(sid)
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)

        with torch.no_grad():
            if len(prompt_tokens_list) == total_len:
                call_model_forward_decode(self.hparams, self.model, tokens, prev_pos, bsz)

            for cur_pos in range(len(prompt_tokens_list), total_len):
                input_tokens = tokens[:, :cur_pos]
                logits = call_model_forward_decode(self.hparams, self.model, input_tokens, prev_pos, bsz)

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token

                if cur_pos < len(prompt_tokens_list) + 8:
                    tok_id = next_token.item()
                    tok_raw = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                    logger.info(
                        f"[DEBUG token] pos={cur_pos} token_id={tok_id} raw={repr(tok_raw)}"
                    )

                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= (next_token == sid)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                tok_id, eos_val = next_token.item(), eos_reached[0].item()  # bsz=1，2次sync→不变
                if eos_val:
                    break

                token_text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                if token_text:
                    yield token_text


    def _encode_messages_to_tokens(self, messages: list) -> list:
        """将多轮对话编码为 token ID 列表（从 generate_stream_multi_turn 中提取）。"""
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)
        if inner_tok is not None and hasattr(inner_tok, 'render_for_completion'):
            conversation = {"messages": messages + [{"role": "assistant", "content": ""}]}
            return inner_tok.render_for_completion(conversation)
        elif inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            bos_id = inner_tok.get_bos_token_id()
            user_start = inner_tok.encode_special("<|user_start|>")
            user_end = inner_tok.encode_special("<|user_end|>")
            asst_start = inner_tok.encode_special("<|assistant_start|>")
            asst_end = inner_tok.encode_special("<|assistant_end|>")
            toks = [bos_id]
            for msg in messages:
                if msg["role"] == "user":
                    toks.append(user_start)
                    toks.extend(inner_tok.encode(msg["content"]))
                    toks.append(user_end)
                elif msg["role"] == "assistant":
                    toks.append(asst_start)
                    toks.extend(inner_tok.encode(msg["content"]))
                    toks.append(asst_end)
            toks.append(asst_start)
            return toks
        else:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            encoded = self.tokenizer.encode(last_user)
            toks = encoded if isinstance(encoded, list) else encoded.tolist()
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            if bos_id is not None and (not toks or toks[0] != bos_id):
                toks = [bos_id] + toks
            return toks

    def generate_batch(self, batch_messages: list, max_tokens: int = 512,
                       temperature: float = 0.8, top_p: float = 0.9):
        """
        批量生成：将 bsz 个对话请求打包为一次 forward pass，
        每步 yield List[str]（每个序列产出的 token，EOS 后补空串）。
        使用左填充使所有序列在同一位置开始生成。
        """
        all_prompt_tokens = [self._encode_messages_to_tokens(msgs) for msgs in batch_messages]
        bsz = len(all_prompt_tokens)
        prompt_lens = [len(t) for t in all_prompt_tokens]
        max_prompt_len = max(prompt_lens)

        pad_id = getattr(self.tokenizer, 'bos_token_id', 0) or 0
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 2048))
        total_len = min(ctx_len, max_tokens + max_prompt_len)

        # 左填充：所有序列内容右对齐于 max_prompt_len，统一从该位置开始生成
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=self.device)
        for i, (pt, plen) in enumerate(zip(all_prompt_tokens, prompt_lens)):
            start = max_prompt_len - plen
            tokens[i, start:max_prompt_len] = torch.tensor(pt, dtype=torch.long, device=self.device)
            input_text_mask[i, :max_prompt_len] = True   # 含左填充区域，防止在此生成

        inner_tok = getattr(self.tokenizer, 'tokenizer', None)
        stop_token_ids = set()
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            for special in ("<|assistant_end|>", "<|assistant_start|>", "<|user_start|>", "<|user_end|>"):
                sid = inner_tok.encode_special(special)
                if sid is not None:
                    stop_token_ids.add(sid)
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        prev_pos = 0
        eos_reached = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        stop_ids_t = (torch.tensor(list(stop_token_ids), dtype=torch.long, device=self.device)
                      if stop_token_ids else None)

        with torch.no_grad():
            for cur_pos in range(max_prompt_len, total_len):
                input_tokens = tokens[:, :cur_pos]
                logits = call_model_forward_decode(self.hparams, self.model, input_tokens, prev_pos, bsz)

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token

                if stop_ids_t is not None:
                    is_stop = (next_token.unsqueeze(1) == stop_ids_t).any(dim=1)
                else:
                    is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                # 一次 CPU 传输取得所有需要的信息，避免逐元素 GPU sync
                cpu = torch.stack([next_token.long(), eos_reached.long()]).cpu()  # 1 次 sync
                next_tok_list = cpu[0].tolist()
                eos_list = cpu[1].tolist()

                if all(eos_list):   # 纯 Python，无 GPU sync
                    break

                token_texts = [
                    '' if eos_list[i] else
                    self.tokenizer.decode([next_tok_list[i]], skip_special_tokens=True)
                    for i in range(bsz)
                ]
                yield token_texts


# ══════════════════════════════════════════════════════════════════════════════
# BatchScheduler: 每 GPU 一个，将并发请求合并为单次批量 forward pass
# ══════════════════════════════════════════════════════════════════════════════

class BatchScheduler:
    """
    收集 max_batch 个并发请求（或等候 wait_ms 超时），
    合并为一次 bsz=N 的 generate_batch 调用，
    再将 tokens 路由回各自的 SSE 流。
    """

    def __init__(self, engine: EBTChatEngine, max_batch: int = 4, wait_ms: float = 50.0):
        self._engine = engine
        self._max_batch = max_batch
        self._wait_s = wait_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    @property
    def device(self):
        return self._engine.device

    def start(self):
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def generate(self, messages: list, max_tokens: int,
                       temperature: float, top_p: float):
        """提交一个请求，异步 yield 流式 token。"""
        out_q: asyncio.Queue = asyncio.Queue()
        await self._queue.put((messages, max_tokens, temperature, top_p, out_q))
        while True:
            item = await out_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _collect_batch(self) -> list:
        """等第一个请求到达，再贪心收满或到超时，立即返回不浪费 GPU。"""
        loop = asyncio.get_event_loop()
        first = await self._queue.get()
        batch = [first]
        # 队列里已有足够请求时，直接 get_nowait 收满，跳过 wait_ms
        while len(batch) < self._max_batch:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if len(batch) >= self._max_batch:
            return batch
        # 未凑满时才进入短暂等待窗口
        deadline = loop.time() + self._wait_s
        while len(batch) < self._max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    async def _batch_loop(self):
        """流水线：GPU 执行当前 batch 时，同步预收集下一个 batch，消除 GPU 空闲间隙。"""
        loop = asyncio.get_event_loop()
        # 预收集第一个 batch
        next_collect = asyncio.create_task(self._collect_batch())
        while True:
            batch = await next_collect
            # 立即开始收集下下个 batch（与 GPU 执行并行）
            next_collect = asyncio.create_task(self._collect_batch())
            # 执行当前 batch（阻塞直到 GPU 完成）
            await self._run_batch(batch, loop)

    async def _run_batch(self, batch: list, loop: asyncio.AbstractEventLoop):
        out_qs = [item[4] for item in batch]
        all_messages = [item[0] for item in batch]
        max_tokens = max(item[1] for item in batch)
        temperature = batch[0][2]
        top_p = batch[0][3]

        token_q: asyncio.Queue = asyncio.Queue()

        def run_gen():
            try:
                for token_list in self._engine.generate_batch(
                        all_messages, max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p):
                    loop.call_soon_threadsafe(token_q.put_nowait, token_list)
            except Exception as exc:
                loop.call_soon_threadsafe(token_q.put_nowait, RuntimeError(str(exc)))
            finally:
                loop.call_soon_threadsafe(token_q.put_nowait, None)

        t = threading.Thread(target=run_gen, daemon=True)
        t.start()
        n = len(batch)
        try:
            while True:
                item = await token_q.get()
                if item is None:
                    break
                if isinstance(item, RuntimeError):
                    for q in out_qs:
                        q.put_nowait(item)
                    break
                for i in range(n):
                    if item[i]:
                        out_qs[i].put_nowait(item[i])
        finally:
            for q in out_qs:
                q.put_nowait(None)
            t.join(timeout=30)
        logger.info(f"[BATCH] device={self._engine.device} bsz={n} "
                    f"max_tokens={max_tokens} scheduler_q={self._queue.qsize()}")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Web Server
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

# ── 运行时可调参数 (通过 /command API 修改) ──
runtime_config = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_tokens,
}

# ── 调度器列表: 每 GPU 一个 BatchScheduler，在 lifespan 中初始化 ──
_schedulers: List["BatchScheduler"] = []
_sched_idx: int = 0          # round-robin 计数器（原子更新无竞争，asyncio 单线程）
engine_pool: asyncio.Queue   # 兼容旧代码（load/status 等接口），lifespan 中初始化


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

class CommandRequest(BaseModel):
    command: str


_LOAD_PROMPTS = [
    "请写一首关于春天的七言律诗，要求平仄合律，意境优美。",
    "用费曼技巧解释量子纠缠现象，假设对方是高中生。",
    "写一段 Python 代码，实现归并排序并附带时间复杂度分析。",
    "描述工业革命对19世纪欧洲社会结构的深远影响。",
    "给我三个关于人工智能伦理的核心争论，并分析各方立场。",
    "解释贝叶斯定理，并给出一个医学诊断中的实际应用案例。",
    "写一个简短的科幻故事，主题是'意识上传到数字世界后的困境'。",
    "解释为什么快速排序在实践中比堆排序更快，尽管两者都是 O(n log n)。",
    "讨论气候变化对全球粮食安全的三个主要威胁及可能的应对策略。",
    "用通俗语言解释 Transformer 中注意力机制的核心原理。",
    "写一段对话：一个哲学家和一个物理学家争论'时间是否存在'。",
    "解释 TCP 三次握手的原理，以及为什么不能用两次握手。",
    "描述深度学习中梯度消失问题的原因及常用解决方案。",
    "分析第一次世界大战爆发的深层原因。",
    "解释 HTTPS 中 TLS 握手的完整过程，包括证书验证和密钥交换。",
    "解释卡尔曼滤波的基本原理及其在自动驾驶中的应用。",
    "设计一个微服务架构，说明服务发现和负载均衡如何实现。",
    "解释为什么人类对损失的敏感度高于对等量收益的敏感度（前景理论）。",
    "写一段 C++ 代码实现线程安全的生产者-消费者队列。",
    "解释黎曼猜想的基本内容，以及它为什么如此重要。",
]


# 压测运行门控：set=运行，clear=暂停（所有 worker 阻塞在 wait()）
_load_run_gate: Optional[asyncio.Event] = None


async def _builtin_load_worker(worker_id: int, schedulers: list,
                               stop: asyncio.Event):
    """直接向 BatchScheduler 提交请求的内置压测 worker。
    不走 HTTP，完全绕开代理。请求经调度器聚合后批量执行，自动拉高 GPU 利用率。"""
    import random
    rng = random.Random(worker_id * 7919)
    req_count = 0

    while not stop.is_set():
        await _load_run_gate.wait()
        if stop.is_set():
            break

        prompt = rng.choice(_LOAD_PROMPTS)
        messages = [{"role": "user", "content": prompt}]

        # round-robin 选调度器（与用户请求共享，帮助填满 batch）
        sched = schedulers[worker_id % len(schedulers)]

        t0 = time.time()
        ntok = 0
        try:
            async for tok in sched.generate(messages, max_tokens=512,
                                             temperature=0.8, top_p=0.9):
                ntok += 1
        except Exception as exc:
            logger.warning(f"[LOAD-W{worker_id:02d}] gen error: {exc}")

        elapsed = time.time() - t0
        req_count += 1
        logger.info(f"[LOAD-W{worker_id:02d}] req#{req_count}  {ntok}tok  "
                    f"{elapsed:.1f}s  {ntok/max(elapsed,0.1):.0f}tok/s  "
                    f"sched={sched.device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，每 GPU 一个实例，每实例绑定一个 BatchScheduler。"""
    global _schedulers, engine_pool
    num_gpus = args.num_gpus
    batch_size = args.batch_size
    batch_wait_ms = args.batch_wait_ms
    print("=" * 70)
    print(f"EBT Web Chat Server - 正在初始化 {num_gpus} 个模型实例 "
          f"(batch_size={batch_size}, wait={batch_wait_ms:.0f}ms) ...")
    print("=" * 70)

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    torch.set_float32_matmul_precision('medium')

    engine_pool = asyncio.Queue()   # 兼容 /load/status 接口
    _schedulers = []
    for i in range(num_gpus):
        device = f"cuda:{i}" if args.device == 'cuda' else args.device
        print(f"\n[{i+1}/{num_gpus}] 加载模型到 {device} ...")
        engine = EBTChatEngine(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=device,
            dtype=dtype,
            show_mcmc=args.show_mcmc,
            verbose=args.verbose,
            show_energy=args.show_energy,
            show_distribution=args.show_distribution,
            override_mcmc_steps=args.override_mcmc_steps,
            override_noise_std=args.override_noise_std,
            override_alpha=args.override_alpha,
        )
        engine_pool.put_nowait(engine)  # 兼容接口
        sched = BatchScheduler(engine, max_batch=batch_size, wait_ms=batch_wait_ms)
        _schedulers.append(sched)

    # 保留一个引擎引用供 /status /command 等管理接口使用
    app.state.engine = engine_pool._queue[0]

    # 启动每个调度器的 batch_loop 协程
    for sched in _schedulers:
        sched.start()

    print(f"\n✓ EBT Web Chat Server ready at http://0.0.0.0:{args.port}  "
          f"({num_gpus} GPU × bsz={batch_size})")

    # 内置 GPU 压测 worker
    _load_stop = asyncio.Event()
    _load_tasks = []
    if args.load_workers > 0:
        global _load_run_gate
        _load_run_gate = asyncio.Event()
        _load_run_gate.set()
        print(f"[LOAD] 启动 {args.load_workers} 个内置压测 worker "
              f"→ 通过 BatchScheduler 聚合填满 bsz={batch_size}")
        print(f"[LOAD] 控制接口: POST /load/pause  POST /load/resume  GET /load/status")
        for _w in range(args.load_workers):
            _load_tasks.append(asyncio.create_task(
                _builtin_load_worker(_w, _schedulers, _load_stop)))

    yield

    # 关闭
    if _load_tasks:
        _load_stop.set()
        await asyncio.gather(*_load_tasks, return_exceptions=True)
    for sched in _schedulers:
        await sched.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── GET / ── 内嵌 HTML UI ──
@app.get("/")
async def root():
    return HTMLResponse(content=EBT_CHAT_HTML, media_type="text/html; charset=utf-8")

@app.get("/logo.png")
async def logo_horizontal():
    return FileResponse("/mnt/petrelfs/lixueyan/nar/sudoku_screenshot/上海人工智能实验室logo-横版.png", media_type="image/png")

@app.get("/logo-vertical.png")
async def logo_vertical():
    return FileResponse("/mnt/petrelfs/lixueyan/nar/sudoku_screenshot/上海人工智能实验室logo-竖版.png", media_type="image/png")


# ── POST /chat/completions ── 流式对话 ──
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="至少需要一条消息")

    # 日志
    logger.info("=" * 40)
    for msg in request.messages:
        logger.info(f"[{msg.role.upper()}]: {msg.content}")
    logger.info("-" * 40)

    temp = request.temperature if request.temperature is not None else runtime_config["temperature"]
    top_p = request.top_p if request.top_p is not None else runtime_config["top_p"]
    max_tok = request.max_tokens if request.max_tokens is not None else runtime_config["max_tokens"]

    # Clamp
    temp = max(0.0, min(2.0, temp))
    top_p = max(0.0, min(1.0, top_p))
    max_tok = max(1, min(4096, max_tok))

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    # Round-robin 选 BatchScheduler（各 GPU 轮流承接用户请求）
    global _sched_idx
    sched = _schedulers[_sched_idx % len(_schedulers)]
    _sched_idx += 1

    response_tokens: List[str] = []

    async def stream_sse():
        logger.info(f"[USER→SCHED] device={sched.device} q={sched._queue.qsize()}")
        try:
            async for tok in sched.generate(messages_dicts, max_tokens=max_tok,
                                             temperature=temp, top_p=top_p):
                response_tokens.append(tok)
                yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        full_response = "".join(response_tokens)
        logger.info(f"[ASSISTANT]: {full_response}")
        logger.info("=" * 40)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        stream_sse(),
        media_type="text/event-stream; charset=utf-8",
    )


# ── POST /command ── 处理 /xx 命令 ──
@app.post("/command")
async def handle_command(req: CommandRequest):
    cmd = req.command.strip()
    parts = cmd.split()
    action = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else None
    engine: EBTChatEngine = app.state.engine

    if action in ['/quit', '/exit']:
        return {"result": "quit 命令仅在终端模式下有效，网页端请直接关闭页面。"}

    if action == '/clear':
        return {"result": "对话已清空。", "action": "clear"}

    if action == '/help':
        return {"result": (
            "可用命令:\n"
            "  /temp [值]        - 查看/设置温度 (0.0-2.0)\n"
            "  /topp [值]        - 查看/设置 Top-P (0.0-1.0)\n"
            "  /tokens [值]      - 查看/设置最大 tokens (1-4096)\n"
            "  /mcmc             - 切换 MCMC 显示\n"
            "  /verbose          - 切换详细模式\n"
            "  /energy           - 切换能量显示\n"
            "  /status           - 显示当前设置\n"
            "  /info             - 显示模型信息\n"
            "  /clear            - 清空对话历史\n"
            "  /help             - 显示此帮助"
        )}

    if action == '/temp' or action == '/temperature':
        if arg is None:
            return {"result": f"当前温度: {runtime_config['temperature']}"}
        try:
            val = float(arg)
            if 0.0 <= val <= 2.0:
                runtime_config['temperature'] = val
                return {"result": f"✓ 温度已设置为: {val}"}
            return {"result": "✗ 温度必须在 0.0-2.0 之间", "error": True}
        except ValueError:
            return {"result": "✗ 无效的温度值", "error": True}

    if action == '/topp':
        if arg is None:
            return {"result": f"当前 Top-P: {runtime_config['top_p']}"}
        try:
            val = float(arg)
            if 0.0 <= val <= 1.0:
                runtime_config['top_p'] = val
                return {"result": f"✓ Top-P 已设置为: {val}"}
            return {"result": "✗ Top-P 必须在 0.0-1.0 之间", "error": True}
        except ValueError:
            return {"result": "✗ 无效的 Top-P 值", "error": True}

    if action == '/tokens':
        if arg is None:
            return {"result": f"当前最大 Tokens: {runtime_config['max_tokens']}"}
        try:
            val = int(arg)
            if 1 <= val <= 4096:
                runtime_config['max_tokens'] = val
                return {"result": f"✓ 最大 Tokens 已设置为: {val}"}
            return {"result": "✗ 最大 Tokens 必须在 1-4096 之间", "error": True}
        except ValueError:
            return {"result": "✗ 无效的 Tokens 值", "error": True}

    if action == '/mcmc':
        engine.show_mcmc = not engine.show_mcmc
        return {"result": f"✓ MCMC 显示已{'开启' if engine.show_mcmc else '关闭'}"}

    if action == '/verbose':
        engine.verbose = not engine.verbose
        return {"result": f"✓ 详细模式已{'开启' if engine.verbose else '关闭'}"}

    if action == '/energy':
        engine.show_energy = not engine.show_energy
        return {"result": f"✓ 能量显示已{'开启' if engine.show_energy else '关闭'}"}

    if action == '/status':
        info = engine.get_model_info()
        return {"result": (
            f"当前设置:\n"
            f"  温度: {runtime_config['temperature']}\n"
            f"  Top-P: {runtime_config['top_p']}\n"
            f"  最大 Tokens: {runtime_config['max_tokens']}\n"
            f"  显示 MCMC: {'是' if info['show_mcmc'] else '否'}\n"
            f"  详细模式: {'是' if info['verbose'] else '否'}\n"
            f"  显示能量: {'是' if info['show_energy'] else '否'}"
        )}

    if action == '/info':
        info = engine.get_model_info()
        return {"result": (
            f"模型配置:\n"
            f"  嵌入维度: {info['embedding_dim']}\n"
            f"  层数: {info['num_layers']}\n"
            f"  注意力头数: {info['num_heads']}\n"
            f"  MCMC 步数: {info['mcmc_steps']}\n"
            f"  MCMC 步长 (alpha): {info['alpha']}\n"
            f"  Langevin 噪声: {info['noise_std']}\n"
            f"  上下文长度: {info['context_length']}"
        )}

    return {"result": f"未知命令: {action}。输入 /help 查看所有命令。", "error": True}


# ── GET /health ──
@app.get("/health")
async def health():
    engine = getattr(app.state, 'engine', None)
    return {
        "status": "ok",
        "ready": engine is not None,
        "device": args.device,
    }


# ── GET /status ──
@app.get("/status")
async def status():
    engine: EBTChatEngine = app.state.engine
    info = engine.get_model_info()
    info.update(runtime_config)
    return info


# ── 压测开关接口 ──
@app.api_route("/load/pause", methods=["GET", "POST"])
async def load_pause():
    if _load_run_gate is None:
        return {"load": "not_configured"}
    _load_run_gate.clear()
    logger.info("[LOAD] 压测已暂停")
    return {"load": "paused"}


@app.api_route("/load/resume", methods=["GET", "POST"])
async def load_resume():
    if _load_run_gate is None:
        return {"load": "not_configured"}
    _load_run_gate.set()
    logger.info("[LOAD] 压测已恢复")
    return {"load": "running"}


@app.api_route("/load/status", methods=["GET", "POST"])
async def load_status():
    if _load_run_gate is None:
        return {"load": "not_configured"}
    return {
        "load": "running" if _load_run_gate.is_set() else "paused",
        "schedulers": len(_schedulers),
        "batch_size": args.batch_size,
        "pending_per_sched": [s._queue.qsize() for s in _schedulers],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 内嵌 HTML UI
# ══════════════════════════════════════════════════════════════════════════════

EBT_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>EBT Chat</title>
    <style>
        :root { color-scheme: light; }
        * { box-sizing: border-box; }
        html, body { height: 100%; margin: 0; }
        body {
            font-family: ui-sans-serif, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
            background: #fff; color: #111827;
            min-height: 100dvh; display: flex; flex-direction: column;
        }

        /* ── Header ── */
        .header {
            background: #fff; padding: 0.7rem 1.5rem;
            display: flex; align-items: center; justify-content: space-between;
            border-bottom: 1px solid #eef1f6;
            position: sticky; top: 0; z-index: 10;
        }
        .header-left { display: flex; align-items: center; gap: 0.75rem; }
        .header h1 { font-size: 1.05rem; font-weight: 600; margin: 0; color: #111827; }
        .header-tag {
            font-size: 0.67rem; background: #f0f1ff; color: #5b57d1;
            padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-weight: 500; letter-spacing: 0.01em;
        }
        .header-right { display: flex; align-items: center; gap: 0.75rem; }
        .icon-btn {
            width: 30px; height: 30px; padding: 0; border: 1px solid #ebebeb;
            border-radius: 0.45rem; background: #fff; color: #6b7280; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.15s;
        }
        .icon-btn:hover { background: #f7f8fa; border-color: #d1d5db; color: #374151; }
        .header-logo { height: 30px; object-fit: contain; opacity: 0.9; }

        /* ── Chat area ── */
        .chat-container { flex: 1; overflow-y: auto; background: #fff; }
        .chat-wrapper {
            max-width: 780px; margin: 0 auto; padding: 2rem 1.5rem 2rem;
            display: flex; flex-direction: column; gap: 1.5rem;
        }

        /* ── Empty state ── */
        .empty-state {
            display: flex; flex-direction: column; align-items: center;
            padding: 3rem 1rem 2rem; text-align: center;
        }
        .empty-avatar {
            width: 72px; height: 72px; border-radius: 50%; overflow: hidden;
            border: 2px solid #e5e7eb; margin-bottom: 1.25rem;
            background: #fff;
        }
        .empty-avatar img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 15%; }
        .empty-state h2 { font-size: 1.4rem; font-weight: 600; margin: 0 0 0.4rem; }
        .empty-state p { color: #6b7280; font-size: 0.9rem; margin: 0 0 2rem; max-width: 420px; line-height: 1.6; }
        .example-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; width: 100%; max-width: 560px; }
        .example-card {
            background: #fafafa; border: 1px solid #ebebeb; border-radius: 0.75rem;
            padding: 0.85rem 1rem; cursor: pointer; text-align: left;
            transition: border-color 0.15s, box-shadow 0.15s, background 0.15s; font-size: 0.85rem;
        }
        .example-card:hover { background: #fff; border-color: #b8b6ff; box-shadow: 0 2px 10px rgba(79,70,229,0.07); }
        .example-card .card-icon { font-size: 1.1rem; margin-bottom: 0.3rem; }
        .example-card .card-text { color: #374151; line-height: 1.4; }

        /* ── Messages ── */
        .message { display: flex; gap: 0.75rem; }
        .message.user { justify-content: flex-end; }
        .message.assistant { justify-content: flex-start; align-items: flex-start; }
        .message.console { justify-content: flex-start; }

        .msg-avatar {
            width: 32px; height: 32px; border-radius: 50%; overflow: hidden;
            flex-shrink: 0; border: 1px solid #e5e7eb; background: #fff; margin-top: 2px;
        }
        .msg-avatar img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 15%; }

        .msg-body { display: flex; flex-direction: column; gap: 0.3rem; max-width: 82%; }
        .message.user .msg-body { align-items: flex-end; max-width: 72%; }

        .message-content {
            white-space: pre-wrap; line-height: 1.75; font-size: 0.95rem;
        }
        .message.assistant .message-content {
            color: #111827; padding: 0; background: transparent;
        }
        .message.user .message-content {
            background: #f4f4f6; border: 1px solid #ebebeb;
            border-radius: 1.1rem 1.1rem 0.25rem 1.1rem;
            padding: 0.7rem 1rem; color: #111827; cursor: pointer;
            transition: background 0.15s;
        }
        .message.user .message-content:hover { background: #edeef2; }
        .message.console .message-content {
            font-family: 'Monaco','Menlo','Consolas','Courier New', monospace;
            font-size: 0.82rem; background: #f8fafc; border: 1px solid #e2e8f0;
            padding: 0.75rem 1rem; color: #374151; border-radius: 0.5rem;
        }

        /* ── Message actions (copy / regenerate) ── */
        .msg-actions {
            display: flex; gap: 0.25rem; opacity: 0; transition: opacity 0.15s;
        }
        .msg-body:hover .msg-actions { opacity: 1; }
        .action-btn {
            padding: 0.2rem 0.4rem; border: none; background: transparent;
            color: #9ca3af; cursor: pointer; border-radius: 0.3rem;
            font-size: 0.75rem; display: flex; align-items: center; gap: 0.2rem;
            transition: color 0.15s, background 0.15s;
        }
        .action-btn:hover { color: #374151; background: #f3f4f6; }
        .action-btn svg { width: 13px; height: 13px; }

        .typing-indicator { color: #9ca3af; font-size: 1.2rem; letter-spacing: 0.1em; }
        .typing-indicator::after { content: '···'; animation: typing 1.2s infinite; }
        @keyframes typing { 0%,60%,100%{opacity:.2} 30%{opacity:1} }

        .error-message {
            background: #fee2e2; border: 1px solid #fecaca; color: #b91c1c;
            padding: 0.65rem 0.9rem; border-radius: 0.6rem; font-size: 0.88rem;
        }

        /* ── Input area ── */
        .input-container {
            background: #fff;
            padding: 0.75rem 1rem calc(0.75rem + env(safe-area-inset-bottom));
            border-top: 1px solid #eef1f6;
        }
        .input-wrapper { max-width: 780px; margin: 0 auto; }
        .input-box {
            display: flex; align-items: flex-end; gap: 0;
            background: #fff; border: 1px solid #d7ddff; border-radius: 1rem;
            box-shadow: 0 8px 30px rgba(31, 41, 55, 0.08);
            transition: border-color 0.2s, box-shadow 0.2s;
            padding: 0.5rem 0.5rem 0.5rem 0.75rem;
        }
        .input-box:focus-within {
            border-color: #8d8cff; box-shadow: 0 8px 30px rgba(31, 41, 55, 0.10), 0 0 0 2px rgba(141,140,255,0.18);
        }
        .input-tools { display: flex; align-items: center; gap: 0.1rem; flex-shrink: 0; margin-right: 0.35rem; }
        .tool-btn {
            width: 28px; height: 28px; border: none; background: transparent;
            color: #9ca3af; cursor: pointer; border-radius: 0.4rem;
            display: flex; align-items: center; justify-content: center;
            transition: color 0.15s, background 0.15s; font-size: 0.8rem;
        }
        .tool-btn:hover { color: #374151; background: #f3f4f6; }
        .chat-input {
            flex: 1; border: none; outline: none; background: transparent;
            color: #111827; font-size: 0.95rem; line-height: 1.6;
            resize: none; min-height: 36px; max-height: 180px;
            font-family: inherit; padding: 0.2rem 0;
        }
        .chat-input::placeholder { color: #9ca3af; }
        .send-btn {
            flex-shrink: 0; width: 34px; height: 34px; border: none;
            border-radius: 0.6rem; background: #111827; color: #fff;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: background 0.15s; margin-left: 0.35rem;
        }
        .send-btn:hover:not(:disabled) { background: #4f46e5; }
        .send-btn:disabled { background: #e5e7eb; color: #9ca3af; cursor: not-allowed; }
        .send-btn.stop-mode { background: #ef4444; }
        .send-btn.stop-mode:hover { background: #dc2626; }
        .input-hint { font-size: 0.72rem; color: #9ca3af; margin-top: 0.4rem; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <button class="icon-btn" onclick="newConversation()" title="新会话 (Ctrl+Shift+N)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
            </button>
            <h1>EBT Chat</h1>
            <span class="header-tag">Energy-Based Transformer</span>
        </div>
        <div class="header-right">
            <img src="/logo.png" alt="上海人工智能实验室" class="header-logo">
        </div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="chat-wrapper" id="chatWrapper">
            <div class="empty-state" id="emptyState">
                <div class="empty-avatar">
                    <img src="/logo-vertical.png" alt="EBT">
                </div>
                <h2>EBT Chat</h2>
                <p>Energy-Based Transformer — iterative MCMC refinement for higher-quality generation.<br>Ask anything in English.</p>
                <div class="example-cards">
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="card-icon">🔍</div>
                        <div class="card-text">What is an Energy-Based Model and how does it differ from a standard Transformer?</div>
                    </div>
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="card-icon">🧮</div>
                        <div class="card-text">Explain the role of Langevin dynamics in MCMC sampling</div>
                    </div>
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="card-icon">💡</div>
                        <div class="card-text">Write Python code to find the minimum of f(x) = x² + 3x + 2 using gradient descent</div>
                    </div>
                    <div class="example-card" onclick="fillExample(this)">
                        <div class="card-icon">📝</div>
                        <div class="card-text">Briefly explain contrastive divergence training for EBMs</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="input-container">
        <div class="input-wrapper">
            <div class="input-box">
                <div class="input-tools">
                    <button class="tool-btn" onclick="fillInput('/help')" title="/help — 查看命令">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>
                    </button>
                    <button class="tool-btn" onclick="newConversation()" title="清空会话">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.98"/></svg>
                    </button>
                </div>
                <textarea id="chatInput" class="chat-input"
                    placeholder="Ask EBT anything, type / for commands..."
                    rows="1" onkeydown="handleKeyDown(event)"></textarea>
                <button id="sendButton" class="send-btn" onclick="handleSendOrStop()" disabled>
                    <svg id="sendIcon" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                    <svg id="stopIcon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="display:none"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
                </button>
            </div>
            <div class="input-hint">Shift+Enter 换行 · Ctrl+Shift+N 新会话 · /help 查看命令</div>
        </div>
    </div>

<script>
const API_URL = '';
const chatContainer = document.getElementById('chatContainer');
const chatWrapper   = document.getElementById('chatWrapper');
const chatInput     = document.getElementById('chatInput');
const sendButton    = document.getElementById('sendButton');
const sendIcon      = document.getElementById('sendIcon');
const stopIcon      = document.getElementById('stopIcon');
const emptyState    = document.getElementById('emptyState');

let messages = [];
let isGenerating = false;
let abortController = null;

chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 180) + 'px';
    sendButton.disabled = !this.value.trim() || isGenerating;
});

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.shiftKey && e.key === 'N') { e.preventDefault(); if (!isGenerating) newConversation(); }
});

function fillExample(card) {
    chatInput.value = card.querySelector('.card-text').textContent.trim();
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
    sendButton.disabled = false;
    chatInput.focus();
}

function fillInput(text) {
    chatInput.value = text;
    chatInput.style.height = 'auto';
    sendButton.disabled = false;
    chatInput.focus();
}

function setGenerating(v) {
    isGenerating = v;
    sendButton.disabled = false;
    if (v) {
        sendButton.classList.add('stop-mode');
        sendIcon.style.display = 'none';
        stopIcon.style.display = '';
    } else {
        sendButton.classList.remove('stop-mode');
        sendIcon.style.display = '';
        stopIcon.style.display = 'none';
        sendButton.disabled = !chatInput.value.trim();
    }
}

function handleSendOrStop() {
    if (isGenerating) {
        if (abortController) abortController.abort();
    } else {
        sendMessage();
    }
}

function newConversation() {
    if (abortController) abortController.abort();
    messages = [];
    chatWrapper.innerHTML = '';
    chatWrapper.appendChild(emptyState);
    emptyState.style.display = '';
    chatInput.value = ''; chatInput.style.height = 'auto';
    setGenerating(false); chatInput.focus();
}

function hideEmptyState() {
    emptyState.style.display = 'none';
}

function addMessage(role, content, messageIndex) {
    hideEmptyState();
    const wrap = document.createElement('div');
    wrap.className = 'message ' + role;

    if (role === 'assistant') {
        const av = document.createElement('div');
        av.className = 'msg-avatar';
        av.innerHTML = '<img src="/logo-vertical.png" alt="EBT">';
        wrap.appendChild(av);
    }

    const body = document.createElement('div');
    body.className = 'msg-body';

    const c = document.createElement('div');
    c.className = 'message-content';
    c.textContent = content;

    if (role === 'user' && messageIndex !== undefined) {
        c.title = '点击编辑并从此处重新开始';
        c.addEventListener('click', () => { if (!isGenerating) editMessage(messageIndex); });
    }
    body.appendChild(c);

    if (role === 'assistant' && messageIndex !== undefined) {
        const actions = document.createElement('div');
        actions.className = 'msg-actions';
        actions.innerHTML = `
          <button class="action-btn" title="复制" onclick="copyMsg(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            复制
          </button>
          <button class="action-btn" title="重新生成" onclick="regenFromAction(this)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.98"/></svg>
            重新生成
          </button>`;
        body.appendChild(actions);
    }

    wrap.appendChild(body);
    chatWrapper.appendChild(wrap);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return c;
}

function copyMsg(btn) {
    const text = btn.closest('.msg-body').querySelector('.message-content').textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '已复制'; setTimeout(() => { btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> 复制`; }, 1500);
    });
}

function regenFromAction(btn) {
    if (isGenerating) return;
    const msgEl = btn.closest('.message');
    const allMsgs = [...chatWrapper.querySelectorAll('.message')];
    const idx = allMsgs.indexOf(msgEl);
    if (idx >= 0) regenerateMessage(idx);
}

function editMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'user') return;
    chatInput.value = messages[idx].content;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    if (messages.length === 0) { chatWrapper.appendChild(emptyState); emptyState.style.display = ''; }
    setGenerating(false); chatInput.focus();
}

async function regenerateMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'assistant') return;
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    await generateAssistantResponse();
}

async function generateAssistantResponse() {
    setGenerating(true);
    abortController = new AbortController();
    const el = addMessage('assistant', '');
    el.innerHTML = '<span class="typing-indicator"></span>';
    try {
        const resp = await fetch(API_URL + '/chat/completions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ messages: messages }),
            signal: abortController.signal,
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const reader = resp.body.getReader();
        const dec = new TextDecoder('utf-8');
        let sseBuf = '';
        let full = ''; el.textContent = '';
        let serverDone = false;
        while (true) {
            const { done, value } = await reader.read();
            sseBuf += dec.decode(value ? value : new Uint8Array(), { stream: !done });
            if (done) sseBuf += dec.decode();
            for (;;) {
                const nl = sseBuf.indexOf('\n');
                if (nl < 0) break;
                const line = sseBuf.slice(0, nl).replace(/\r$/, '');
                sseBuf = sseBuf.slice(nl + 1);
                if (!line.startsWith('data: ')) continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.token) { full += d.token; el.textContent = full; chatContainer.scrollTop = chatContainer.scrollHeight; }
                    if (d.error) { el.innerHTML = '<div class="error-message">Error: ' + d.error + '</div>'; serverDone = true; break; }
                    if (d.done) { serverDone = true; break; }
                } catch(_){}
            }
            if (serverDone || done) break;
        }
        if (serverDone) { try { await reader.cancel(); } catch(_) {} }
        const tail = sseBuf.trim();
        if (!serverDone && tail.startsWith('data: ')) {
            try {
                const d = JSON.parse(tail.slice(6));
                if (d.token) { full += d.token; el.textContent = full; }
                if (d.error) { el.innerHTML = '<div class="error-message">Error: ' + d.error + '</div>'; }
            } catch(_){}
        }
        const aidx = messages.length;
        messages.push({role:'assistant', content: full});
        // add action buttons after done
        const body = el.closest('.msg-body');
        if (body) {
            const actions = document.createElement('div');
            actions.className = 'msg-actions';
            actions.innerHTML = `
              <button class="action-btn" title="复制" onclick="copyMsg(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                复制
              </button>
              <button class="action-btn" title="重新生成" onclick="regenFromAction(this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.98"/></svg>
                重新生成
              </button>`;
            body.appendChild(actions);
        }
    } catch(err) {
        if (err.name === 'AbortError') {
            if (!el.textContent) el.innerHTML = '<span style="color:#9ca3af;font-size:0.85rem">已停止生成</span>';
            messages.push({role:'assistant', content: el.textContent});
        } else {
            el.innerHTML = '<div class="error-message">Error: ' + err.message + '</div>';
        }
    } finally {
        setGenerating(false);
        abortController = null;
    }
}

async function handleSlashCommand(cmd) {
    try {
        const resp = await fetch(API_URL + '/command', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({command: cmd})
        });
        const data = await resp.json();
        if (data.action === 'clear') { newConversation(); return; }
        addMessage('console', data.result);
    } catch(err) {
        addMessage('console', 'Error: ' + err.message);
    }
}

async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg || isGenerating) return;
    chatInput.value = ''; chatInput.style.height = 'auto';
    if (msg.startsWith('/')) { await handleSlashCommand(msg); return; }
    const uidx = messages.length;
    messages.push({role:'user', content: msg});
    addMessage('user', msg, uidx);
    await generateAssistantResponse();
}

setGenerating(false);
chatInput.focus();

fetch(API_URL + '/health').then(r=>r.json()).then(d=>{
    console.log('EBT Engine status:', d);
}).catch(() => {
    hideEmptyState();
    const err = document.createElement('div');
    err.className = 'error-message';
    err.style.margin = '2rem auto'; err.style.maxWidth = '600px';
    err.textContent = 'EBT 引擎未就绪，请等待模型加载完成后刷新页面。';
    chatWrapper.appendChild(err);
});
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print(f"Starting EBT Web Chat Server on port {args.port}")
    print(f"Temperature: {args.temperature}, Top-P: {args.top_p}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
