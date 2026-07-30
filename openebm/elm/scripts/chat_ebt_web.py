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
import torch
from contextlib import asynccontextmanager, nullcontext
from typing import Optional, List, Dict, Any, AsyncGenerator

from openebm.elm.generate import call_model_forward_decode, _get_tokenizer, sample_top_p

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
parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定地址')
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

        self.tokenizer = _get_tokenizer(self.hparams)

        vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else len(self.tokenizer)
        print(f"  Raw tokenizer vocab size: {vocab_size}")

        if not hasattr(self.tokenizer, 'get_vocab_size'):
            if hasattr(self.tokenizer, 'tokenizer_obj') and hasattr(self.tokenizer.tokenizer_obj, 'get_vocab_size'):
                self.tokenizer.get_vocab_size = self.tokenizer.tokenizer_obj.get_vocab_size
            else:
                self.tokenizer.get_vocab_size = lambda: len(self.tokenizer)

        self.hparams.tokenizer_obj = self.tokenizer

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
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
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
        stop_token_ids = set()
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            asst_end_id = inner_tok.encode_special("<|assistant_end|>")
            if asst_end_id is not None:
                stop_token_ids.add(asst_end_id)
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

        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
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
            asst_end_id = inner_tok.encode_special("<|assistant_end|>")
            if asst_end_id is not None:
                stop_token_ids.add(asst_end_id)
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

                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= (next_token == sid)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                if all(eos_reached):
                    break

                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                if token_text:
                    yield token_text


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Web Server
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# ── 运行时可调参数 (通过 /command API 修改) ──
runtime_config = {
    "temperature": args.temperature,
    "top_p": args.top_p,
    "max_tokens": args.max_tokens,
}

# ── 全局锁: EBT 模型同一时刻只能处理一个请求 ──
generate_lock = asyncio.Lock()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    print("=" * 70)
    print("EBT Web Chat Server - 正在初始化...")
    print("=" * 70)

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    torch.set_float32_matmul_precision('medium')

    app.state.engine = EBTChatEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=dtype,
        show_mcmc=args.show_mcmc,
        verbose=args.verbose,
        show_energy=args.show_energy,
        show_distribution=args.show_distribution,
        override_mcmc_steps=args.override_mcmc_steps,
        override_noise_std=args.override_noise_std,
        override_alpha=args.override_alpha,
    )
    print(f"\n✓ EBT Web Chat Server ready at http://0.0.0.0:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── GET / ── 内嵌 HTML UI ──
@app.get("/")
async def root():
    return HTMLResponse(content=EBT_CHAT_HTML)


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

    engine: EBTChatEngine = app.state.engine
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    response_tokens = []

    async def stream_sse():
        async with generate_lock:
            loop = asyncio.get_event_loop()
            # 因为 EBT generate 是同步阻塞的 (GPU forward), 需要跑在线程池里
            gen = engine.generate_stream_multi_turn(messages_dicts, max_tokens=max_tok,
                                                     temperature=temp, top_p=top_p)
            # 用 queue 桥接同步生成器 -> 异步 SSE
            q: asyncio.Queue = asyncio.Queue()

            def _run_gen():
                try:
                    for tok in gen:
                        q.put_nowait(tok)
                except Exception as e:
                    q.put_nowait(e)
                finally:
                    q.put_nowait(None)  # sentinel

            fut = loop.run_in_executor(None, _run_gen)

            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    yield f"data: {json.dumps({'error': str(item)})}\n\n"
                    break
                response_tokens.append(item)
                yield f"data: {json.dumps({'token': item}, ensure_ascii=False)}\n\n"

            await fut  # 确保线程结束

        full_response = "".join(response_tokens)
        logger.info(f"[ASSISTANT]: {full_response}")
        logger.info("=" * 40)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream_sse(), media_type="text/event-stream")


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
            background-color: #ffffff; color: #111827;
            min-height: 100dvh; display: flex; flex-direction: column;
        }
        .header {
            background-color: #ffffff; padding: 1rem 1.5rem;
            display: flex; align-items: center; justify-content: space-between;
            border-bottom: 1px solid #f3f4f6;
        }
        .header-left { display: flex; align-items: center; gap: 0.75rem; }
        .header h1 { font-size: 1.25rem; font-weight: 600; margin: 0; color: #111827; }
        .header-tag {
            font-size: 0.7rem; background: #eef2ff; color: #4f46e5;
            padding: 0.15rem 0.5rem; border-radius: 0.25rem; font-weight: 500;
        }
        .new-btn {
            width: 32px; height: 32px; padding: 0; border: 1px solid #e5e7eb;
            border-radius: 0.5rem; background: #fff; color: #6b7280; cursor: pointer;
            display: flex; align-items: center; justify-content: center; transition: all 0.2s;
        }
        .new-btn:hover { background: #f3f4f6; border-color: #d1d5db; color: #374151; }

        .chat-container { flex: 1; overflow-y: auto; background: #ffffff; }
        .chat-wrapper {
            max-width: 48rem; margin: 0 auto; padding: 2rem 1.5rem 3rem;
            display: flex; flex-direction: column; gap: 0.75rem;
        }
        .message { display: flex; margin-bottom: 0.5rem; color: #0d0d0d; }
        .message.assistant { justify-content: flex-start; }
        .message.user { justify-content: flex-end; }
        .message-content { white-space: pre-wrap; line-height: 1.6; max-width: 100%; }
        .message.assistant .message-content {
            background: transparent; border: none; cursor: pointer; border-radius: 0.5rem;
            padding: 0.5rem; margin-left: -0.5rem; transition: background-color 0.2s;
        }
        .message.assistant .message-content:hover { background: #f9fafb; }
        .message.user .message-content {
            background-color: #f3f4f6; border-radius: 1.25rem; padding: 0.8rem 1rem;
            max-width: 65%; cursor: pointer; transition: background-color 0.2s;
        }
        .message.user .message-content:hover { background-color: #e5e7eb; }
        .message.console .message-content {
            font-family: 'Monaco','Menlo','Consolas','Courier New', monospace;
            font-size: 0.85rem; background: #f8fafc; border: 1px solid #e2e8f0;
            padding: 0.75rem 1rem; color: #374151; max-width: 85%; border-radius: 0.5rem;
        }

        .input-container { background: #fff; padding: 1rem; padding-bottom: calc(1rem + env(safe-area-inset-bottom)); }
        .input-wrapper { max-width: 48rem; margin: 0 auto; display: flex; gap: 0.75rem; align-items: flex-end; }
        .chat-input {
            flex: 1; padding: 0.8rem 1rem; border: 1px solid #d1d5db; border-radius: 0.75rem;
            background: #fff; color: #111827; font-size: 1rem; line-height: 1.5;
            resize: none; outline: none; min-height: 54px; max-height: 200px;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .chat-input::placeholder { color: #9ca3af; }
        .chat-input:focus { border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
        .send-btn {
            flex-shrink: 0; width: 54px; height: 54px; border: 1px solid #111827;
            border-radius: 0.75rem; background: #111827; color: #fff;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: background 0.2s, border-color 0.2s;
        }
        .send-btn:hover:not(:disabled) { background: #4f46e5; border-color: #4f46e5; }
        .send-btn:disabled { cursor: not-allowed; border-color: #d1d5db; background: #e5e7eb; color: #9ca3af; }

        .typing-indicator { display: inline-block; color: #6b7280; letter-spacing: 0.15em; }
        .typing-indicator::after { content: '···'; animation: typing 1.4s infinite; }
        @keyframes typing { 0%,60%,100%{opacity:.2;} 30%{opacity:1;} }
        .error-message {
            background: #fee2e2; border: 1px solid #fecaca; color: #b91c1c;
            padding: 0.75rem 1rem; border-radius: 0.75rem; margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <button class="new-btn" onclick="newConversation()" title="New Conversation (Ctrl+Shift+N)">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
            </button>
            <h1>EBT Chat</h1>
            <span class="header-tag">Energy-Based Transformer</span>
        </div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="chat-wrapper" id="chatWrapper"></div>
    </div>

    <div class="input-container">
        <div class="input-wrapper">
            <textarea id="chatInput" class="chat-input" placeholder="输入消息或 /help 查看命令..." rows="1" onkeydown="handleKeyDown(event)"></textarea>
            <button id="sendButton" class="send-btn" onclick="sendMessage()" disabled>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
            </button>
        </div>
    </div>

<script>
const API_URL = '';
const chatContainer = document.getElementById('chatContainer');
const chatWrapper   = document.getElementById('chatWrapper');
const chatInput     = document.getElementById('chatInput');
const sendButton    = document.getElementById('sendButton');

let messages = [];
let isGenerating = false;

chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    sendButton.disabled = !this.value.trim() || isGenerating;
});

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.shiftKey && e.key === 'N') { e.preventDefault(); if (!isGenerating) newConversation(); }
});

function newConversation() {
    messages = []; chatWrapper.innerHTML = '';
    chatInput.value = ''; chatInput.style.height = 'auto';
    sendButton.disabled = false; isGenerating = false; chatInput.focus();
}

function addMessage(role, content, messageIndex) {
    const div = document.createElement('div');
    div.className = 'message ' + role;
    const c = document.createElement('div');
    c.className = 'message-content';
    c.textContent = content;
    if (role === 'user' && messageIndex !== undefined) {
        c.title = '点击编辑并从此处重新开始';
        c.addEventListener('click', () => { if (!isGenerating) editMessage(messageIndex); });
    }
    if (role === 'assistant' && messageIndex !== undefined) {
        c.title = '点击重新生成此回复';
        c.addEventListener('click', () => { if (!isGenerating) regenerateMessage(messageIndex); });
    }
    div.appendChild(c);
    chatWrapper.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return c;
}

function editMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'user') return;
    chatInput.value = messages[idx].content;
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    sendButton.disabled = false; chatInput.focus();
}

async function regenerateMessage(idx) {
    if (idx < 0 || idx >= messages.length || messages[idx].role !== 'assistant') return;
    messages = messages.slice(0, idx);
    const all = chatWrapper.querySelectorAll('.message');
    for (let i = idx; i < all.length; i++) all[i].remove();
    await generateAssistantResponse();
}

async function generateAssistantResponse() {
    isGenerating = true; sendButton.disabled = true;
    const el = addMessage('assistant', '');
    el.innerHTML = '<span class="typing-indicator"></span>';
    try {
        const resp = await fetch(API_URL + '/chat/completions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ messages: messages })
        });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let full = ''; el.textContent = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            for (const line of dec.decode(value).split('\n')) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const d = JSON.parse(line.slice(6));
                    if (d.token) { full += d.token; el.textContent = full; chatContainer.scrollTop = chatContainer.scrollHeight; }
                    if (d.error) { el.innerHTML = '<div class="error-message">Error: ' + d.error + '</div>'; }
                } catch(_){}
            }
        }
        const aidx = messages.length;
        messages.push({role:'assistant', content: full});
        el.title = '点击重新生成此回复';
        el.addEventListener('click', () => { if (!isGenerating) regenerateMessage(aidx); });
    } catch(err) {
        el.innerHTML = '<div class="error-message">Error: ' + err.message + '</div>';
    } finally {
        isGenerating = false; sendButton.disabled = !chatInput.value.trim();
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

sendButton.disabled = false;
chatInput.focus();

fetch(API_URL + '/health').then(r=>r.json()).then(d=>{
    console.log('EBT Engine status:', d);
}).catch(err=>{
    chatWrapper.innerHTML = '<div class="error-message">EBT 引擎未就绪，请等待模型加载完成后刷新页面。</div>';
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
