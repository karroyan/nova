#!/usr/bin/env python3
"""
EBT 交互式对话脚本 - 与训练好的 EBT 模型进行交互式对话

该脚本加载 EBT checkpoint，支持交互式对话，并可选择性地展示 MCMC 步骤的迭代过程，
便于用户直观评估 EBT 模型的效果和各项指标。

使用方法:
    python -m scripts.chat_ebt                           # 默认参数
    python -m scripts.chat_ebt --show-mcmc               # 展示 MCMC 步骤
    python -m scripts.chat_ebt --show-mcmc --verbose     # 详细展示每步信息
    python -m scripts.chat_ebt -c /path/to/checkpoint    # 指定 checkpoint
"""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# 从 generate.py 导入核心逻辑，保证行为 100% 一致
from openebm.elm.generate import call_model_forward_decode, _get_tokenizer, sample_top_p

# 清除分布式训练环境变量
for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
    if var in os.environ:
        del os.environ[var]

# 设置离线模式
os.environ['NANOCHAT_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"

# 颜色代码
class Colors:
    BLUE = '\033[1;34m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[1;31m'
    CYAN = '\033[1;36m'
    MAGENTA = '\033[1;35m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # MCMC 步骤颜色渐变
    MCMC_COLORS = [
        '\033[38;5;196m',  # 红
        '\033[38;5;208m',  # 橙
        '\033[38;5;226m',  # 黄
        '\033[38;5;118m',  # 绿
        '\033[38;5;51m',   # 青
    ]


def print_colored(text: str, color: str):
    """打印带颜色的文本"""
    print(f"{color}{text}{Colors.RESET}")


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██████╗ ████████╗     ██████╗██╗  ██╗ █████╗ ████████╗             ║
║   ██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝             ║
║   █████╗  ██████╔╝   ██║       ██║     ███████║███████║   ██║                ║
║   ██╔══╝  ██╔══██╗   ██║       ██║     ██╔══██║██╔══██║   ██║                ║
║   ███████╗██████╔╝   ██║       ╚██████╗██║  ██║██║  ██║   ██║                ║
║   ╚══════╝╚═════╝    ╚═╝        ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                ║
║                                                                              ║
║           Energy-Based Transformer Interactive Chat Terminal                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print_colored(banner, Colors.CYAN)


class EBTChatEngine:
    """EBT 对话引擎"""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        show_mcmc: bool = False,
        verbose: bool = False,
        show_energy: bool = False,
        show_distribution: bool = False,
        override_mcmc_steps: Optional[int] = None,
        override_noise_std: Optional[float] = None,
        override_alpha: Optional[float] = None,
    ):
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

    def _load_model(self, checkpoint_path: str, tokenizer_path: str):
        """加载模型和 tokenizer"""
        print_colored("正在加载模型...", Colors.YELLOW)

        # 加载 checkpoint
        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 提取 hparams
        if 'hyper_parameters' in checkpoint:
            self.hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            self.hparams = checkpoint['hparams']
        else:
            raise ValueError("Cannot find hyperparameters in checkpoint")

        # 转换为 namespace 风格访问
        class HParamsNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        if isinstance(self.hparams, dict):
            self.hparams = HParamsNamespace(self.hparams)

        # 【修复点 1】: 使用 generate.py 的 _get_tokenizer
        self.tokenizer = _get_tokenizer(self.hparams)
        
        # 兼容性：如果 tokenizer 有 get_vocab_size 方法则调用，否则使用 len
        vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else len(self.tokenizer)
        print(f"  Raw tokenizer vocab size: {vocab_size}")

        # 【修复点 1.5】: 动态补齐 get_vocab_size 方法
        if not hasattr(self.tokenizer, 'get_vocab_size'):
            if hasattr(self.tokenizer, 'tokenizer_obj') and hasattr(self.tokenizer.tokenizer_obj, 'get_vocab_size'):
                self.tokenizer.get_vocab_size = self.tokenizer.tokenizer_obj.get_vocab_size
            else:
                self.tokenizer.get_vocab_size = lambda: len(self.tokenizer)

        self.hparams.tokenizer_obj = self.tokenizer

        # 兼容旧 checkpoint：补齐新版 setup_ebt() 要求的属性
        if not hasattr(self.hparams, 'use_ve'):
            self.hparams.use_ve = False

        # 创建模型
        from openebm.elm.modeling_ebt import EBT_NLP
        self.model = EBT_NLP(self.hparams)

        # 【修复点 5】: 彻底清洗 state_dict 键名，解决 torch.compile 带来的 _orig_mod 前缀问题
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            # 1. 剥离 PyTorch Lightning 或 DDP 可能带来的 'model.' 前缀
            if new_key.startswith('model.'):
                new_key = new_key[6:]
            # 2. 剥离 torch.compile 带来的 '_orig_mod.' 前缀
            if new_key.startswith('_orig_mod.'):
                new_key = new_key[10:]
            
            new_state_dict[new_key] = v

        # 尝试加载权重，并强制要求严格匹配 (strict=True)，这样如果有遗漏能立刻发现
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print_colored("  ✓ 权重加载成功 (strict=True，所有参数完美匹配)", Colors.GREEN)
        except Exception as e:
            print_colored(f"  ⚠ 严格加载失败，正在回退到 strict=False。详细错误:\n{e}", Colors.YELLOW)
            self.model.load_state_dict(new_state_dict, strict=False)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        print_colored("✓ 模型加载与初始化完成", Colors.GREEN)
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        embed_dim = getattr(self.hparams, 'embedding_dim', getattr(self.hparams, 'dim', '未知'))
        n_layers = getattr(self.hparams, 'num_layers', getattr(self.hparams, 'n_layers', '未知'))
        n_heads = getattr(self.hparams, 'num_heads', getattr(self.hparams, 'n_heads', '未知'))
        mcmc_steps = getattr(self.hparams, 'mcmc_num_steps', '未知')
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', '未知'))
        
        alpha_val = '未知'
        if hasattr(self.model, 'alpha'):
            alpha_val = self.model.alpha.item() if isinstance(self.model.alpha, torch.Tensor) else self.model.alpha

        noise_std_val = '未知'
        if hasattr(self.model, 'langevin_dynamics_noise_std'):
            noise_std_val = self.model.langevin_dynamics_noise_std.item() if isinstance(self.model.langevin_dynamics_noise_std, torch.Tensor) else self.model.langevin_dynamics_noise_std

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
                mcmc_steps = self.override_mcmc_steps

                trained_noise = getattr(self.hparams, 'langevin_dynamics_noise', 0.0)
                if self.override_noise_std is None and trained_noise == 0:
                    auto_noise = 0.0
                    self.hparams.langevin_dynamics_noise = auto_noise
                    self.model.hparams.langevin_dynamics_noise = auto_noise
                    if hasattr(self.model, 'langevin_dynamics_noise_std'):
                        self.model.langevin_dynamics_noise_std.data.fill_(auto_noise)
                    noise_std_val = auto_noise
            elif self.override_mcmc_steps < original_steps:
                self.hparams.mcmc_num_steps = mcmc_steps = self.override_mcmc_steps
                self.model.hparams.mcmc_num_steps = self.override_mcmc_steps

        if self.override_noise_std is not None:
            if hasattr(self.model, 'langevin_dynamics_noise_std'):
                self.model.langevin_dynamics_noise_std.data.fill_(self.override_noise_std)
                self.hparams.langevin_dynamics_noise = self.override_noise_std
                self.model.hparams.langevin_dynamics_noise = self.override_noise_std
                noise_std_val = self.override_noise_std

        if self.override_alpha is not None:
            if hasattr(self.model, 'alpha'):
                self.model.alpha = torch.tensor(
                    self.override_alpha,
                    dtype=self.model.alpha.dtype,
                    device=self.model.alpha.device
                )
                alpha_val = self.override_alpha

        print()
        print(f"  {Colors.BOLD}模型配置:{Colors.RESET}")
        print(f"    - 嵌入维度: {embed_dim}")
        print(f"    - 层数: {n_layers}")
        print(f"    - 注意力头数: {n_heads}")
        print(f"    - MCMC 步数: {mcmc_steps}")
        
        if isinstance(alpha_val, float):
            print(f"    - MCMC 步长 (alpha): {alpha_val:.6f}")
        else:
            print(f"    - MCMC 步长 (alpha): {alpha_val}")
            
        if isinstance(noise_std_val, float):
            print(f"    - Langevin 噪声: {noise_std_val:.6f}")
        else:
            print(f"    - Langevin 噪声: {noise_std_val}")
            
        print(f"    - 上下文长度: {ctx_len}")
        print()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        stream: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """生成文本 - 完全复用 generate.py 的逻辑"""
        
        # 使用 nanochat 的 render_conversation 格式构建 prompt
        # 格式: <|bos|> <|user_start|> {text} <|user_end|> <|assistant_start|>
        # 这与 SFT 训练时 dataset_sft.py -> render_conversation() 的格式完全一致
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)  # NanoChatTokenizerWrapper.tokenizer = RustBPETokenizer
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            bos_id       = inner_tok.get_bos_token_id()
            user_start   = inner_tok.encode_special("<|user_start|>")
            user_end     = inner_tok.encode_special("<|user_end|>")
            asst_start   = inner_tok.encode_special("<|assistant_start|>")
            content_ids  = inner_tok.encode(prompt)
            prompt_tokens_list = [bos_id, user_start] + content_ids + [user_end, asst_start]
        else:
            # Fallback: 直接 encode（base model 场景，无 chat template）
            encoded = self.tokenizer.encode(prompt)
            prompt_tokens_list = encoded if isinstance(encoded, list) else encoded.tolist()
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            if bos_id is not None and (not prompt_tokens_list or prompt_tokens_list[0] != bos_id):
                prompt_tokens_list = [bos_id] + prompt_tokens_list

        # pad_id 仅用于填充 tokens 张量，不作为停止 token
        # NanoChatTokenizerWrapper 把 eos/pad 都设为 bos_id，但 bos_id 不应是停止 token
        # （SFT packed 序列里 assistant_end 后紧跟 bos，模型可能在 assistant_start 后生成 bos，
        #  若把 bos 加入 stop_token_ids 会导致立即停止、输出为空）
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            pad_id = self.tokenizer.bos_token_id
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = 0

        bsz = 1
        min_prompt_len = len(prompt_tokens_list)
        max_prompt_len = len(prompt_tokens_list)

        # 安全获取 context_length
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 2048))
        total_len = min(ctx_len, max_tokens + max_prompt_len)

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        tokens[0, :len(prompt_tokens_list)] = torch.tensor(prompt_tokens_list, dtype=torch.long, device=self.device)

        # 【修复】用位置掩码而非 token 值掩码，避免 bos_id == pad_id 导致 prompt 第一个 token 被误判为 padding
        input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=self.device)
        input_text_mask[0, :len(prompt_tokens_list)] = True

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        start_time = time.time()

        # 【修复 v2】扩充 stop_token_ids 以应对过拟合 ckpt 在 OOD 输入上不输出 <|assistant_end|> 的情况
        # 安全性：L322-324 已使用位置掩码 input_text_mask，prompt 区域的 bos/special tokens
        # 不会被误判为"生成了停止信号"（仅对 cur_pos 及之后的位置生效）。
        stop_token_ids = set()
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            asst_end_id = inner_tok.encode_special("<|assistant_end|>")
            if asst_end_id is not None:
                stop_token_ids.add(asst_end_id)
            # 扩充：新一轮对话起始 token 也视为停止信号
            try:
                user_start_id = inner_tok.encode_special("<|user_start|>")
                if user_start_id is not None:
                    stop_token_ids.add(user_start_id)
            except Exception:
                pass
        # 扩充：bos_id 作为停止信号（模型把生成结束表达为 bos 的情况）
        if hasattr(inner_tok, 'get_bos_token_id'):
            try:
                bos_id_extra = inner_tok.get_bos_token_id()
                if bos_id_extra is not None:
                    stop_token_ids.add(bos_id_extra)
            except Exception:
                pass
        # fallback：若没有任何 special token，用 pad_id 兜底
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        with torch.no_grad():
            if min_prompt_len == total_len:
                logits = call_model_forward_decode(self.hparams, self.model, tokens, prev_pos, bsz)

            for cur_pos in range(min_prompt_len, total_len):
                input_tokens = tokens[:, :cur_pos]

                logits = call_model_forward_decode(self.hparams, self.model, input_tokens, prev_pos, bsz)

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)

                # ===== 调试打印：加在这里 =====
                # tok_id = next_token.item()
                # tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                # print(f"\n[DEBUG] pos={cur_pos} token_id={tok_id} repr={repr(tok_str)} "
                #     f"is_stop={tok_id in stop_token_ids} "
                #     f"top5_probs={torch.topk(probs[0], 5)}")
                # ===== 调试打印结束 =====

                next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token

                # 流式输出 (Streaming)
                if stream and cur_pos >= min_prompt_len:
                    token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    print(token_text, end='', flush=True)

                # EOS 检查：<|assistant_end|>（不含 bos/pad，避免 SFT 模型生成 bos 时误停）
                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= (next_token == sid)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                if all(eos_reached):
                    break

        # 提取生成的 tokens
        toks = tokens[0].tolist()
        start = len(prompt_tokens_list)
        toks = toks[start : len(prompt_tokens_list) + max_tokens]

        # cut to first stop token
        for sid in stop_token_ids:
            if sid in toks:
                toks = toks[:toks.index(sid)]

        generated_text = self.tokenizer.decode(toks, skip_special_tokens=True)

        # print(f"[DEBUG] prompt_tokens_list = {prompt_tokens_list}")
        # print(f"[DEBUG] prompt decoded = {repr(self.tokenizer.decode(prompt_tokens_list, skip_special_tokens=False))}")
        # print(f"[DEBUG] pad_id={pad_id} stop_token_ids={stop_token_ids}")

        stats = {
            'tokens_generated': len(toks),
            'total_time': time.time() - start_time,
            'tokens_per_second': len(toks) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
            'avg_energy_change': 0,
            'avg_token_prob': 0,
        }

        return generated_text, stats


def print_help():
    """显示帮助信息"""
    print()
    print(f"{Colors.BOLD}可用命令:{Colors.RESET}")
    print("  /quit, /exit        - 退出对话")
    print("  /clear              - 清空对话历史")
    print("  /temp <值>          - 设置温度 (0.0-2.0)")
    print("  /topp <值>          - 设置 top-p (0.0-1.0)")
    print("  /tokens <值>        - 设置最大 tokens (1-4096)")
    print("  /mcmc               - 切换 MCMC 显示")
    print("  /verbose            - 切换详细模式")
    print("  /energy             - 切换能量显示")
    print("  /single             - 切换到单行输入模式")
    print("  /multi              - 切换到多行输入模式 (空行结束)")
    print("  /status             - 显示当前设置")
    print("  /info               - 显示模型信息")
    print("  /help               - 显示此帮助")
    print()


def print_status(engine: EBTChatEngine, temperature: float, top_p: float, max_tokens: int):
    """显示当前状态"""
    print()
    print(f"{Colors.BOLD}当前设置:{Colors.RESET}")
    print(f"  温度: {temperature}")
    print(f"  Top-P: {top_p}")
    print(f"  最大 Tokens: {max_tokens}")
    print(f"  显示 MCMC: {'是' if engine.show_mcmc else '否'}")
    print(f"  详细模式: {'是' if engine.verbose else '否'}")
    print(f"  显示能量: {'是' if engine.show_energy else '否'}")
    print()


def print_generation_stats(stats: Dict[str, Any]):
    """打印生成统计信息"""
    print()
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print(f"{Colors.BOLD}生成统计:{Colors.RESET}")
    print(f"  生成 tokens: {stats['tokens_generated']}")
    print(f"  总时间: {stats['total_time']:.2f}s")
    print(f"  速度: {stats['tokens_per_second']:.2f} tokens/s")
    print(f"  平均 token 概率: {stats['avg_token_prob']:.4f}")
    print(f"  平均能量变化: {stats['avg_energy_change']:.4f}")
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description='EBT 交互式对话终端')
    parser.add_argument('-c', '--checkpoint', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt",
                       help='Checkpoint 路径')
    parser.add_argument('--tokenizer', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer",
                       help='Tokenizer 路径')
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                       help='生成温度 (默认: 0.8)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-P 采样 (默认: 0.9)')
    parser.add_argument('-m', '--max-tokens', type=int, default=256,
                       help='最大生成 tokens (默认: 256)')
    parser.add_argument('--show-mcmc', action='store_true',
                       help='展示 MCMC 步骤过程')
    parser.add_argument('--verbose', action='store_true',
                       help='详细模式')
    parser.add_argument('--show-energy', action='store_true',
                       help='展示能量值变化')
    parser.add_argument('--show-distribution', action='store_true',
                       help='展示概率分布变化')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16',
                       choices=['float32', 'bfloat16'],
                       help='数据类型 (默认: bfloat16)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (默认: cuda)')

    # 高级推理参数 (覆盖训练配置)
    parser.add_argument('--override-mcmc-steps', type=int, default=None,
                       help='覆盖训练时的 MCMC 步数 (默认: 使用训练值)')
    parser.add_argument('--override-noise-std', type=float, default=None,
                       help='覆盖 Langevin 噪声标准差 (默认: 使用训练值)')
    parser.add_argument('--override-alpha', type=float, default=None,
                       help='覆盖 MCMC 步长 alpha (默认: 使用训练值)')

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 验证 checkpoint
    if not os.path.exists(args.checkpoint):
        print_colored(f"错误: 找不到 checkpoint: {args.checkpoint}", Colors.RED)
        return 1

    # 初始化引擎
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    # 性能优化: 启用 TF32 (H200/A100 Tensor Core 加速)
    torch.set_float32_matmul_precision('medium')

    try:
        engine = EBTChatEngine(
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
    except Exception as e:
        print_colored(f"错误: 加载模型失败 - {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return 1

    # 显示使用说明
    print("=" * 70)
    print_colored("EBT 对话终端已就绪!", Colors.GREEN)
    print("=" * 70)
    print("使用说明:")
    print("  - 默认多行输入模式：输入多行文本后按两次 Enter (空行) 结束")
    print("  - 输入 /single 切换到单行输入模式")
    print("  - 输入 /quit 或 /exit 退出")
    print("  - 输入 /help 查看所有命令")
    print("  - 输入 /mcmc 切换 MCMC 步骤显示")
    print("=" * 70)
    print()

    # 参数
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    # 对话循环
    session_total_tokens = 0
    session_total_time = 0.0
    session_turns = 0
    multiline_mode = True

    while True:
        try:
            if multiline_mode:
                print(f'{Colors.BLUE}你 (空行结束):{Colors.RESET}')
                lines = []
                while True:
                    try:
                        line = input()
                        if line == '' and lines:
                            break
                        lines.append(line)
                    except EOFError:
                        break
                user_input = '\n'.join(lines)
            else:
                user_input = input(f'{Colors.BLUE}你:{Colors.RESET} ')

            # 处理命令
            cmd = user_input.strip().lower()

            if cmd in ['/quit', '/exit']:
                print_colored('\n再见!', Colors.GREEN)
                break

            if cmd == '/clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print_banner()
                continue

            if cmd == '/help':
                print_help()
                continue

            if cmd == '/status':
                print_status(engine, temperature, top_p, max_tokens)
                continue

            if cmd == '/info':
                engine._print_model_info()
                continue

            if cmd == '/mcmc':
                engine.show_mcmc = not engine.show_mcmc
                status = "开启" if engine.show_mcmc else "关闭"
                print_colored(f'\n✓ MCMC 显示已{status}\n', Colors.GREEN)
                continue

            if cmd == '/verbose':
                engine.verbose = not engine.verbose
                status = "开启" if engine.verbose else "关闭"
                print_colored(f'\n✓ 详细模式已{status}\n', Colors.GREEN)
                continue

            if cmd == '/energy':
                engine.show_energy = not engine.show_energy
                status = "开启" if engine.show_energy else "关闭"
                print_colored(f'\n✓ 能量显示已{status}\n', Colors.GREEN)
                continue

            if cmd == '/single':
                multiline_mode = False
                print_colored('\n✓ 已切换到单行输入模式\n', Colors.GREEN)
                continue

            if cmd == '/multi':
                multiline_mode = True
                print_colored('\n✓ 已切换到多行输入模式 (空行结束)\n', Colors.GREEN)
                continue

            if cmd.startswith('/temp '):
                try:
                    new_temp = float(cmd.split()[1])
                    if 0.0 <= new_temp <= 2.0:
                        temperature = new_temp
                        print_colored(f'\n✓ 温度已设置为: {temperature}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ 温度必须在 0.0-2.0 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的温度值\n', Colors.RED)
                continue

            if cmd.startswith('/topp '):
                try:
                    new_topp = float(cmd.split()[1])
                    if 0.0 <= new_topp <= 1.0:
                        top_p = new_topp
                        print_colored(f'\n✓ Top-P 已设置为: {top_p}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ Top-P 必须在 0.0-1.0 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的 Top-P 值\n', Colors.RED)
                continue

            if cmd.startswith('/tokens '):
                try:
                    new_tokens = int(cmd.split()[1])
                    if 1 <= new_tokens <= 4096:
                        max_tokens = new_tokens
                        print_colored(f'\n✓ 最大 Tokens 已设置为: {max_tokens}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ 最大 Tokens 必须在 1-4096 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的 Tokens 值\n', Colors.RED)
                continue

            # 空输入
            if not user_input.strip():
                continue

            # 生成回复
            print(f'{Colors.GREEN}EBT:{Colors.RESET} ', end='', flush=True)

            try:
                generated_text, stats = engine.generate(
                    prompt=user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                print()  # 换行

                # 累计会话统计
                session_total_tokens += stats['tokens_generated']
                session_total_time += stats['total_time']
                session_turns += 1

                # 始终显示简要统计 (verbose 时显示完整)
                if engine.verbose or engine.show_mcmc:
                    print_generation_stats(stats)
                else:
                    print(f"{Colors.GRAY}  [{stats['tokens_generated']} tokens, {stats['total_time']:.1f}s, {stats['tokens_per_second']:.1f} tok/s]{Colors.RESET}")
                    print()

            except Exception as e:
                print()
                print_colored(f'✗ 生成错误: {str(e)}', Colors.RED)
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print_colored('\n\n检测到 Ctrl+C，退出...', Colors.YELLOW)
            break
        except EOFError:
            print_colored('\n\n检测到 EOF，退出...', Colors.YELLOW)
            break

    # 会话汇总
    if session_turns > 0:
        avg_tps = session_total_tokens / session_total_time if session_total_time > 0 else 0
        print()
        print("=" * 50)
        print(f"  会话汇总")
        print("=" * 50)
        print(f"  对话轮数:     {session_turns}")
        print(f"  总生成 tokens: {session_total_tokens}")
        print(f"  总生成时间:   {session_total_time:.2f}s")
        print(f"  平均吞吐量:   {avg_tps:.2f} tokens/s")
        print(f"  平均每轮:     {session_total_tokens/session_turns:.0f} tokens, {session_total_time/session_turns:.1f}s")
        print("=" * 50)
        print(f"[SESSION_SUMMARY] turns={session_turns} tokens={session_total_tokens} time={session_total_time:.1f}s throughput={avg_tps:.2f}tok/s")

    print_colored('\n对话已结束。', Colors.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())