"""
VE (Value Embedding) 核心模块

实现 GPT 风格的 Value Embedding，为每个启用层建立 nn.Embedding(vocab_size, kv_dim) 表：
- real 分支：用离散 token id 查表
- predicted 分支：用 argmax 离散化后查表（避免 softmax+matmul 大张量）

通过 --use_ve 参数控制是否启用该功能。
"""

import torch
import torch.nn as nn


def has_ve(layer_idx, n_layers):
    """
    判断某一层是否使用 VE。

    策略：每隔一层使用 VE，从最后一层开始交替。
    例如：n_layers=12 时，层 1, 3, 5, 7, 9, 11 使用 VE。
    """
    return layer_idx % 2 == (n_layers - 1) % 2


def build_value_embeds(n_layers, vocab_size, kv_dim):
    """
    为需要 VE 的层构建 Embedding 表。
    """
    return nn.ModuleDict(
        {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layers) if has_ve(i, n_layers)}
    )


def build_layer_ve(value_embeds, layer_id, real_token_ids, predicted_tokens, extra_prefix_tokens=0):
    """
    为指定层构建 VE 张量。

    real 分支和 predicted 分支均使用离散 token ID 直接查表。
    predicted 分支通过 argmax 将连续概率分布离散化，避免 softmax+matmul
    产生的 (B, S, vocab_size) 大张量，节省 ~3.5 GB 激活值显存。
    """
    layer_key = str(layer_id)
    if layer_key not in value_embeds:
        return None

    value_table = value_embeds[layer_key]

    ve_real = value_table(real_token_ids)
    pred_ids = predicted_tokens.argmax(dim=-1)  # (B, S) 离散化
    ve_pred = value_table(pred_ids)              # 直接查表

    if extra_prefix_tokens > 0:
        batch_size = real_token_ids.shape[0]
        kv_dim = value_table.weight.shape[1]
        prefix = torch.zeros(
            batch_size,
            extra_prefix_tokens,
            kv_dim,
            device=ve_real.device,
            dtype=ve_real.dtype,
        )
        return torch.cat((prefix, ve_real, ve_pred), dim=1)

    return torch.cat((ve_real, ve_pred), dim=1)
