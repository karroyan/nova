#!/usr/bin/env python3
"""
持续压测 EBT Web Chat 服务，保证 GPU 显存利用率。

用法（TCP，外网可访问的地址）:
    python scripts/gpu_load_test.py --url http://<host>:10050 --workers 16

用法（Unix socket，绕过透明代理，本机内部使用）:
    python scripts/gpu_load_test.py --socket /tmp/ebt_load.sock --workers 16

每个 worker 循环发请求 → 读完 SSE 流 → 立刻发下一个，
workers 数量建议为 GPU 数 * 2，确保任何时刻所有 GPU 都有任务。
"""

import argparse
import asyncio
import json
import random
import time
import sys
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ── 多样化 prompt 池，避免 KV-cache 副作用 ──────────────────────────────────
_PROMPT_POOL = [
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
    "描述人类大脑处理语言的神经机制，重点介绍布洛卡区和韦尼克区。",
    "写一个关于孤独旅行者在撒哈拉沙漠迷路的短篇故事开头（300字）。",
    "分析《红楼梦》中贾宝玉的人物性格及其在封建社会中的象征意义。",
    "解释双缝实验如何揭示量子力学的波粒二象性。",
    "设计一个简单的分布式缓存系统，说明数据一致性如何保证。",
    "解释自然语言处理中 BPE（字节对编码）分词算法的工作原理。",
    "讨论尼采的'永恒轮回'思想及其对现代存在主义的影响。",
    "用博弈论分析'囚徒困境'，并举出现实中的一个对应案例。",
    "写一段 C++ 代码实现线程安全的生产者-消费者队列。",
    "解释黎曼猜想的基本内容，以及它为什么如此重要。",
    "描述深度学习中梯度消失问题的原因及常用解决方案。",
    "分析第一次世界大战爆发的深层原因，超越'萨拉热窝刺杀'这一导火索。",
    "解释 HTTPS 中 TLS 握手的完整过程，包括证书验证和密钥交换。",
    "写一首关于失去的英文诗，风格参考 Emily Dickinson。",
    "解释卡尔曼滤波的基本原理及其在自动驾驶中的应用。",
    "分析《哈姆雷特》中'to be or not to be'独白的多层含义。",
    "设计一个微服务架构，说明服务发现和负载均衡如何实现。",
    "解释为什么人类对损失的敏感度高于对等量收益的敏感度（前景理论）。",
]


# ── 全局统计 ────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.total_requests = 0
        self.total_tokens = 0
        self.total_errors = 0
        self.active = 0
        self.t_start = time.time()

    async def record(self, tokens: int, error: bool = False):
        async with self.lock:
            self.total_requests += 1
            self.total_tokens += tokens
            if error:
                self.total_errors += 1

    def report(self) -> str:
        elapsed = time.time() - self.t_start
        rps = self.total_requests / max(elapsed, 1)
        tps = self.total_tokens / max(elapsed, 1)
        return (f"elapsed={elapsed:.0f}s  reqs={self.total_requests}"
                f"  errors={self.total_errors}  active={self.active}"
                f"  tok/s={tps:.1f}  req/s={rps:.2f}")


stats = Stats()


def make_client(socket_path: Optional[str]) -> "httpx.AsyncClient":
    """创建 httpx 客户端：有 socket_path 则走 Unix socket，否则走普通 TCP。"""
    headers = {"Accept": "text/event-stream"}
    if socket_path:
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
        return httpx.AsyncClient(transport=transport,
                                 base_url="http://localhost",
                                 headers=headers)
    return httpx.AsyncClient(headers=headers)


# ── 单次请求 ─────────────────────────────────────────────────────────────────
async def do_request(client: "httpx.AsyncClient", url: str, prompt: str) -> int:
    """发送一次请求，流式读取完整响应，返回 token 数量。"""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    token_count = 0
    async with client.stream("POST", url, json=payload, timeout=300.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
                if "choices" in obj:
                    delta = obj["choices"][0].get("delta", {})
                    if delta.get("content"):
                        token_count += 1
                elif obj.get("type") == "token":
                    token_count += 1
                elif obj.get("type") in ("done", "error"):
                    break
            except json.JSONDecodeError:
                pass
    return token_count


# ── Worker 协程 ─────────────────────────────────────────────────────────────
async def worker(worker_id: int, completions_url: str, socket_path: Optional[str],
                 stats: Stats, stop_event: asyncio.Event):
    """无限循环发请求，直到 stop_event 被设置。"""
    rng = random.Random(worker_id * 1337)
    async with make_client(socket_path) as client:
        while not stop_event.is_set():
            prompt = rng.choice(_PROMPT_POOL)
            stats.active += 1
            t0 = time.time()
            try:
                ntok = await do_request(client, completions_url, prompt)
                elapsed = time.time() - t0
                await stats.record(ntok)
                print(f"[W{worker_id:02d}] ok  {ntok:4d} tok  {elapsed:.1f}s  "
                      f"{ntok/max(elapsed,0.1):.0f} tok/s", flush=True)
            except Exception as e:
                elapsed = time.time() - t0
                await stats.record(0, error=True)
                print(f"[W{worker_id:02d}] ERR {elapsed:.1f}s: {e}", flush=True)
                await asyncio.sleep(2.0)
            finally:
                stats.active -= 1


# ── 定时打印总统计 ───────────────────────────────────────────────────────────
async def stats_reporter(stats: Stats, stop_event: asyncio.Event, interval: float = 10.0):
    while not stop_event.is_set():
        await asyncio.sleep(interval)
        print(f"\n{'='*60}", flush=True)
        print(f"  {stats.report()}", flush=True)
        print(f"{'='*60}\n", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="EBT GPU 压测工具")
    parser.add_argument("--url", default=None,
                        help="TCP 服务地址，例如 http://10.140.60.3:10050")
    parser.add_argument("--socket", default=None,
                        help="Unix socket 路径，绕过透明代理，例如 /tmp/ebt_load.sock")
    parser.add_argument("--workers", type=int, default=16,
                        help="并发 worker 数（建议 = GPU数 * 2）")
    parser.add_argument("--duration", type=int, default=0,
                        help="运行秒数（0=无限）")
    args = parser.parse_args()

    if not HAS_HTTPX:
        print("ERROR: httpx 未安装。请运行: pip install httpx", file=sys.stderr)
        sys.exit(1)

    if args.socket is None and args.url is None:
        print("ERROR: 必须指定 --url 或 --socket", file=sys.stderr)
        sys.exit(1)

    socket_path = args.socket
    # Unix socket 模式下，url 只需要路径部分（httpx 忽略 host）
    if socket_path:
        base_url = "http://localhost"
        completions_url = "http://localhost/chat/completions"
        health_url = "http://localhost/health"
        print(f"压测模式: Unix socket  ({socket_path})", flush=True)
    else:
        base_url = args.url.rstrip("/")
        completions_url = base_url + "/chat/completions"
        health_url = base_url + "/health"
        print(f"压测模式: TCP  ({base_url})", flush=True)

    print(f"并发 workers: {args.workers}")
    print(f"运行时长: {'无限' if args.duration == 0 else f'{args.duration}s'}")
    print(f"Prompt 池大小: {len(_PROMPT_POOL)}")
    print("-" * 60, flush=True)

    stop_event = asyncio.Event()

    # 健康检查
    try:
        async with make_client(socket_path) as c:
            r = await c.get(health_url, timeout=5.0)
            r.raise_for_status()
            print(f"服务健康检查通过: {r.status_code}", flush=True)
    except Exception as e:
        print(f"WARNING: 服务健康检查失败: {e}，仍继续压测…", flush=True)

    tasks = []
    for i in range(args.workers):
        tasks.append(asyncio.create_task(
            worker(i, completions_url, socket_path, stats, stop_event)))
    tasks.append(asyncio.create_task(stats_reporter(stats, stop_event)))

    if args.duration > 0:
        await asyncio.sleep(args.duration)
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    else:
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop_event.set()

    print("\n最终统计:")
    print(stats.report())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n中止。", flush=True)
        print(stats.report())
