#!/usr/bin/env python3
"""
Sudoku Compare — EBM vs LLMs on Sudoku benchmarks.
"""

import asyncio
import json
import os
import threading
import time
import argparse
import logging

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Sudoku solver (for validation only)
# ═══════════════════════════════════════════════════════════════════════════════

def _possible(board, r, c, n):
    if n in board[r]:
        return False
    if any(board[i][c] == n for i in range(9)):
        return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    return not any(board[br + i][bc + j] == n for i in range(3) for j in range(3))

def _solve(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                for n in range(1, 10):
                    if _possible(board, r, c, n):
                        board[r][c] = n
                        if _solve(board):
                            return True
                        board[r][c] = 0
                return False
    return True

def compute_solution(grid):
    b = [row[:] for row in grid]
    return b if _solve(b) else None

def is_valid_solution(puzzle_grid, sol_grid):
    if not sol_grid:
        return False
    for r in range(9):
        for c in range(9):
            if sol_grid[r][c] == 0:
                return False
            if puzzle_grid[r][c] != 0 and puzzle_grid[r][c] != sol_grid[r][c]:
                return False
    for i in range(9):
        if sorted(sol_grid[i]) != list(range(1, 10)):
            return False
        if sorted(sol_grid[r][i] for r in range(9)) != list(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            box = [sol_grid[br * 3 + i][bc * 3 + j] for i in range(3) for j in range(3)]
            if sorted(box) != list(range(1, 10)):
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ── DATA INTERFACES ─────────────────────────────────────────────────────────
# Replace puzzle grids and LLM results here when real data is available.
# ═══════════════════════════════════════════════════════════════════════════════

# Five hard sudoku puzzles (81-char strings, 0 = blank).
# Replace the strings and names to swap in new puzzles.
_RAW_PUZZLES = [
    ("Puzzle 1 · Hard",
     "006000000"
     "000051000"
     "050700310"
     "000300000"
     "030005720"
     "800200000"
     "070020150"
     "900000000"
     "000007604"),

    ("Puzzle 2 · Hard",
     "003020600"
     "900305001"
     "001806400"
     "008102900"
     "700000008"
     "006708200"
     "002609500"
     "800203009"
     "005010300"),

    ("Puzzle 3 · Hard",
     "200080300"
     "060070084"
     "030500209"
     "000105408"
     "000000000"
     "402706000"
     "301007040"
     "720040060"
     "004010003"),

    ("Puzzle 4 · Hard",
     "000000907"
     "000420180"
     "000705026"
     "100904000"
     "050000040"
     "000507009"
     "920108000"
     "034059000"
     "507000000"),

    ("Puzzle 5 · Hard",
     "030050040"
     "008010500"
     "460000012"
     "070502080"
     "000603000"
     "040109030"
     "250000098"
     "003040600"
     "080060020"),
]

PUZZLES = []
for _name, _flat in _RAW_PUZZLES:
    _grid = [[int(_flat[r * 9 + c]) for c in range(9)] for r in range(9)]
    _sol = compute_solution(_grid)
    PUZZLES.append({"name": _name, "grid": _grid, "solution": _sol})
    logger.info(f"Loaded {_name}: solution={'OK' if _sol else 'FAILED'}")


def get_llm_offline_result(llm_id: int, puzzle_id: int) -> dict:
    """
    ── INTERFACE: replace with real offline LLM results ──────────────────────

    Args:
        llm_id   : 1–5
        puzzle_id: 0–4

    Returns:
        {
            "text": str,                         # full model output text
            "grid": list[list[int]] | None,      # parsed 9×9 solution, or None
        }

    The returned grid is compared cell-by-cell against the known correct solution.
    """
    solution = PUZZLES[puzzle_id]["solution"]

    # MOCK texts — replace with real precomputed outputs
    mock_texts = {
        1: (
            "I'll solve this step by step using constraint propagation.\n\n"
            "Row analysis complete. Column analysis complete.\n"
            "Box constraints applied. Unique solution found.\n\n"
        ),
        2: (
            "Analyzing the puzzle using naked singles and hidden pairs...\n"
            "Iteration 1: 12 cells resolved.\n"
            "Iteration 2: 18 cells resolved.\n"
            "Iteration 3: Final 9 cells resolved.\n\n"
        ),
        3: (
            "Applying backtracking search with forward checking.\n"
            "Branching factor reduced by arc consistency.\n"
            "Solution found after 847 node expansions.\n\n"
        ),
        4: (
            "Let me carefully work through each row and column...\n"
            "Using elimination: row 1 → [3,7,9], row 2 → [2,4,6]...\n"
            "Cross-referencing boxes... complete.\n\n"
        ),
        5: (
            "Sudoku constraint satisfaction problem.\n"
            "Applying MRV heuristic + degree heuristic.\n"
            "Search complete.\n\n"
        ),
    }

    text = mock_texts.get(llm_id, "Solving...\n\n")

    # MOCK correctness: LLM 1, 3, 5 correct; LLM 2, 4 wrong
    # Replace this block with real precomputed grids.
    if solution is None:
        grid = None
    elif llm_id in (1, 3, 5):
        grid = [row[:] for row in solution]
        text += "\n".join("".join(str(d) for d in row) for row in solution)
    else:
        # Introduce a deliberate error to show incorrect coloring
        grid = [row[:] for row in solution]
        # swap two non-given cells to make it wrong
        puzzle_grid = PUZZLES[puzzle_id]["grid"]
        swapped = False
        for r in range(9):
            for c in range(8):
                if puzzle_grid[r][c] == 0 and puzzle_grid[r][c + 1] == 0 and not swapped:
                    grid[r][c], grid[r][c + 1] = grid[r][c + 1], grid[r][c]
                    swapped = True
        text += "\n".join("".join(str(d) for d in row) for row in grid)

    return {"text": text, "grid": grid}


async def stream_ebm_result(puzzle_id: int):
    """
    Async generator yielding (event_type, data) tuples:
        ("token", str)                        — streaming text chunk
        ("done",  {grid, correct, elapsed})   — final result
        ("error", str)                        — error message
    """
    t0 = time.time()
    flat81 = "".join(str(v) for row in PUZZLES[puzzle_id]["grid"] for v in row)
    puzzle_text = _flat_to_grid_text(flat81)
    prompt = _SUDOKU_PROMPT.format(puzzle=puzzle_text)

    if _engine is None:
        # mock mode
        for tok in ["(mock) EBM 推理中...\n", "解题完成。\n"]:
            yield ("token", tok)
            await asyncio.sleep(0.15)
        sol = PUZZLES[puzzle_id]["solution"]
        yield ("done", {"grid": sol, "correct": sol is not None,
                        "elapsed": round(time.time() - t0, 2)})
        return

    # 真实推理：在线程里跑同步 generate_stream，通过 asyncio.Queue 传 token
    loop = asyncio.get_event_loop()
    token_q: asyncio.Queue = asyncio.Queue()
    full_text = []

    def _run():
        try:
            for tok in _engine.generate_stream(prompt, max_tokens=300, temperature=0.0):
                full_text.append(tok)
                loop.call_soon_threadsafe(token_q.put_nowait, tok)
        except Exception as e:
            loop.call_soon_threadsafe(token_q.put_nowait, RuntimeError(str(e)))
        finally:
            loop.call_soon_threadsafe(token_q.put_nowait, None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    try:
        while True:
            item = await token_q.get()
            if item is None:
                break
            if isinstance(item, RuntimeError):
                yield ("error", str(item))
                return
            yield ("token", item)
    finally:
        t.join(timeout=30)

    response = "".join(full_text)
    flat = _parse_board(response)
    if flat is not None:
        grid = [[flat[r * 9 + c] for c in range(9)] for r in range(9)]
        correct = is_valid_solution(PUZZLES[puzzle_id]["grid"], grid)
    else:
        grid = None
        correct = False

    yield ("done", {"grid": grid, "correct": correct,
                    "elapsed": round(time.time() - t0, 2)})


# ═══════════════════════════════════════════════════════════════════════════════
# EBM engine (loaded once at startup, None if no --checkpoint given)
# ═══════════════════════════════════════════════════════════════════════════════

_SUDOKU_PROMPT = (
    "Solve this sudoku puzzle. Replace each 0 with the correct digit (1-9):\n{puzzle}"
)

def _flat_to_grid_text(flat81: str) -> str:
    return "\n".join(
        " ".join(flat81[r * 9 + c] for c in range(9))
        for r in range(9)
    )

def _parse_board(text: str):
    nums = [int(x) for x in text.split() if x.isdigit() and len(x) == 1]
    if len(nums) >= 81:
        return nums[-81:]
    nums = [int(c) for c in text if c.isdigit()]
    if len(nums) >= 81:
        return nums[-81:]
    return None


class _SudokuEngine:
    """Minimal wrapper around the EBT model for sudoku inference."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
        from openebm.elm.generate import call_model_forward_decode, _get_tokenizer, sample_top_p
        from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
        from openebm.elm.modeling_ebt import EBT_NLP

        self.device = device
        self._call_model = call_model_forward_decode
        self._sample_top_p = sample_top_p

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hparams = checkpoint["hyper_parameters"]
        self.hparams = hparams

        model = EBT_NLP(hparams)
        sd = checkpoint["state_dict"]
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model.to(torch.bfloat16).to(device).eval()
        self.model = model

        self.tokenizer = NanoChatTokenizerWrapper(tokenizer_dir=tokenizer_path)
        inner = self.tokenizer.tokenizer
        self._bos = inner.get_bos_token_id()
        self._user_start = inner.encode_special("<|user_start|>")
        self._user_end = inner.encode_special("<|user_end|>")
        self._asst_start = inner.encode_special("<|assistant_start|>")
        self._asst_end = inner.encode_special("<|assistant_end|>")
        self._inner_tok = inner

        stop = set()
        for sp in ("<|assistant_end|>", "<|user_start|>", "<|user_end|>"):
            sid = inner.encode_special(sp)
            if sid is not None:
                stop.add(sid)
        self._stop_ids = stop
        logger.info(f"[SudokuEngine] loaded on {device}  stop_ids={stop}")

    def generate_stream(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0, top_p: float = 0.9):
        """Yield decoded text tokens one by one (runs in a thread)."""
        inner = self._inner_tok
        ids = [self._bos, self._user_start] + inner.encode(prompt) + [self._user_end, self._asst_start]

        ctx_len = getattr(self.hparams, "context_length", getattr(self.hparams, "max_seq_len", 2048))
        pad_id = self._bos
        total_len = min(ctx_len, len(ids) + max_tokens)

        tokens = torch.full((1, total_len), pad_id, dtype=torch.long, device=self.device)
        tokens[0, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        input_mask = torch.zeros(1, total_len, dtype=torch.bool, device=self.device)
        input_mask[0, :len(ids)] = True

        prev_pos = 0
        eos_reached = torch.tensor([False], device=self.device)

        with torch.no_grad():
            for cur_pos in range(len(ids), total_len):
                logits = self._call_model(self.model, tokens, prev_pos, cur_pos)
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tok = self._sample_top_p(probs, top_p)
                else:
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                next_tok = next_tok.reshape(-1)
                next_tok = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_tok)
                tokens[:, cur_pos] = next_tok
                prev_pos = cur_pos
                eos_reached |= (~input_mask[:, cur_pos]) & (torch.isin(next_tok, torch.tensor(list(self._stop_ids), device=self.device)))
                if not input_mask[0, cur_pos]:
                    tok_id = next_tok[0].item()
                    if tok_id not in self._stop_ids:
                        yield inner.decode([tok_id])
                if eos_reached.all():
                    break


_engine: "_SudokuEngine | None" = None


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=10051)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="EBT sudoku checkpoint path; omit for mock mode")
parser.add_argument("--tokenizer", type=str,
                    default="/mnt/petrelfs/lixueyan/nar/tokenizer",
                    help="Tokenizer directory")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    if args.checkpoint:
        logger.info(f"[SudokuCompare] Loading EBM from {args.checkpoint} ...")
        _engine = _SudokuEngine(args.checkpoint, args.tokenizer, args.device)
        logger.info("[SudokuCompare] EBM engine ready")
    else:
        logger.info("[SudokuCompare] No --checkpoint given, running in mock mode")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/puzzles")
async def api_puzzles():
    return [{"id": i, "name": p["name"]} for i, p in enumerate(PUZZLES)]


@app.get("/api/puzzle/{puzzle_id}")
async def api_puzzle(puzzle_id: int):
    if puzzle_id >= len(PUZZLES):
        return {"error": "invalid puzzle_id"}
    p = PUZZLES[puzzle_id]
    return {"id": puzzle_id, "name": p["name"], "grid": p["grid"], "solution": p["solution"]}


@app.get("/api/solve/ebm/{puzzle_id}")
async def api_solve_ebm(puzzle_id: int):
    if puzzle_id >= len(PUZZLES):
        return {"error": "invalid puzzle_id"}

    async def stream():
        async for etype, data in stream_ebm_result(puzzle_id):
            yield f"data: {json.dumps({'type': etype, 'data': data})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/solve/llm/{llm_id}/{puzzle_id}")
async def api_solve_llm(llm_id: int, puzzle_id: int):
    if puzzle_id >= len(PUZZLES) or not (1 <= llm_id <= 5):
        return {"error": "invalid parameters"}

    result = get_llm_offline_result(llm_id, puzzle_id)
    correct = is_valid_solution(PUZZLES[puzzle_id]["grid"], result["grid"])

    async def stream():
        text = result["text"]
        # Stream text in small chunks to simulate typing
        chunk = 6
        for i in range(0, len(text), chunk):
            yield f"data: {json.dumps({'type': 'token', 'data': text[i:i+chunk]})}\n\n"
            await asyncio.sleep(0.018)
        elapsed = round(len(text) * 0.004, 2)
        yield f"data: {json.dumps({'type': 'done', 'data': {'grid': result['grid'], 'correct': correct, 'elapsed': elapsed}})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    return HTMLResponse(content=_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML / CSS / JS
# ═══════════════════════════════════════════════════════════════════════════════

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EBT Sudoku</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: #fff; color: #111; }
body { font-family: 'Courier New', Courier, monospace; font-size: 14px; display: flex; flex-direction: column; min-height: 100vh; }

/* ── Header ── */
.header { padding: 1.1rem 2rem; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e5e7eb; }
.header-title { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }
.ebm-badge { background: #111; color: #fff; font-size: 0.75rem; font-weight: 700; padding: 0.3rem 0.75rem; border-radius: 4px; letter-spacing: 0.05em; }

/* ── Subtitle ── */
.subtitle { padding: 1rem 2rem 0; color: #374151; line-height: 1.6; }
.subtitle p { max-width: 900px; }
.subtitle p + p { margin-top: 0.3rem; font-size: 0.85rem; color: #6b7280; }

/* ── Main layout ── */
.main { display: flex; flex: 1; padding: 1.5rem 2rem; gap: 2.5rem; }

/* ── Left panel ── */
.left-panel { width: 380px; flex-shrink: 0; display: flex; flex-direction: column; gap: 1.2rem; }
.section-label { font-weight: 700; font-size: 0.8rem; margin-bottom: 0.5rem; letter-spacing: 0.03em; }

.puzzle-select { width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #d1d5db; font-family: inherit; font-size: 0.9rem; background: #fff; cursor: pointer; outline: none; }
.puzzle-select:focus { border-color: #111; }

.btn { width: 100%; padding: 0.65rem 1rem; font-family: inherit; font-size: 0.9rem; font-weight: 600; border: 1px solid #111; cursor: pointer; text-align: center; transition: background 0.15s, color 0.15s; }
.btn-outline { background: #fff; color: #111; }
.btn-outline:hover { background: #f3f4f6; }
.btn-solid { background: #111; color: #fff; }
.btn-solid:hover { background: #374151; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }

/* ── Puzzle preview grid ── */
.preview-wrap { }
.sudoku-table { border-collapse: collapse; }
.sudoku-table td {
    text-align: center; vertical-align: middle;
    border: 1px solid #ccc;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    user-select: none;
}
.sudoku-table td.box-top    { border-top:    2px solid #111; }
.sudoku-table td.box-left   { border-left:   2px solid #111; }
.sudoku-table td.box-bottom { border-bottom: 2px solid #111; }
.sudoku-table td.box-right  { border-right:  2px solid #111; }
.cell-given   { font-weight: 700; color: #111; background: #fff; }
.cell-correct { color: #166534; background: #dcfce7; }
.cell-wrong   { color: #991b1b; background: #fee2e2; }
.cell-blank   { color: #d1d5db; background: #f9fafb; }

/* ── Right panel ── */
.right-panel { flex: 1; }
.results-header { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em; margin-bottom: 0.75rem; }
.cards-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }

/* ── Model card ── */
.model-card { border: 1px solid #e5e7eb; padding: 0.75rem 1rem; background: #fff; }
.model-card.ebm-card { border-color: #111; }

.card-header { display: flex; align-items: center; justify-content: space-between; }
.card-name { font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 0.4rem; }
.ebm-star { color: #111; }
.card-status { font-size: 0.8rem; color: #6b7280; }
.card-status.running { color: #2563eb; }
.card-status.correct { color: #166534; }
.card-status.wrong   { color: #991b1b; }
.card-status.error   { color: #991b1b; }

.card-output {
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: #374151;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 110px;
    overflow-y: auto;
    line-height: 1.5;
}
.card-grid { margin-top: 0.6rem; }

/* ── Footer ── */
.footer { padding: 1rem 2rem; border-top: 1px solid #e5e7eb; display: flex; justify-content: space-between; color: #9ca3af; font-size: 0.75rem; }
</style>
</head>
<body>

<div class="header">
  <div class="header-title">EBT Sudoku</div>
  <div class="ebm-badge">EBM</div>
</div>

<div class="subtitle">
  <p>Compare our EBM reasoning model against frontier LLMs on hard Sudoku puzzles.</p>
  <p>Select a puzzle, then click Compare to run all models simultaneously.</p>
</div>

<div class="main">
  <!-- Left panel -->
  <div class="left-panel">
    <div>
      <div class="section-label">Sudoku Puzzle</div>
      <select class="puzzle-select" id="puzzleSelect" onchange="onPuzzleChange(this.value)"></select>
    </div>

    <button class="btn btn-solid" id="compareBtn" onclick="compareAll()">✦ Compare All Models</button>

    <div class="preview-wrap">
      <div class="section-label">Puzzle Preview</div>
      <div id="puzzlePreview"></div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-panel">
    <div class="results-header">Results</div>
    <div class="cards-grid" id="cardsGrid"></div>
  </div>
</div>

<div class="footer">
  <span>EBT Sudoku Compare</span>
  <span>©</span>
</div>

<script>
const MODELS = [
  { id: 'ebm', name: 'EBT 1.0 EBM', ebm: true },
  { id: 'llm1', name: 'LLM 1', ebm: false },
  { id: 'llm2', name: 'LLM 2', ebm: false },
  { id: 'llm3', name: 'LLM 3', ebm: false },
  { id: 'llm4', name: 'LLM 4', ebm: false },
  { id: 'llm5', name: 'LLM 5', ebm: false },
];

let puzzles = [];
let currentPuzzle = null;   // { id, name, grid, solution }
let isRunning = false;

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  const sel = document.getElementById('puzzleSelect');
  const resp = await fetch('/api/puzzles');
  puzzles = await resp.json();
  puzzles.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.name;
    sel.appendChild(opt);
  });
  buildCards();
  await selectPuzzle(0);
}

// ── Card scaffolding ──────────────────────────────────────────────────────────

function buildCards() {
  const grid = document.getElementById('cardsGrid');
  grid.innerHTML = '';
  MODELS.forEach(m => {
    const card = document.createElement('div');
    card.className = 'model-card' + (m.ebm ? ' ebm-card' : '');
    card.id = 'card-' + m.id;
    card.innerHTML = `
      <div class="card-header">
        <div class="card-name">${m.ebm ? '<span class="ebm-star">✦</span>' : ''} ${m.name}</div>
        <div class="card-status" id="status-${m.id}">Ready</div>
      </div>
      <div class="card-output" id="output-${m.id}" style="display:none"></div>
      <div class="card-grid"  id="grid-${m.id}"></div>
    `;
    grid.appendChild(card);
  });
}

function resetCards() {
  MODELS.forEach(m => {
    setStatus(m.id, 'ready', 'Ready');
    const out = document.getElementById('output-' + m.id);
    out.textContent = '';
    out.style.display = 'none';
    document.getElementById('grid-' + m.id).innerHTML = '';
  });
}

function setStatus(modelId, state, text) {
  const el = document.getElementById('status-' + modelId);
  el.textContent = text;
  el.className = 'card-status ' + state;
}

// ── Puzzle select ─────────────────────────────────────────────────────────────

async function onPuzzleChange(id) {
  await selectPuzzle(parseInt(id));
}

async function selectPuzzle(id) {
  const resp = await fetch('/api/puzzle/' + id);
  currentPuzzle = await resp.json();
  renderPreview(currentPuzzle.grid);
  resetCards();
}

// ── Grid rendering ────────────────────────────────────────────────────────────

function makeGrid(puzzleGrid, modelGrid, solution, cellPx) {
  const table = document.createElement('table');
  table.className = 'sudoku-table';
  for (let r = 0; r < 9; r++) {
    const tr = document.createElement('tr');
    for (let c = 0; c < 9; c++) {
      const td = document.createElement('td');
      td.style.width = td.style.height = cellPx + 'px';
      td.style.fontSize = Math.round(cellPx * 0.48) + 'px';

      const classes = [];
      if (r % 3 === 0) classes.push('box-top');
      if (c % 3 === 0) classes.push('box-left');
      if (r === 8)      classes.push('box-bottom');
      if (c === 8)      classes.push('box-right');

      const given = puzzleGrid[r][c];
      if (given !== 0) {
        classes.push('cell-given');
        td.textContent = given;
      } else if (modelGrid) {
        const filled = modelGrid[r][c];
        if (filled && filled !== 0) {
          const correct = solution && solution[r][c] === filled;
          classes.push(correct ? 'cell-correct' : 'cell-wrong');
          td.textContent = filled;
        } else {
          classes.push('cell-blank');
          td.textContent = '·';
        }
      }
      td.className = classes.join(' ');
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  return table;
}

function renderPreview(grid) {
  const el = document.getElementById('puzzlePreview');
  el.innerHTML = '';
  el.appendChild(makeGrid(grid, null, null, 34));
}

function renderCardGrid(modelId, modelGrid) {
  const el = document.getElementById('grid-' + modelId);
  el.innerHTML = '';
  if (!currentPuzzle) return;
  el.appendChild(makeGrid(currentPuzzle.grid, modelGrid, currentPuzzle.solution, 24));
}

// ── Compare ───────────────────────────────────────────────────────────────────

async function compareAll() {
  if (isRunning || !currentPuzzle) return;
  isRunning = true;
  document.getElementById('compareBtn').disabled = true;
  resetCards();

  const pid = currentPuzzle.id;
  const promises = MODELS.map(m => runModel(m, pid));
  await Promise.allSettled(promises);

  isRunning = false;
  document.getElementById('compareBtn').disabled = false;
}

async function runModel(model, puzzleId) {
  const url = model.ebm
    ? `/api/solve/ebm/${puzzleId}`
    : `/api/solve/llm/${model.id.replace('llm', '')}/${puzzleId}`;

  const outEl = document.getElementById('output-' + model.id);
  outEl.textContent = '';
  outEl.style.display = 'block';

  const t0 = Date.now();
  setStatus(model.id, 'running', 'Running…');

  try {
    const resp = await fetch(url);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let msg;
        try { msg = JSON.parse(line.slice(6)); } catch { continue; }

        if (msg.type === 'token') {
          outEl.textContent += msg.data;
          outEl.scrollTop = outEl.scrollHeight;
          // live timer
          setStatus(model.id, 'running', `Running… ${((Date.now()-t0)/1000).toFixed(1)}s`);

        } else if (msg.type === 'done') {
          const d = msg.data;
          outEl.style.display = 'none';
          if (d.grid) {
            renderCardGrid(model.id, d.grid);
            const label = `Done in ${d.elapsed}s ${d.correct ? '✓' : '✗'}`;
            setStatus(model.id, d.correct ? 'correct' : 'wrong', label);
          } else {
            setStatus(model.id, 'error', 'No solution ✗');
            outEl.style.display = 'block';
          }

        } else if (msg.type === 'error') {
          outEl.style.display = 'block';
          outEl.textContent += '\n[Error] ' + msg.data;
          setStatus(model.id, 'error', 'Error ✗');
        }
      }
    }
  } catch (e) {
    setStatus(model.id, 'error', 'Error ✗');
    outEl.style.display = 'block';
    outEl.textContent = e.message;
  }
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
