"""
Sudoku 评估器 — 整盘准确率 (board accuracy) + 单格准确率 (cell accuracy)

用于评估模型在 Sudoku SFT 任务上的表现。
"""

import re
from typing import List, Optional


def parse_board_from_text(text):
    """从模型生成文本中提取 9x9 数字矩阵。

    支持格式:
      - 空格分隔的 9x9 网格 (每行 9 个数字)
      - 连续 81 个数字

    Returns:
        9x9 list of ints, or None if parsing fails
    """
    if text is None:
        return None

    # Try parsing as space-separated 9x9 grid
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) >= 9:
        board = []
        for line in lines[:9]:
            digits = re.findall(r"\d", line)
            if len(digits) >= 9:
                board.append([int(d) for d in digits[:9]])
        if len(board) == 9:
            return board

    # Try parsing as 81 consecutive digits
    all_digits = re.findall(r"\d", text)
    if len(all_digits) >= 81:
        board = []
        for r in range(9):
            board.append([int(all_digits[r * 9 + c]) for c in range(9)])
        return board

    return None


def evaluate_sudoku(predictions, references):
    """Evaluate sudoku predictions against ground truth.

    Args:
        predictions: list of str — model-generated text for each puzzle
        references: list of dict — each with "puzzle" and "solution" (9x9 int lists)

    Returns:
        dict with:
          - board_accuracy: fraction of puzzles solved completely correctly
          - cell_accuracy: fraction of individual cells correct (across all puzzles)
          - n_samples: number of evaluated samples
          - n_parsed: number of predictions successfully parsed as 9x9 boards
    """
    assert len(predictions) == len(references), (
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    n_samples = len(predictions)
    n_parsed = 0
    n_board_correct = 0
    total_cells = 0
    correct_cells = 0

    for pred_text, ref in zip(predictions, references):
        solution = ref["solution"]
        pred_board = parse_board_from_text(pred_text)

        if pred_board is None:
            # Failed to parse — count as all wrong
            total_cells += 81
            continue

        n_parsed += 1
        board_correct = True
        for r in range(9):
            for c in range(9):
                total_cells += 1
                if pred_board[r][c] == solution[r][c]:
                    correct_cells += 1
                else:
                    board_correct = False

        if board_correct:
            n_board_correct += 1

    return {
        "board_accuracy": n_board_correct / n_samples if n_samples > 0 else 0.0,
        "cell_accuracy": correct_cells / total_cells if total_cells > 0 else 0.0,
        "n_samples": n_samples,
        "n_parsed": n_parsed,
    }


def validate_sudoku_board(board):
    """Check if a 9x9 board is a valid sudoku solution.

    Returns True if all rows, columns, and 3x3 boxes contain digits 1-9 exactly once.
    """
    if board is None or len(board) != 9:
        return False

    target = set(range(1, 10))

    # Check rows
    for row in board:
        if len(row) != 9 or set(row) != target:
            return False

    # Check columns
    for c in range(9):
        col = {board[r][c] for r in range(9)}
        if col != target:
            return False

    # Check 3x3 boxes
    for box_r in range(3):
        for box_c in range(3):
            box = set()
            for r in range(box_r * 3, box_r * 3 + 3):
                for c in range(box_c * 3, box_c * 3 + 3):
                    box.add(board[r][c])
            if box != target:
                return False

    return True
