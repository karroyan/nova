import torch
import random

data_dir = 'openebm/elm/data/sudoku_cache_v2'
splits = {
    'Train (rrn_train.pt)': torch.load(f'{data_dir}/rrn_train.pt', weights_only=False),
    'Validation (rrn_val.pt)': torch.load(f'{data_dir}/rrn_val.pt', weights_only=False),
    'Test (satnet_test.pt)': torch.load(f'{data_dir}/satnet_test.pt', weights_only=False),
}

PROMPT_TEMPLATES = [
    'Solve this sudoku puzzle. Replace each 0 with the correct digit (1-9):\n{puzzle}',
    'Complete the following sudoku grid. Empty cells are marked as 0:\n{puzzle}',
    'Fill in the missing numbers in this 9x9 sudoku:\n{puzzle}',
    'Here is a sudoku puzzle where 0 represents an empty cell. Find the solution:\n{puzzle}',
    'I need help solving this sudoku. Each row, column, and 3x3 box must contain digits 1-9 exactly once:\n{puzzle}',
]
RESPONSE_TEMPLATES = [
    '{solution}',
    'Here is the completed sudoku:\n{solution}',
    'The solution is:\n{solution}',
]

def format_board(board):
    return '\n'.join(' '.join(str(c) for c in row) for row in board)

for split_name, samples in splits.items():
    print('=' * 80)
    print(f'  {split_name}  (total: {len(samples)} samples)')
    print('=' * 80)
    for i in range(2):
        s = samples[i]
        puzzle = s['puzzle']
        solution = s['solution']
        puzzle_str = format_board(puzzle)
        solution_str = format_board(solution)
        prompt = PROMPT_TEMPLATES[0].format(puzzle=puzzle_str)
        response = RESPONSE_TEMPLATES[0].format(solution=solution_str)
        hints = sum(1 for row in puzzle for c in row if c != 0)
        print(f'\n--- Example {i+1} (hints: {hints}) ---')
        print(f'[USER]')
        print(prompt)
        print(f'\n[ASSISTANT]')
        print(response)
        print()
    print()