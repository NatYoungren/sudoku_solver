# Nat Youngren
# August 1, 2023
#

import numpy as np
from copy import deepcopy
import json
import time

puzzle_file = 'sudoku_puzzle.json'

with open(puzzle_file) as f:
    puzzles = json.load(f)


# # # # # # #
# Utilities
#

def get_cell(puzzle, x, y): # TODO: Faster way to do this?
    x = x // 3
    y = y // 3
    rows = puzzle[x*3:x*3+3]
    cols = rows[:,y*3:y*3+3]
    cell = cols.flatten()
    return cell

def get_options(row, col, cell): # TODO: Faster way to do this?
    taken = set(np.concatenate((row, col, cell)).flatten())
    return [i for i in range(1, 10) if i not in taken]


def generate_probability_field(puzzle: np.array):
    puzzle = np.array(puzzle)
    prob_field = np.zeros((9, 9, 9))
    for x, row in enumerate(puzzle):
        for y, col in enumerate(puzzle.T):
            
            if puzzle[x][y] != 0:
                prob_field[x][y][puzzle[x][y]-1] = 1
                continue
            
            cell = get_cell(puzzle, x, y)
            opts = get_options(row, col, cell)
            for i in opts:
                prob_field[x][y][i-1] = 1 # / len(opts)

    return prob_field

def collapse_probability_field(prob_field: np.array, x: int, y: int, i: int):
    pf = deepcopy(prob_field)
    pf = perpetuate_collapse(pf, x, y, i)
    pf[x][y][i] = 1
    
    return pf

def perpetuate_collapse(prob_field: np.array, x: int, y: int, i: int):
    prob_field[x,:,i] = 0
    prob_field[:,y,i] = 0
    xx = x // 3
    yy = y // 3
    prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0
    
    return prob_field

def evaluate_collapse_sum(prob_field: np.array, x: int, y: int, i: int):
    return evaluate_row_value(prob_field, x, y, i) + evaluate_col_value(prob_field, x, y, i) + evaluate_cell_value(prob_field, x, y, i)

def make_collapse_map(prob_field: np.array):
    collapse_map = np.zeros((9, 9, 9))
    for x in range(9):
        for y in range(9):
            for i, cell in enumerate(prob_field[x][y]):
                if cell:
                    collapse_map[x][y][i] = evaluate_collapse_sum(prob_field, x, y, i)
    return collapse_map
                
def get_collapse_value(prob_field: np.array, x: int, y: int, i: int):
    return min(evaluate_row_value(prob_field, x, y, i), evaluate_col_value(prob_field, x, y, i), evaluate_cell_value(prob_field, x, y, i))

def evaluate_row_value(prob_field: np.array, x: int, y: int, i: int):
    return prob_field[x,:,i].sum() - prob_field[x,y,i]

def evaluate_col_value(prob_field: np.array, x: int, y: int, i: int):
    return prob_field[:,y,i].sum() - prob_field[x,y,i]

def evaluate_cell_value(prob_field: np.array, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum() - prob_field[x,y,i]

def prob_field_to_puzzle(prob_field: np.array):
    out_puzzle = np.zeros((9, 9))
    for x, row in enumerate(prob_field):
        for y, col in enumerate(prob_field.T):
            if sum(prob_field[x][y]) == 1:
                out_puzzle[x][y] = np.argmax(prob_field[x][y]) + 1
    return out_puzzle

# # # # # # #
# Solvers
#

# TODO: Implement a solver which uses np.where to find the next cell to solve

#
# Confident Solve
def simple_solve(puzzle: np.array, verbose=False):
    for x, row in enumerate(puzzle):
        for y, col in enumerate(puzzle.T):
            
            if puzzle[x][y] != 0:
                continue
            
            cell = get_cell(puzzle, x, y)
            opts = get_options(row, col, cell)
            
            if len(opts) == 1:
                if verbose: print(f'({x}, {y}) = {opts[0]}')
                puzzle[x][y] = opts[0]
                
                return simple_solve(puzzle, verbose)
    
    return puzzle

#
# Probability Field Solver
def collapse_solve(prob_field: np.array, verbose=False):
    # init_field = deepcopy(prob_field)
    
    counter = 1
    last_sum = prob_field.sum()
    while prob_field.sum() > 81:
        c_map = make_collapse_map(prob_field)
        min_position = np.argwhere(c_map == np.min(c_map[np.nonzero(c_map)]))
        for x, y, i in min_position:
            if verbose: print(f'\tCollapse: {counter} ({x}, {y}) = {i+1}')

            # p_f = deepcopy(prob_field)
            # print(prob_field[0, 4])
            prob_field = collapse_probability_field(prob_field, x, y, i)
            # print(prob_field[0, 4])
            # input()
            # print(np.array_equal(p_f, prob_field))
            
            counter += 1
            # p_p = prob_field_to_puzzle(p_f)
            # p_pp = prob_field_to_puzzle(prob_field)
            # print(np.argwhere(p_p != p_pp))
            # print(prob_field_to_puzzle(prob_field))
            break
    
    
        
    return prob_field_to_puzzle(prob_field)
    # print(make_collapse_map(prob_field))
    # while(sum(prob_field) > 81):
    #     if verbose: print(f'\tCollapse: {counter}')

# # # # # # #
# Evaluation
#

def evaluate(puzzles, solver):
    for name, puzzle in puzzles.items():
        print('\n Solving', name, '\n')
        puzzle = np.array(puzzle)
        
        start_time = time.time()
        solution = solver(puzzle)
        end_time = time.time()
        unsolved_nodes = np.count_nonzero(solution == 0)
        print(solution)
        print(f'Finished in {end_time - start_time} seconds with {unsolved_nodes} unsolved nodes.')
    
# print(simple_solve(np.array(puzzles['easy']), verbose=True))
# evaluate(puzzles, simple_solve)
print(simple_solve(np.array(puzzles['easy']), verbose=True))
prob_field = generate_probability_field(puzzles['easy'])
print(prob_field)
print(collapse_solve(prob_field, verbose=True))
