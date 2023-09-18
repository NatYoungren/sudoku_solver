# Nat Youngren
# August 1, 2023
#

import numpy as np
# from numba import njit
import json
import time
import timeit

puzzle_file = 'sudoku_puzzle.json'

with open(puzzle_file) as f:
    puzzles = json.load(f)


# # # # # # #
# Utilities
#

# @njit # 0.1608099
def get_cell(puzzle, x, y): # TODO: Faster way to do this?
    x = x // 3
    y = y // 3
    rows = puzzle[x*3:x*3+3]
    cols = rows[:,y*3:y*3+3]
    cell = cols.flatten()
    return cell


# @njit # 0.1649119
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

# @njit
def collapse_probability_field(prob_field: np.array, x: int, y: int, i: int): # Make copy bool param
    pf = prob_field.copy()
    pf = perpetuate_collapse(pf, x, y, i)
    pf[x,y,:] = 0
    pf[x][y][i] = 1
    
    return pf

# @njit
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
    # vs = prob_field.sum(axis=2)
    # print(vs)
    for x in range(9):
        for y in range(9):
            
            v = prob_field[x][y].sum() # NOTE: What is this doing?
            if v == 1:
                continue
            elif v == 0:
                return None
            
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
            if prob_field[x][y].sum() == 1:
                out_puzzle[x][y] = np.argmax(prob_field[x][y]) + 1
            elif prob_field[x][y].sum() > 1:
                # print('UNRESOLVED', x, y, prob_field[x][y])
                out_puzzle[x][y] = -1
            # elif prob_field[x][y].sum() == 0:
                # print('OVERRESOLVED', x, y, prob_field[x][y])
    return out_puzzle

def validate_collapse_map(prob_field: np.array):
    row_sums = prob_field.sum(axis=1)
    print(row_sums)
    col_sums = prob_field.sum(axis=0)
    print(col_sums)
    cell_sums = prob_field.sum(axis=2)



# # # # # # #
# Solvers
#

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
    counter = 1
    while True:
        
        c_map = make_collapse_map(prob_field)
        if c_map.sum() == 0:
            break
        min_position = np.argwhere(c_map == np.min(c_map[np.nonzero(c_map)]))
        for x, y, i in min_position:
            if verbose: print(f'\tCollapse: {counter} ({x}, {y}) = {i+1}')

            prob_field = collapse_probability_field(prob_field, x, y, i)

            counter += 1
            break
        
    return prob_field_to_puzzle(prob_field)

#
# Recursive Probability Field Solver
def recursive_collapse_solve(prob_field: np.array, solution, layer=1, verbose=False):
    loop_counter = 1

    
    c_map = make_collapse_map(prob_field)
    if c_map is None:
        print('no map')
        return None
    
    if c_map.sum() == 0:
        print('zero sum')
        if prob_field.sum() == 81:
            print('solved')
            return prob_field
        return None
    
    result = prob_field.copy()

    n_z = np.nonzero(c_map)
    n_z_v = c_map[n_z]
    ordered_indices = np.argsort(n_z_v)
    
    # reverse_indices = ordered_indices[::-1]
    # for x, y, i in zip(n_z[0][reverse_indices], n_z[1][reverse_indices], n_z[2][reverse_indices]):
    #     print(f'\t<{layer}>\tCollapse: {loop_counter} ({x}, {y}) = {i+1}, conf({c_map[x][y][i]})')
        
    for x, y, i in zip(n_z[0][ordered_indices], n_z[1][ordered_indices], n_z[2][ordered_indices]):
        if verbose: print(f'<{layer}>\tCollapse: {loop_counter} ({x}, {y}) = {i+1}, conf({c_map[x][y][i]})')
        
        result = collapse_probability_field(prob_field, x, y, i)
        
        result = recursive_collapse_solve(result, solution, layer+1, verbose)
        
        if result is not None:
            return result
        print('removing option', i+1, 'from', x, y)
        prob_field[x][y][i] = 0
    # else:
    #     return result


#
# Recursive Ripple Solver
def ripple_solve(prob_field: np.array, resolved=None, verbose=False):
    if resolved is None:
        resolved = np.zeros((9, 9))
    prev_sum = 0
    while True:
        resolution_map = prob_field.sum(axis=2)
        # print(resolution_map, resolution_map.sum())
        if not resolution_map.all():
            return None
        
        new_sum = resolution_map.sum()
        if new_sum == 81:
            break
        
        # print(np.where(resolution_map == resolved))
        
        if prev_sum != new_sum:
            resolved_indices = np.argwhere(resolution_map == 1)
            for x, y in resolved_indices:
                if resolved[x][y]:
                    continue
                resolved[x][y] = 1
                prob_field = collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))
        else:
            # v = np.argmin(resolution_map[resolution_map > 1])
            # x = v // 9
            # y = v % 9
            
            unresolved_indices = np.argwhere(resolution_map > 1)
            x, y = unresolved_indices[np.argmin(resolution_map > 1)]            
            for i in np.where(prob_field[x][y])[0]:
                # print('recursive')
                r = ripple_solve(collapse_probability_field(prob_field, x, y, i), resolved.copy(), verbose=verbose)# resolved=resolved, verbose=verbose)
                if r is not None:
                    return r
            return None
        
        prev_sum = new_sum
        
    return prob_field

# # # # # # #
# Evaluation
#

def evaluate(puzzles, solver, iterations=10, verbose_loop=False):
    
    times = {}
    overall_st = time.time()
    for name, puzzle in puzzles.items():
        if verbose_loop: print('\n Solving', name, '\n')
        puzzle = generate_probability_field(puzzle)
        
        # Solve once to frontload numba compilation
        start_time = time.time()
        solution = solver(puzzle)
        end_time = time.time()
        
        
        
        #  
        t = timeit.timeit(lambda: solver(puzzle), number=iterations)
        
        times[name] = t / iterations
        if verbose_loop:
            solved_puzzle = prob_field_to_puzzle(solution)
            unsolved_nodes = np.count_nonzero(solved_puzzle == 0)
            print(solved_puzzle)
            print(f'Average time: {t / iterations} seconds.')
            print(f'Finished in {end_time - start_time} seconds with {unsolved_nodes} unsolved nodes.')
            
        
    
    print(f'\n\n {iterations} iterations of {solver.__name__}:\n')
    for name, t in times.items():
        print(f' {name:<12}: {t:.7f}')
    print(f'Average time : {sum(times.values()) / len(times):.7f}')
    print(f'Overall time : {time.time() - overall_st:.7f}')
    
evaluate(puzzles, ripple_solve)
# prob_field = generate_probability_field(puzzles['evil2'])

# print(prob_field_to_puzzle(ripple_solve_blank(prob_field, verbose=True)))
