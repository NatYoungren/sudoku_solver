# Nat Youngren
# August 1, 2023
#

import numpy as np
from numba import njit
import json
import time
import timeit

PUZZLE_FILE = 'sudoku_puzzle.json'
ITERATIONS = 100
PREP_TIMEIT = True

VERBOSE_LOOP = True
VERBOSE_END = True

#       Numba speed comparison
# 1000 iterations of ripple_solve:

#       Non-njit:
#  sudokupy    : 0.8660402
#  easy        : 0.0012183
#  medium      : 0.0021823
#  evil        : 0.0045218
#  evil2       : 0.0199788
#  blank       : 0.0221743
# Average time : 0.1526860

#       Njit:
#  sudokupy    : 0.0480540
#  easy        : 0.0000409
#  medium      : 0.0000895
#  evil        : 0.0002152
#  evil2       : 0.0010506
#  blank       : 0.0015018
# Average time : 0.0084920


# # # # # # #
# Setup Utilities
#


def get_region(puzzle, x, y):  # TODO: Faster way to do this
    """ Returns the 3x3 region of the puzzle that contains the given cell.

    Args:
        puzzle (np.ndarray): Sudoku puzzle as a 9x9 numpy array.
        x (int): X coordinate of the cell.
        y (int): Y coordinate of the cell.

    Returns:
        np.ndarray: Flattened 3x3 region of the puzzle.
    """
    x = x // 3
    y = y // 3
    rows = puzzle[x*3:x*3+3]
    cols = rows[:, y*3:y*3+3]
    region = cols.flatten()
    return region


def get_options(row, col, region):  # TODO: Faster way to do this?
    """ Returns all unresolved options (1-9) for a row/column/region combination.

    Args:
        row (np.ndarray): The 9 numerical values in a row.
        col (np.ndarray): The 9 numerical values in a column.
        region (np.ndarray): The 9 numerical values in a region.
                                Empty regions are represented as 0.

    Returns:
        list: List of 1-9 integers not present in any row/column/region.
    """
    taken = set(np.concatenate((row, col, region)).flatten())
    return [i for i in range(1, 10) if i not in taken]


def generate_probability_field(puzzle: np.ndarray):
    """ Generates a probability field for a given puzzle.


    Args:
        puzzle (np.ndarray): A 9x9 numpy array representing a sudoku puzzle.
                            Filled cells contain their 1-9 value, empty cells are represented as 0.

    Returns:
        np.ndarry: A 9x9x9 numpy array, the list at each cell represents the viable options for that cell (index = value-1).
                    Remaining options are represented as 1, removed options are represented as 0.
    """
    puzzle = np.array(puzzle)
    prob_field = np.zeros((9, 9, 9))
    for x, row in enumerate(puzzle):
        for y, col in enumerate(puzzle.T):

            if puzzle[x][y] != 0:
                prob_field[x][y][puzzle[x][y]-1] = 1
                continue

            region = get_region(puzzle, x, y)
            opts = get_options(row, col, region)
            for i in opts:
                prob_field[x][y][i-1] = 1

    return prob_field

# # # # # # #
# Solver Utilities
#


@njit  # Integral to ripple_solve
def collapse_probability_field(prob_field: np.ndarray, x: int, y: int, i: int):
    """ Collapses an x, y cell to a single given value-index (i).
        Perpetuates that change to the rest of the grid by removing the value-index (i) from all other cells in the row, column, and region.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        x (int): X coordinate of the cell to collapse.
        y (int): Y coordinate of the cell to collapse.
        i (int): Index of the value to collapse to. (0-8)

    Returns:
        np.ndarray: Altered copy of the probability field.
    """
    pf = prob_field.copy()
    pf[x, :, i] = 0         # Set option i to 0 for all cells in the x column.
    pf[:, y, i] = 0         # Set option i to 0 for all cells in the y row.
    xx = x // 3             # Set option i to 0 for all cells in the region.
    yy = y // 3             # (xx, yy) is the top-left corner of the region containing x, y.
    pf[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0
    pf[x, y, :] = 0         # Set all options for the x, y cell to 0.
    pf[x][y][i] = 1         # Set the option i for the x, y cell to 1.

    return pf


def make_collapse_map(prob_field: np.ndarray):
    collapse_map = np.zeros((9, 9, 9))
    for x in range(9):
        for y in range(9):

            v = prob_field[x][y].sum()  # NOTE: What is this doing?
            if v == 1:
                continue
            elif v == 0:
                return None

            for i, cell in enumerate(prob_field[x][y]):
                if cell:
                    collapse_map[x][y][i] = evaluate_collapse_sum(prob_field, x, y, i)
    return collapse_map


def evaluate_collapse_sum(prob_field: np.ndarray, x: int, y: int, i: int):
    return evaluate_row_value(prob_field, x, y, i) + evaluate_col_value(prob_field, x, y, i) + evaluate_region_value(prob_field, x, y, i)


def get_collapse_value(prob_field: np.ndarray, x: int, y: int, i: int):
    return min(evaluate_row_value(prob_field, x, y, i), evaluate_col_value(prob_field, x, y, i), evaluate_region_value(prob_field, x, y, i))


def evaluate_row_value(prob_field: np.ndarray, x: int, y: int, i: int):
    return prob_field[x, :, i].sum() - prob_field[x, y, i]


def evaluate_col_value(prob_field: np.ndarray, x: int, y: int, i: int):
    return prob_field[:, y, i].sum() - prob_field[x, y, i]


def evaluate_region_value(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum() - prob_field[x, y, i]


def prob_field_to_puzzle(prob_field: np.ndarray):
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


# # # # # # #
# Solvers
#

#
# Confident Solve
def simple_solve(puzzle: np.ndarray, verbose=False):
    for x, row in enumerate(puzzle):
        for y, col in enumerate(puzzle.T):

            if puzzle[x][y] != 0:
                continue

            region = get_region(puzzle, x, y)
            opts = get_options(row, col, region)

            if len(opts) == 1:
                if verbose:
                    print(f'({x}, {y}) = {opts[0]}')
                puzzle[x][y] = opts[0]

                return simple_solve(puzzle, verbose)

    return puzzle

#
# Probability Field Solver


def collapse_solve(prob_field: np.ndarray, verbose=False):
    counter = 1
    while True:

        c_map = make_collapse_map(prob_field)
        if c_map.sum() == 0:
            break
        min_position = np.argwhere(c_map == np.min(c_map[np.nonzero(c_map)]))
        for x, y, i in min_position:
            if verbose:
                print(f'\tCollapse: {counter} ({x}, {y}) = {i+1}')

            prob_field = collapse_probability_field(prob_field, x, y, i)

            counter += 1
            break

    return prob_field_to_puzzle(prob_field)

#
# Recursive Probability Field Solver


def recursive_collapse_solve(prob_field: np.ndarray, solution, layer=1, verbose=False):
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

    for x, y, i in zip(n_z[0][ordered_indices], n_z[1][ordered_indices], n_z[2][ordered_indices]):
        if verbose:
            print(
                f'<{layer}>\tCollapse: {loop_counter} ({x}, {y}) = {i+1}, conf({c_map[x][y][i]})')

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
# TODO: Try to flatten this out and remove the while loop.
@njit
def ripple_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    """ Each recursion of ripple_solve will collapse every resolved cell until a choice must be made.
        These collapses are propagated to the rest of the grid, possibly resolving or breaking other cells.
        At that point, a new ripple_solve will recursively each explore option of the cell with the lowest number of options.

        A 9x9 grid (collapsed_cells) tracking which cells have bee collapsed is handed down to each recursive call, 
            this is to avoid pointlessly recollapsing every solved cell in the grid each time.

        Most effective solver so far.

    Args:
        prob_field (np.ndarray): A 9x9x9 numpy array representing a sudoku puzzle probability field.
        collapsed_cells (np.ndarray, optional): A 9x9 array, tracking whether each cell position has been collapsed. Defaults to None.
                                                    If None, a blank array will be created.

    Returns:
        np.ndarray, optional: A 9x9x9 numpy array representing a solved sudoku puzzle probability field.
                                If no solution can be found, returns None.
    """
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9))

    # Used to track whether the probability field was altered by the last iteration.
    prev_sum = 0
    while True:
        # Sum the probability field along the value axis.
        resolution_map = prob_field.sum(axis=2)

        # If any cell has no options, the puzzle is unsolvable.
        if not resolution_map.all():
            break

        # Sum the resolution map to see if the puzzle has been altered.
        new_sum = resolution_map.sum()

        # If there is one choice per cell, return the solved probability field.
        if new_sum == 81:
            return prob_field

        # If the puzzle has been altered, look for new cells that can be collapsed.
        if prev_sum != new_sum:
            # Iterate over all cells with only one option.
            resolved_indices = np.argwhere(resolution_map == 1)
            for x, y in resolved_indices:

                # Skip cells that have been previously collapsed.
                # FIXME: Rather than skipping these, we should avoid including them in resolved indices.
                if collapsed_cells[x][y]:
                    continue

                # Collapse the cell to the only option and propagate that change.
                prob_field = collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))

                # Mark the cell as collapsed.
                collapsed_cells[x][y] = 1

        # If the puzzle was not altered, then there are no new cells which can be collapsed.
        # Instead, select the cell with the lowest number of options and recursively solve for each option.
        else:
            # TODO: Find a simpler way to do this.
            unresolved_indices = np.argwhere(resolution_map > 1)
            x, y = unresolved_indices[np.argmin(resolution_map > 1)]
            collapsed_cells[x][y] = 1
            
            # Recurse, passing a collapsed probability field and a copy of the collapsed cells.
            for i in np.where(prob_field[x][y])[0]:
                r = ripple_solve(collapse_probability_field(prob_field, x, y, i), collapsed_cells.copy())
                
                # If a solution is found, return it.
                if r is not None:
                    return r
            # If no option lead to a solution, the puzzle is unsolvable.
            break

        prev_sum = new_sum

    return None



# # # # # # #
# Evaluation
#

def evaluate(puzzles, solver, iterations=10, verbose_loop: bool = True, verbose_end: bool = True):

    times = {}
    for name, puzzle in puzzles.items():
        if verbose_loop:
            print('\n Solving', name, '\n')

        puzzle = generate_probability_field(puzzle)

        # Solve once manually for solution
        start_time = time.time()
        solution = solver(puzzle)
        end_time = time.time()

        # Solve again repeatedly for average time
        t = timeit.timeit(lambda: solver(puzzle), number=iterations)

        times[name] = t / iterations
        if verbose_loop:
            solved_puzzle = prob_field_to_puzzle(solution)
            unsolved_nodes = np.count_nonzero(solved_puzzle == 0)
            print(solved_puzzle)
            print(f'Average time: {t / iterations} seconds.')
            print(
                f'Finished in {end_time - start_time} seconds with {unsolved_nodes} unsolved nodes.')

    if verbose_end:
        print(f'\n\n {iterations} iterations of {solver.__name__}:\n')
        for name, t in times.items():
            print(f' {name:<12}: {t:.7f}')
        print(f'Average time : {sum(times.values()) / len(times):.7f}')


if __name__ == '__main__':

    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)

    if PREP_TIMEIT:
        # Presolve blank puzzle:
        print(' >> Prepping timeit')
        timeit.timeit(lambda: ripple_solve(np.zeros((9, 9, 9))), number=1)

    evaluate(puzzles, ripple_solve, iterations=ITERATIONS,
             verbose_loop=VERBOSE_LOOP)
