# Nat Youngren
# August 1, 2023
#

import numpy as np
from numba import njit
import json
import time
import timeit

np.set_printoptions(linewidth=np.inf)

PUZZLE_FILE = 'sudoku_puzzle.json'
ITERATIONS = 100
PREP_TIMEIT = True

VERBOSE_LOOP = True
VERBOSE_END = True



# # # # # # #
# Setup Utilities
#

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


def prob_field_to_puzzle(prob_field: np.ndarray):
    """ Converts a probability field into a puzzle.

    Args:
        prob_field (np.ndarray): 9x9x9 probability field.

    Returns:
        np.array: 9x9 numpy array holding cell values.
                    0 represents an unsolved cell.
                    -1 represents an impossible cell.
    """
    out_puzzle = np.zeros((9, 9))
    for x, _ in enumerate(prob_field):
        for y, _ in enumerate(prob_field.T):
            s = prob_field[x][y].sum()
            if s == 1:
                out_puzzle[x][y] = np.argmax(prob_field[x][y]) + 1
            elif not s:
                out_puzzle[x][y] = -1
    return out_puzzle

# # # # # # #
# Solver Utilities
#

@njit
def propagate_collapse(prob_field: np.ndarray, collapsed_cells: np.ndarray):
    """ Collapses all cells with only one option until no more remain.
        Returns the final state of the probability field.
        Modifies collapsed_cells in place.
    """
    
    collapse_count = 0
    prev_count = collapse_count
    while True:
        # Sum the probability field along the value axis.
        resolution_map = prob_field.sum(axis=2)
        
        # If any cell has no options, the puzzle is unsolvable.
        if not resolution_map.all():
            return None, collapse_count

        # If there is one choice per cell, return the solved probability field.
        if resolution_map.sum() == 81:
            break

        # If the puzzle has been altered, look for new cells that can be collapsed.
        if prev_count != collapse_count: 
            # Iterate over all cells with only one option.
            resolved_indices = np.argwhere(resolution_map == 1)
            for x, y in resolved_indices:
                
                # Skip cells that have been previously collapsed.
                # FIXME: Rather than skipping these, we should avoid including them in resolved indices.
                if collapsed_cells[x][y]:
                    continue
                
                # Abort if the cell has no options.
                if not prob_field[x][y].sum():
                    return None, collapse_count
                    
                # Collapse the cell to the only option and propagate that change.
                i = np.argmax(prob_field[x][y])
                    
                prob_field = collapse_probability_field(prob_field, x, y, i)
                collapse_count += 1

                # Mark the cell as collapsed.
                collapsed_cells[x][y] = 1
        else:
            break
    
    return prob_field, collapse_count


@njit
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
    yy = y // 3             # (xx, yy) is the top-left corner of the 3x3 region containing x, y.
    pf[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0
    pf[x, y, :] = 0         # Set all options for the x, y cell to 0.
    pf[x, y, i] = 1         # Set the option i for the x, y cell to 1.

    return pf


@njit # TODO: Look for a more efficient njit-compatible method
def mask_2darray(arr: np.ndarray, mask_arr: np.ndarray, maskval=10):
    arr = arr.copy()
    for x, _ in enumerate(arr):
        for y, _ in enumerate(arr.T):
            if mask_arr[x][y]:
                arr[x][y] = maskval
    return arr

@njit # Same as above, but in-place.
def mask_2darray_inplace(arr: np.ndarray, mask_arr: np.ndarray, maskval=10):
    w, h = arr.shape[:2]
    for x in range(w):
        for y in range(h):
            if mask_arr[x][y]:
                arr[x][y] = maskval
                
@njit # Same as above, inverse mask.
def inverse_mask_2darray_inplace(arr: np.ndarray, mask_arr: np.ndarray, maskval=10):
    w, h = arr.shape[:2]
    for x in range(w):
        for y in range(h):
            if not mask_arr[x][y]:
                arr[x][y] = maskval

def get_region(puzzle, x, y):  # TODO: Faster way to do this?
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


# # # # # # #
# Heuristic Utilities
#
# NOTE: These utilities are used to weight choices by their relationships with other cells.
#

@njit
def generate_heuristic_maps(prob_field: np.ndarray): # TODO: Consider returning the 3 sets of sums.
    """ Generates heuristics for each cell in the probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.

    Returns:
        collapse_map (np.ndarray), value_map (np.ndarray): 9x9x9 heuristic grids.
            > collapse_map: 9x9x9 grid tracking the minimum number of competing cells for that value in a row/column/region.
            > weight_map: 9x9x9 grid tracking the number of cells affected any given collapse.
    """
    
    row_sums = prob_field.sum(axis=0)
    col_sums = prob_field.sum(axis=1)
    region_sums = np.zeros((9, 9))
    
    # TODO: Make this more efficient?
    for x in range(3):
        for y in range(3):
            region_sums[x*3+y] = prob_field[x*3:x*3+3, y*3:y*3+3].sum(axis=0).sum(axis=0)
    
    # NOTE: Consider using prob_field.copy() multiplying the values.
    #       This would leave all impossible values as 0.
    collapse_map = np.zeros((9, 9, 9))
    weight_map = np.zeros((9, 9, 9))
    
    for x in range(9):
        for y in range(9):
            for i in range(9):
                # NOTE: Consider using sum?
                weight_map[x, y, i] = max(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
                collapse_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
                
    return collapse_map, weight_map # TODO: Revisit these names, swap them?
    
    # for x in range(9):
    #     for y in range(9):
    #         # collapse_map[x, y] = np.minimum(col_sums[x], row_sums[y], region_sums[x//3, y//3])
    #         for i in range(9):
    #             print(collapse_map[x, y, i])
                # collapse_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[x//3, y//3, i])
                # print(x, y, i,'____', collapse_map[x, y, i],'->', collapse_value(prob_field, x, y, i))
                
    print('col', col_sums)
    print('row', row_sums)
    print('reg', region_sums)
    print('map', collapse_map)
    
    return collapse_map

def collapse_sums(prob_field):
    col_sums = prob_field.sum(axis=0)
    row_sums = prob_field.sum(axis=1)
    region_sums = np.zeros((3, 3, 9))
    for x in range(3):
        for y in range(3):
            region_sums[x, y] = prob_field[x*3:x*3+3, y*3:y*3+3].sum(axis=(0, 1))

# Low collapse value means the choice is less likely to lead to a broken board state.
@njit
def collapse_value(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return min(prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum(), prob_field[x, :, i].sum(), prob_field[:, y, i].sum())

# Note that the value of the cell overall included once in the final sum:
#   row + col + (region - row/reg_overlap - col/reg_overlap) = sum
#   cell + cell + (cell - cell - cell) = cell
@njit
def collapse_sum(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    row = prob_field[x, :, i].sum()
    col = prob_field[:, y, i].sum()
    region = prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum() - prob_field[xx*3:xx*3+3, y, i].sum() - prob_field[x, yy*3:yy*3+3, i].sum()
    return row + col + region

# Leaving these alone for future reference, note that they all subtract the cell value from the result.
@njit 
def evaluate_collapse_sum(prob_field: np.ndarray, x: int, y: int, i: int):
    return evaluate_row_value(prob_field, x, y, i) + evaluate_col_value(prob_field, x, y, i) + evaluate_region_value(prob_field, x, y, i)

@njit
def evaluate_row_value(prob_field: np.ndarray, x: int, y: int, i: int):
    return prob_field[x, :, i].sum() - prob_field[x, y, i]

@njit
def evaluate_col_value(prob_field: np.ndarray, x: int, y: int, i: int):
    return prob_field[:, y, i].sum() - prob_field[x, y, i]

@njit
def evaluate_region_value(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum() - prob_field[x, y, i]

# # # # # # #
# Solvers
#

#
# Naive Solve
def naive_solve(puzzle: np.ndarray, verbose=False):
    """ Attempts to solve a sudoku puzzle by iteratively solving cells with only one option.
            Begins again from the beginning whenever a cell is solved.
            If all cells are checked and no more cells are solved, the puzzle is returned as is.
            
            NOTE: This is not a very effective solver, and is included only for comparison.
                  Most puzzles are impossible to solve with this method.

    Args:
        puzzle (np.ndarray): A 9x9 numpy array representing a sudoku puzzle.
        verbose (bool, optional): If True, print each choice as it is made. Defaults to False.

    Returns:
        np.ndarray: A 9x9 numpy array representing the final state of the sudoku puzzle. Oftentimes this is not fully solved.
    """
    
    for x, row in enumerate(puzzle):
        for y, col in enumerate(puzzle.T):
            
            # Skip solved cells.
            if puzzle[x][y] != 0:
                continue
            
            # Determine the options for the unsolved cell.
            region = get_region(puzzle, x, y)
            opts = get_options(row, col, region)

            # If there is only one option, solve the cell and recurse.
            if len(opts) == 1:
                if verbose:
                    print(f'({x}, {y}) = {opts[0]}')
                puzzle[x][y] = opts[0]

                return naive_solve(puzzle, verbose)
    
    return puzzle

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
    
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    collapse_count = 0      # Total number of cells collapsed.
    
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
            return prob_field, recursions, failed_recursions, collapse_count

        # If the puzzle has been altered, look for new cells that can be collapsed.
        if prev_sum != new_sum: 
            # Iterate over all cells with only one option.
            resolved_indices = np.argwhere(resolution_map == 1) # FIXME: By preselecting the cells, we may be missing invalid states.
            for x, y in resolved_indices:
                
                # Skip cells that have been previously collapsed.
                # FIXME: Rather than skipping these, we should avoid including them in resolved indices.
                if collapsed_cells[x][y]:
                    continue
                
                # Abort if the cell has no options.
                if not prob_field[x][y].sum():
                    break
                    
                # Collapse the cell to the only option and propagate that change.
                i = np.argmax(prob_field[x][y])
                    
                prob_field = collapse_probability_field(prob_field, x, y, i)
                collapse_count += 1

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
            indexes = np.where(prob_field[x][y])[0]
            c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
            indexes = [x for _, x in sorted(zip(c_values, indexes), reverse=False)]
            
            for i in indexes:
                
                # Result, recursion_count, failed_recursions
                r, _rs, _frs, _c = ripple_solve(collapse_probability_field(prob_field, x, y, i), collapsed_cells.copy())
                
                recursions += _rs           # Update the tracked metrics.
                failed_recursions += _frs   # NOTE: Used for heuristic testing.
                collapse_count += _c + 1    # 

                # If a solution is found, return it.
                if r is not None:
                    return r,  recursions, failed_recursions, collapse_count
                
                # If no solution is found, increment the failed recursion count.
                failed_recursions += 1
                
            # If no option lead to a solution, the puzzle is unsolvable.
            break

        prev_sum = new_sum

    return None, recursions, failed_recursions, collapse_count



#
# Recursive Heuristic Solver
# This solver is a work in progress, attempting to incorporate more heuristic information into the solver.
# Trying to minimize breaking changes to the probability field.
@njit
def heuristic_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9))
    
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    collapse_count = 0      # Total number of cells collapsed.
    
    # Used to track whether the probability field was altered by the last iteration.
    while not collapsed_cells.all():
        
        # Sum the probability field along the value axis.
        # TODO: Replace resolution mapping with collapse map
        resolution_map = prob_field.sum(axis=2)

        # If any cell has no options, the puzzle is unsolvable.
        if not resolution_map.all():
            prob_field = None
            break
        
        if resolution_map.sum() == 81:
            break
        
        mask_2darray(resolution_map, collapsed_cells)
        
        # c_map = make_collapse_map(prob_field)
        
        # print(resolution_map)
        
        
        
        # print(x, y)
        collapsed_cells[x, y] = 1
        
        #   Old one-cell-per-loop naive collapse.
        # i = np.argmin(resolution_map)
        # x, y = i // 9, i % 9
        
        # if resolution_map[x, y] == 1:
        #     collapse_count += 1
        #     prob_field = collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))
        
        
        while np.min(resolution_map) == 1:
            i = np.argmin(resolution_map)
            x, y = i // 9, i % 9
            collapse_count += 1
            collapsed_cells[x, y] = 1
            prob_field = collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))
            
            resolution_map = prob_field.sum(axis=2)
            mask_2darray(resolution_map, collapsed_cells)
            
        else:
            i = np.argmin(resolution_map)
            x, y = i // 9, i % 9
            
            collapsed_cells[x, y] = 1
            
            indexes = np.where(prob_field[x][y])[0]
            c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
            indexes = [x for _, x in sorted(zip(c_values, indexes), reverse=False)]
            
            for i in indexes:
                
                # Result, recursion_count, failed_recursions
                r, _rs, _frs, _c = heuristic_solve(collapse_probability_field(prob_field, x, y, i), collapsed_cells.copy())
                
                recursions += _rs           # Update the tracked metrics.
                failed_recursions += _frs   # NOTE: Used for heuristic testing.
                collapse_count += _c + 1    # 

                # If a solution is found, return it.
                if r is not None:
                    return r,  recursions, failed_recursions, collapse_count
                                
                # If no solution is found, increment the failed recursion count.
                failed_recursions += 1
                
            # If no option lead to a solution, the puzzle is unsolvable.
            prob_field = None
            break

    return prob_field, recursions, failed_recursions, collapse_count

# # # # # # #
# Evaluation
#

def evaluate(puzzles, solver, iterations=10, verbose_loop: bool = True, verbose_end: bool = True):

    times = {}
    recursions = {}
    failed_recursions = {}
    collapses = {}
    
    for name, puzzle in puzzles.items():
        if verbose_loop:
            print('\n Solving', name, '\n')

        # Generate solver input
        puzzle = generate_probability_field(puzzle)

        # Solve once manually for solution and metrics
        start_time = time.time()
        solution, recursions[name], failed_recursions[name], collapses[name] = solver(puzzle)
        end_time = time.time()
        
        # Solve repeatedly for average time
        t = timeit.timeit(lambda: solver(puzzle), number=iterations)
        times[name] = t / iterations
        
        # Print puzzle results
        if verbose_loop:
            solved_puzzle = prob_field_to_puzzle(solution)
            unsolved_nodes = np.count_nonzero(solved_puzzle == 0)
            print(solved_puzzle)
            print(f'Average time: {t / iterations} seconds.')
            print(f'Finished in {end_time - start_time} seconds with {unsolved_nodes} unsolved nodes.')

    # Print overall results
    if verbose_end:
        print(f'\n\n {iterations} iterations of {solver.__name__}:\n')
        for name, t in times.items():
            r = recursions.get(name, np.nan)
            failed_r = failed_recursions.get(name, np.nan)
            c_count = collapses.get(name, np.nan)
            print(f' {name:<12}: {t:0.7f} ({r:>4} total recursions - failed {failed_r:>4}) [{(100/r)*failed_r:0.2f}%] ({c_count} total collapses)')
        print(f'Average time : {sum(times.values()) / len(times):.7f}')


if __name__ == '__main__':
    TEST_FUNC = heuristic_solve

    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)
        puzzles.pop('ai_escargot', None)
        
    if PREP_TIMEIT:
        # Presolve impossible puzzle:
        print(' >> Prepping timeit')
        
       
        timeit.timeit(lambda: TEST_FUNC(np.zeros((9, 9, 9))), number=1)

    evaluate(puzzles, TEST_FUNC, iterations=ITERATIONS, verbose_loop=VERBOSE_LOOP, verbose_end=VERBOSE_END)
