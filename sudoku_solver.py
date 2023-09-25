# Nat Youngren
# August 1, 2023
#

import numpy as np
from numba import njit
import json

from setup_utils import get_region, get_options, generate_probability_field, prob_field_to_puzzle, validate_solution
from solver_utils import collapse_probability_field, propagate_collapse, mask_2darray, mask_2darray_inplace, inverse_mask_2darray_inplace
from heuristic_utils import generate_heuristic_maps, collapse_value

# # # # # # #
# Solvers
#

#
# Naive Solve
# NOTE: Not compatible with evaluate.
def naive_solve(puzzle: np.ndarray, verbose=False):
    """ Attempts to solve a sudoku puzzle by iteratively solving cells with only one option.
            Begins again from the beginning whenever a cell is solved.
            If all cells are checked and no more cells are solved, the puzzle is returned as is.
            
            NOTE:   This is not an effective or complete solver, and is included only for comparison.
                    naive_solve was my starting point to get a feel for the problem.
                    Most puzzles are impossible to solve with this method.
            
            NOTE:   solver_utils.propagate_collapse is an optimized probability field version of this method.
                    Naively collapsing single-option cells turned out to be a useful component in future solvers.
    
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
        At that point, a new ripple_solve will recursively explore each option of the cell with the lowest number of options.

        A 9x9 grid (collapsed_cells) tracking which cells have been collapsed is handed down to each recursive call, 
            this is to avoid pointlessly recollapsing every solved cell in the grid each time.

        First effective solver.

    Args:
        prob_field (np.ndarray): A 9x9x9 numpy array representing a sudoku puzzle probability field.
        collapsed_cells (np.ndarray, optional): A 9x9 array, tracking whether each cell position has been collapsed. Defaults to None.
                                                    If None, a blank array will be created.

    Returns:
        np.ndarray, optional: A 9x9x9 numpy array representing a solved sudoku puzzle probability field.
                                If no solution can be found, returns None.
    """
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
    
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
        
    # Collapse all cells with only one option until no more remain, or the puzzle is unsolvable.
    state, collapse_count = propagate_collapse(prob_field=prob_field, collapsed_cells=collapsed_cells)
    
    if state == 0:
        return None, recursions, failed_recursions, collapse_count
    if state == 2:
        return prob_field, recursions, failed_recursions, collapse_count
    
    # Sum the probability field along the value axis.
    resolution_map = prob_field.sum(axis=2)
        
    # TODO: Find a simpler way to do this.
    unresolved_indices = np.argwhere(resolution_map > 1)
    x, y = unresolved_indices[np.argmin(resolution_map > 1)]
    collapsed_cells[x][y] = 1
    
    # Recurse, passing a collapsed probability field and a copy of the collapsed cells.
    indexes = np.where(prob_field[x][y])[0]
    c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
    indexes = [x for _, x in sorted(zip(c_values, indexes), reverse=False)]
    
    for i in indexes:
        # Create a copy of the probability field and collapse the cell to the current option.
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        # Result, recursion_cunt, failed_recursions
        r, _rs, _frs, _c = ripple_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   # NOTE: Used for heuristic testing.
        collapse_count += _c + 1    # 

        # If a solution is found, return it.
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count
        
        # If no solution is found, increment the failed recursion count.
        failed_recursions += 1
        
    # If no option lead to a solution, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count


#
# Recursive Solver w/ Masking
# Attempt at using array masking to simplify cell selection.
@njit
def masked_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
    
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    collapse_count = 0      # Total number of cells collapsed.
        
    # Sum the probability field along the value axis.
    #   The value of each cell is equal to the number of remaining options for that cell.
    resolution_map = prob_field.sum(axis=2) # TODO: Replace resolution mapping with collapse map
    
    # Overwrite any previously collapsed cells with a high value (10).
    masked_array = mask_2darray(resolution_map, collapsed_cells)

    # Find the cell with the lowest number of options (that has not been previously collapsed).
    c = np.argmin(masked_array)
    x, y = c // 9, c % 9
    
    # Collapse all cells with only one option until no more remain.
    while masked_array[x, y] == 1:
        
        collapse_count += 1
        collapsed_cells[x, y] = 1
        collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x, y]))
        
        # Regenerate the masked array and min cell index.
        resolution_map = prob_field.sum(axis=2)
        masked_array = mask_2darray(resolution_map, collapsed_cells) 
        c = np.argmin(masked_array)
        x, y = c // 9, c % 9
        
    # If any cell has no options, the puzzle is in a unsolvable state.
    if resolution_map[x, y] == 0:
        return None, recursions, failed_recursions, collapse_count
    
    # If all cells have an option and there is one option per cell, the puzzle is solved.
    if resolution_map.sum() == 81: # TODO: Should this be reorganized? Reintroduce the while True?
        return prob_field, recursions, failed_recursions, collapse_count
    
    # The cell with the least options still has > 1 options, so we recurse for each option.
    collapsed_cells[x, y] = 1
    indexes = np.where(prob_field[x][y])[0]
    
    # Calculate the collapse value for each option.
    c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
    
    # Iterate over the options, sorted by lowest collapse value.    
    for i in [x for _, x in sorted(zip(c_values, indexes), reverse=False)]:
        
        # Result, recursion_count, failed_recursions
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        r, _rs, _frs, _c = masked_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   # NOTE: Used for heuristic testing.
        collapse_count += _c + 1    # 

        # If a solution is found, return it.
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count
                        
        # If no solution is found, increment the failed recursion count.
        failed_recursions += 1
            
    # If no option lead to a solution, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count


#
# Recursive Solver
# Essentially ripple solve, but without naive collapse propagation.
@njit
def recursive_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
        
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    collapse_count = 0      # Total number of cells collapsed.
    
    # Sum the probability field along the value axis.
    #   The value of each cell is equal to the number of remaining options for that cell.
    resolution_map = prob_field.sum(axis=2) # TODO: Replace resolution mapping with collapse map

    if not resolution_map.all():
        return None, recursions, failed_recursions, collapse_count
    
    if resolution_map.sum() == 81:
        return prob_field, recursions, failed_recursions, collapse_count
    
    mask_2darray_inplace(resolution_map, collapsed_cells)

    # Find the cell with the lowest number of options (that has not been previously collapsed).
    c = np.argmin(resolution_map)
    x, y = c // 9, c % 9
    collapsed_cells[x, y] = 1
    
    # NOTE: This performs collapse_value calculation even when unneeded, replace with collapse_mapping
    indexes = np.where(prob_field[x][y])[0]
    c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
    for _, i in sorted(zip(c_values, indexes), reverse=False):
        
        # Result, recursion_count, failed_recursions
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        r, _rs, _frs, _c = recursive_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count
        
    return None, recursions, failed_recursions, collapse_count


# @njit
def collapse_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None,
                   c_map: np.ndarray = None, v_map: np.ndarray = None):

    # Thoughts:
    # Collapse field:
    #   9x9x9 array, each cell is a list of 9 values.
    #   Each index is 0 if that value is impossible, C if it is not.
    #   C = Minimum number of competing cells for that value in a row/column/region.
    #   
    #   IF: C = 1/C
    #   C is now the probability of that value being correct.
    #   If C = 1, that value is the only option for that cell?
    #   If we always choose the maximum C, we will always choose the option that collapses the fewest cells.
    #   If 
    #   

    if collapsed_cells is None: # On the first call, generate the heuristic maps.
        collapsed_cells = np.zeros((9, 9), dtype=int)
        
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    
    # Sum the probability field along the value axis.
    #   The value of each cell is equal to the number of remaining options for that cell.
    state, collapse_count = propagate_collapse(prob_field=prob_field, collapsed_cells=collapsed_cells)
    
    if state == 0:
        return None, recursions, failed_recursions, collapse_count
    if state == 2:
        return prob_field, recursions, failed_recursions, collapse_count
    
    # print(prob_field)
    
    
    
    c_map, w_map = generate_heuristic_maps(prob_field, collapsed_cells)
    # valid_idx = np.where(prob_field >= 0)[0]
    # min_index = valid_idx[prob_field[valid_idx].argmin()]
    # inverse_mask_2darray_inplace(resolution_map, remaining_cells)
    # valid_idx = np.where(remaining_cells)[0]
    
    # remaining_cells = 1 - collapsed_cells
    # # print(remaining_cells)
    # while remaining_cells.any():
        # other_min_index = valid_idx[resolution_map[remaining_cells].argmin()]
        # print(resolution_map)
    
    collapse_probs = prob_field / c_map
    print(collapse_probs)
    max_i = np.argmax(collapse_probs)
    x, y, i = np.unravel_index(max_i, collapse_probs.shape)
    print(x, y, i)
    print(max_i)
    print(prob_field[x, y])
    print(c_map[x, y, i])
    print(collapse_probs[x, y, i])
    print(collapsed_cells)

    return prob_field, recursions, failed_recursions, collapse_count
        
        # min_index = np.argmin(resolution_map)
        
        # indexes = np.where(prob_field[x][y])[0]
        # x, y = min_index // 9, min_index % 9
        # if resolution_map[x, y] == 1:
        #     remaining_cells[x, y] = 0
        
        
        # collapse_count += 1
        
        
        

    # print(min_index)
    return prob_field, recursions, failed_recursions, collapse_count



if __name__ == '__main__':
        
    np.set_printoptions(linewidth=np.inf)

    PUZZLE_FILE = 'sudoku_puzzle.json'
# 
    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)
        puzzles.pop('ai_escargot', None)
        
    prob_field = generate_probability_field(puzzles['evil'])
    solution, recursions, failed_recursions, collapses = collapse_solve(prob_field)
    print('Solution:')
    print(prob_field_to_puzzle(solution))
    print(recursions, failed_recursions, collapses)
