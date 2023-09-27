# Nat Youngren
# August 1, 2023
#

import numpy as np
from numba import njit
import json

from setup_utils import get_region, get_options, generate_probability_field, prob_field_to_puzzle, validate_solution
from solver_utils import collapse_probability_field, propagate_collapse, mask_2darray, mask_2darray_inplace, inverse_mask_2darray_inplace
from heuristic_utils import generate_heuristic_maps, collapse_value, generate_unified_heuristic_map, update_heuristic_probability_field

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
    
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
    # collapse_count is initialized below.
    
    # Collapse all cells with only one option until no more remain, or the puzzle is unsolvable.
    state, collapse_count = propagate_collapse(prob_field=prob_field, collapsed_cells=collapsed_cells)
    
    # Return if the puzzle is solved or unsolvable.
    if state != 2:
        r = [None, prob_field][state]
        return r, recursions, failed_recursions, collapse_count
    
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
    
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
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
    
    # The cell with the least options still has > 1 options, identify those options.
    indexes = np.where(prob_field[x][y])[0]
    collapsed_cells[x, y] = 1
    
    # Calculate the collapse value for each option.
    c_values = [collapse_value(prob_field, x, y, i) for i in indexes]
    
    # Recurse over the options, sorted by lowest collapse value.    
    for i in [x for _, x in sorted(zip(c_values, indexes), reverse=False)]:
        
        # Pass copies of the probability field and collapsed cells when recursing.
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        
        # Result, recursion_count, failed_recursions, collapse_count
        r, _rs, _frs, _c = masked_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   # NOTE: Used for heuristic testing.
        collapse_count += _c + 1    # 

        # If a solution is found, return it.
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count
                        
        # If no solution is found, increment the failed recursion count.
        failed_recursions += 1
    
    # If any cell was fully explored without success, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count


#
# Fully Recursive Solver
# Essentially masked solve, but without any collapse propagation.
# Each cell collapse triggers a recursion, this is slow and excessive but robust.
@njit
def recursive_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):
    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
        
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
    collapse_count = 0      # Total number of cells collapsed.
    
    # Sum the probability field along the value axis (into a 9x9 array).
    #   The value of each cell is equal to the number of remaining options for that cell.
    resolution_map = prob_field.sum(axis=2)

    # Abort if any cells are unsolvable.
    if not resolution_map.all():
        return None, recursions, failed_recursions, collapse_count
    
    # Return solution if all cells are solved.
    if resolution_map.sum() == 81:
        return prob_field, recursions, failed_recursions, collapse_count
    
    # Overwrite any previously collapsed cells with a high value (10).
    mask_2darray_inplace(resolution_map, collapsed_cells)

    # Find the cell with the lowest number of options (that has not been previously collapsed).
    c = np.argmin(resolution_map)
    x, y = c // 9, c % 9
    collapsed_cells[x, y] = 1
    
    # Determine the remaining options for that cell.
    indexes = np.where(prob_field[x][y])[0]
    
    # If that cell has multiple options, sort them by collapse value.
    if len(indexes) < 1:
        indexes[:] = [x for _, x in sorted(zip([collapse_value(prob_field, x, y, i) for i in indexes], indexes), reverse=False)]
    
    # Recurse over each option.
    for i in indexes:
        
        # Pass copies of the probability field and collapsed cells when recursing.
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        
        # Result, recursion_count, failed_recursions, collapse_count
        r, _rs, _frs, _c = recursive_solve(pf, collapsed_cells.copy())
        
        recursions += _rs           # Update metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        
        # If a solution was found, return it.
        if r is not None:
            return r,  recursions, failed_recursions, collapse_count

        failed_recursions += 1
        
    # If any cell was fully explored without success, the puzzle is unsolvable.
    return None, recursions, failed_recursions, collapse_count


#
# 
@njit
def collapse_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):

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

    if collapsed_cells is None:
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
    
        
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
    collapse_count = 0      # Total number of cells collapsed.

    while True:
        # Collapse all cells with only one option until no more remain, or the puzzle is unsolvable.
        state, _c = propagate_collapse(prob_field=prob_field, collapsed_cells=collapsed_cells)
        collapse_count += _c
        
        # Return if the puzzle is solved or unsolvable.
        if state != 2:
            r = [None, prob_field][state]
            return r, recursions, failed_recursions, collapse_count

        # Generate 9x9x9 heuristic maps.
        #   C = Minimum number of competing cells for that option in a row/column/region.
        #   W = Maximum number of competing cells for that option in a row/column/region.
        #       Default value of non-option cells is 10.
        c_map, w_map = generate_heuristic_maps(prob_field, collapsed_cells)
        
        # Identify the minimum C value in the heuristic map.
        min_c = np.min(c_map)
        
        # Identify the indexes of all options with that minimum C value.
        min_c_indexes = np.argwhere(c_map == min_c)

        # Solve all trivial cases where C = 1, without recursion.
        if min_c == 1:
            for x, y, i in min_c_indexes:
                collapse_probability_field(prob_field, x, y, i)
                collapsed_cells[x, y] = 1
                collapse_count += 1
            continue
        
        # If there are multiple non-trivial options with the minimum C value, select the one with the lowest W value.
        if len(min_c_indexes) > 1:
            min_w_vals = np.array([w_map[x, y, i] for x, y, i in min_c_indexes])
            x, y, i = min_c_indexes[np.argmin(min_w_vals)]
        
        # If there is only one non-trivial cell with the minimum C value, select it.
        else:
            x, y, i = min_c_indexes[0]
            
        # Collapse the selected cell and recurse.
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        cc = collapsed_cells.copy()
        cc[x, y] = 1
        r, _rs, _frs, _c = collapse_solve(pf, cc)
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        
        # If the recursion found a solution, return it.
        if r is not None:
            return r, recursions, failed_recursions, collapse_count
        
        # If the recursion failed to return a solution, remove that option from the prob_field.
        prob_field[x, y, i] = 0
        failed_recursions += 1
        

@njit
def simpler_collapse_solve(prob_field: np.ndarray, collapsed_cells: np.ndarray = None):

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
        collapsed_cells = np.zeros((9, 9), dtype=np.bool_)
    # h_map = np.full((9, 9, 9), fill_value=100, dtype=np.uint8)
        
    # DEBUG: Track metrics.
    recursions = 1          # Total recursions (including first call)
    failed_recursions = 0   # TODO: Consider tracking to max depth instead.
    collapse_count = 0
    
    while True:
        # Sum the probability field along the value axis.
        #   The value of each cell is equal to the number of remaining options for that cell.
        state, _c = propagate_collapse(prob_field=prob_field, collapsed_cells=collapsed_cells)
        collapse_count += _c
        
        if state != 2:
            # Return if the puzzle is solved or unsolvable.
            r = [None, prob_field][state]
            return r, recursions, failed_recursions, collapse_count
    
        h_map = generate_unified_heuristic_map(prob_field, collapsed_cells)
        
        # Identify index of minimum value in heuristic map.
        index = np.argmin(h_map)
        
        # Convert flat index into x, y, i
        x = index // 81
        y = (index % 81) // 9
        i = index % 9
        
        if h_map[x, y, i] < 20: # If there are trivial indexes, solve them all without recursion.
            for x, y, i in np.argwhere(h_map < 20):
                collapse_probability_field(prob_field, x, y, i)
                
                collapsed_cells[x, y] = 1
                collapse_count += 1 # NOTE: Not tracking heuristic collapses.
            continue
        
        # Recurse
        pf = prob_field.copy()
        collapse_probability_field(pf, x, y, i)
        cc = collapsed_cells.copy()
        cc[x, y] = 1
        r, _rs, _frs, _c = simpler_collapse_solve(pf, cc)
        
        recursions += _rs           # Update the tracked metrics.
        failed_recursions += _frs   #
        collapse_count += _c + 1    #
        
        if r is not None:
            return r, recursions, failed_recursions, collapse_count
        
        failed_recursions += 1

@njit
def heuristic_solve(prob_field: np.ndarray, remaining_options: np.ndarray = None):
    # Remaining options is used to mask the probability field.
    # It is an inverted version of collapsed_cells with a value axis, making it 9x9x9.
    # TODO: Consider swapping back to a 9x9 array.
    if remaining_options is None:
        remaining_options = np.ones((9, 9, 9), dtype=np.bool_)
        
    # NOTE: These are purely performance metrics.
    recursions = 1          # Total recursions, including initial solver call. (min 1)
    failed_recursions = 0   # Total recursions that failed to find a solution. (min 0)
    collapse_count = 0      # Total number of cells collapsed.
    
    # While there are still cells with more than one option (NOTE: verifying that all cells have options occurs inside).
    while np.count_nonzero(prob_field) > 81:
        
        # TODO: Propagate collapse to remove single-option cells. (Inside update_h_p_f?)

        # Fill prob_field with heuristic values.
        #   H = 100 - ((C * 10) + W)
        #      Default value of non-option cells is 0.
        #   Where:
        #       C = Minimum number of competing cells for that option in a row/column/region.
        #       W = Maximum number of competing cells for that option in a row/column/region.
        prob_field = update_heuristic_probability_field(prob_field)
        
        # Mask the probability field to only the remaining cells.
        h_map = np.multiply(prob_field, remaining_options)
        
        # Identify index of maximum value in heuristic map.
        index = np.argmax(h_map)
        x = index // 81
        y = (index % 81) // 9
        i = index % 9
        
        # Solve all trivial cases where H > 80 (C = 1), without recursion.
        if h_map[x, y, i] > 80:
            for x, y, i in np.argwhere(h_map > 80):
                
                # If any cell is being collapsed twice in one pass, the puzzle is unsolvable.
                if not remaining_options[x, y, i]:
                    return None, recursions, failed_recursions, collapse_count
                
                collapse_probability_field(prob_field, x, y, i)
                remaining_options[x, y, :] = 0
                collapse_count += 1
            
            # If any cell has no options, the puzzle is unsolvable.
            if not prob_field.sum(axis=2).all():
                return None, recursions, failed_recursions, collapse_count
            
        # Collapse the cell option and recurse.
        else:
            pf = prob_field.copy()
            collapse_probability_field(pf, x, y, i)
            ro = remaining_options.copy()
            ro[x, y, :] = 1
            r, _rs, _frs, _c = heuristic_solve(pf, ro)
            
            recursions += _rs           # Update the tracked metrics.
            failed_recursions += _frs   #
            collapse_count += _c + 1    #
            
            # If a solution was found, return it.
            if r is not None:
                return r, recursions, failed_recursions, collapse_count
            
            # If the recursion failed to return a solution, remove that option from the prob_field.
            prob_field[x, y, i] = 0
            failed_recursions += 1
            
            # If the updated cell has no options, the puzzle is unsolvable.
            if not np.count_nonzero(prob_field[x, y, :]):
                return None, recursions, failed_recursions, collapse_count
    
    # If all cells have been collapsed, the puzzle is solved.
    return prob_field, recursions, failed_recursions, collapse_count
        
        
if __name__ == '__main__':
    import time
    np.set_printoptions(linewidth=np.inf)
    
    PUZZLE_FILE = 'sudoku_puzzle.json'
    
    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)
    
    # puzzle = np.array(puzzles['evil'])
    print('Precompiling...')
    collapse_solve(np.zeros((9, 9, 9), dtype=np.uint8))
    simpler_collapse_solve(np.zeros((9, 9, 9), dtype=np.uint8))
    
    for name, puzzle in puzzles.items():
        if name not in ['sudokupy']:
            continue
        
        puzzle = np.array(puzzle)
        
        print('Puzzle:', name)
        
        prob_field = generate_probability_field(puzzle)
        t1 = time.time()
        solution, recursions, failed_recursions, collapses = collapse_solve(prob_field.copy())
        t2 = time.time()
        _solution, _recursions, _failed_recursions, _collapses = simpler_collapse_solve(prob_field.copy())
        t3 = time.time()
        print(prob_field_to_puzzle(_solution))

        print('Solution:', (prob_field_to_puzzle(solution) == prob_field_to_puzzle(_solution)).all())
        print(f'collapse: {t2-t1:0.7f}', recursions, failed_recursions, collapses)
        print(f'simple_c: {t3-t2:0.7f}', _recursions, _failed_recursions, _collapses)

