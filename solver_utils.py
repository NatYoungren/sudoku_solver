# Nat Youngren
# September 25, 2023
#

from numba import njit
import numpy as np

#   Critical functions:

# collapse_probability_field:
#   This is the heart of all my wave-function-collapse inspired solvers.
#   If a cell is collapsed to a single value, all other cells in the row, column, and region are updated to remove that value.
#   Whenever a probability grid is modified (at least when collapsing cells to a known value), this function should be handling that change.
#   Modifications occur in-place, so make a copy beforehand if you want to preserve the original state (i.e. when recursing).

# propagate_collapse:
#   This function is used to naively resolve puzzles by collapsing all cells with only one option until no more remain.
#   It also returns a state code indicating whether the puzzle is solved, unsolved, or unsolvable.
#   Whenever a probability field is modified, this function can be called to propagate that change (provided that at least one cell reached a known state).
#   The collapsed_cells parameter is used to avoid collapsing a cell multiple times. Both collapsed_cells and prob_field are modified in-place.


# # # # # # #
# Solver Utilities
#

@njit
def collapse_probability_field(prob_field: np.ndarray, x: int, y: int, i: int):
    """ Collapses an x, y cell to a single given value-index (i).
        > This change would indicate that the cell is known to contain the value (i+1).
        
        Perpetuates that change by setting the value-index (i) to 0 for all other cells in the row, column, and region.
        Modifies prob_field in place.
        
    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        x (int): X coordinate of the cell to collapse.
        y (int): Y coordinate of the cell to collapse.
        i (int): Index of the value to collapse to. (0-8)
    """
    prob_field[x, :, i] = 0         # Set option i to 0 for all cells in the x column.
    prob_field[:, y, i] = 0         # Set option i to 0 for all cells in the y row.
    
    xx = x // 3                     # Set option i to 0 for all cells in the region.
    yy = y // 3                     # (xx, yy) is the top-left corner of the 3x3 region containing x, y.
    prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i] = 0
    
    prob_field[x, y, :] = 0         # Set all options for the x, y cell to 0.
    prob_field[x, y, i] = 1         # Overwrite option i for the x, y cell to 1.
    

@njit
def propagate_collapse(prob_field: np.ndarray, collapsed_cells: np.ndarray): # TODO: Make a version that ONLY iterates over affected cells?
    """ Collapses all cells with only one option until no more remain.
        Modifies prob_field and collapsed_cells in place.
        Returns a state code and the number of cells collapsed.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        collapsed_cells (np.ndarray): 9x9 grid tracking whether each cell has been collapsed.

    Returns:
        int, int: State code and number of cells collapsed.
                    > State codes:  0 = unsolvable board state
                                    1 = unsolved board state
                                    2 = solved board state
    """
    # TODO: Consider replacing collapse_count with a sum of collapsed_cells.
    collapse_count = 0  # Number of cells collapsed.
    prev_count = -1     # Used to detect when no more cells can be collapsed.
    
    while True:
        # Sum the probability field along the value axis.
        resolution_map = prob_field.sum(axis=2)
        
        # If any cell has no options, the puzzle is unsolvable.
        if not resolution_map.all():
            break

        # If there is one choice per cell, return the solved probability field.
        if resolution_map.sum() == 81:
            return 2, collapse_count

        # If the puzzle has been altered, look for new cells that can be collapsed.
        if prev_count != collapse_count: 
            prev_count = collapse_count
            
            # Iterate over all cells with only one option.
            resolvable_indices = np.argwhere(resolution_map == 1)
            for x, y in resolvable_indices:
                
                # Skip cells that have been previously collapsed.
                # FIXME: Rather than skipping these, we should avoid including them in resolved indices.
                if collapsed_cells[x, y]:
                    continue
                
                # Abort if the cell has no options.
                if not prob_field[x, y].sum():
                    break
                
                # Collapse the cell to the only option and propagate that change.                    
                collapse_probability_field(prob_field, x, y, np.argmax(prob_field[x][y]))
                collapse_count += 1

                # Mark the cell as collapsed.
                collapsed_cells[x][y] = 1
        else:
            return 1, collapse_count
        
    return 0, collapse_count


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

# @njit
# def mask_2d_3d(arr: np.ndarray, mask_arr: np.ndarray, maskval=10):
#     arr[mask_arr] = maskval