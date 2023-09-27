# Nat Youngren
# September 25, 2023
#

from numba import njit
import numpy as np

# # # # # # #
# Heuristic Utilities
#
# NOTE: These utilities are used to weight choices by their relationships with other cells.
#


@njit
def update_heuristic_probability_field(prob_field: np.ndarray):
    """ Generates a heuristic map for each cell in the probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        collapsed_cells (np.ndarray): 9x9 grid tracking whether each cell has been collapsed.

    Returns:
        np.ndarray: 9x9x9 heuristic grid, each heuristic value between 0 and 100.
                    The heuristic value of each value-index is calculated as follows:
                        > 100 - 10 * min(collapse_sums) + max(collapse_sums) 
                        
                          collapse_sums are the numbers of possible cells for one value-index in that cell's row/column/region.
                          collapse_sums range between 0 and 9, so the heuristic value ranges between 100 and 1.
                          Note that 100 sums are indicative of invalid board states.
                          
                        Any value-index which cannot be considered defaults to a heuristic value of 0.
                        

    """
    prob_field = np.where(prob_field != 0, np.uint8(1), np.uint8(0))
    row_sums, col_sums, region_sums = get_sums(prob_field)

    for x in range(9):
        for y in range(9):
            r = (x//3)*3+y//3
            for i in range(9):
                prob_field[x, y, i] *= 100 - (min(col_sums[x, i], row_sums[y, i], region_sums[r, i]) * 10 + max(col_sums[x, i], row_sums[y, i], region_sums[r, i]))
                
    return prob_field


@njit
def generate_unified_heuristic_map(prob_field: np.ndarray, collapsed_cells: np.ndarray):
    """ Generates a heuristic map for each cell in the probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        collapsed_cells (np.ndarray): 9x9 grid tracking whether each cell has been collapsed.

    Returns:
        np.ndarray: 9x9x9 heuristic grid, each heuristic value between 0 and 100.
                    The heuristic value of each value-index is calculated as follows:
                        > 10 * min(collapse_sums) + max(collapse_sums) 
                        
                          collapse_sums are the numbers of possible cells for one value-index in that cell's row/column/region.
                          collapse_sums range between 0 and 9, so the heuristic value ranges between 0 and 99.
                          Note that 0 sums are indicative of invalid board states.
                          
                        Any value-index which cannot be considered defaults to a heuristic value of 100.
    """
    # NOTE: This seems to save negligible/no time compared to generate_heuristic_maps.
    row_sums, col_sums, region_sums = get_sums(prob_field)
    heuristic_map = np.full((9, 9, 9), fill_value=100, dtype=np.uint8)
    for x in range(9):
        for y in range(9):
            if collapsed_cells[x, y]:
                continue
            r = (x//3)*3+y//3
            for i in range(9):
                if prob_field[x, y, i]:
                    heuristic_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[r, i]) * 10 + max(col_sums[x, i], row_sums[y, i], region_sums[r, i])
                
    return heuristic_map # TODO: Revisit these names, swap them?


@njit
def generate_heuristic_maps(prob_field: np.ndarray, collapsed_cells: np.ndarray):
    """ Generates heuristics for each cell in the probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.
        collapsed_cells (np.ndarray): 9x9 grid tracking whether each cell has been collapsed.
                                      Collapsed cells default to 10 in the heuristic maps.

    Returns:
        collapse_map (np.ndarray), weight_map (np.ndarray): 9x9x9 heuristic grids.
            > collapse_map: 9x9x9 grid tracking the minimum number of competing cells for that value in a row/column/region.
            > weight_map: 9x9x9 grid tracking the maximum number of competing cells for that value in a row/column/region. # TODO: Consider changing to sum of all.
    """
    row_sums, col_sums, region_sums = get_sums(prob_field)
    
    collapse_map = np.full((9, 9, 9), fill_value=10, dtype=np.uint8)
    weight_map = np.full((9, 9, 9), fill_value=10, dtype=np.uint8) # TODO: Consider changing fill to 1 or 0.
    
    for x in range(9):
        for y in range(9):
            if collapsed_cells[x, y]:
                continue
            r = (x//3)*3+y//3
            for i in range(9):
                if prob_field[x, y, i]:
                    # NOTE: Consider using sum?
                    weight_map[x, y, i] = max(col_sums[x, i], row_sums[y, i], region_sums[r, i])
                    collapse_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[r, i])
                
    return collapse_map, weight_map # TODO: Revisit these names, swap them?


@njit
def get_sums(prob_field: np.ndarray):
    """ Returns the row, column, and region sums for a given probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Returns row, column, and region summed along the non-value axes.
                                            Gives the sum of possibilities for each value in each row/column/region.
    """
    row_sums = prob_field.sum(axis=0)
    col_sums = prob_field.sum(axis=1)
    region_sums = np.zeros((9, 9), dtype=np.uint8)
    
    # TODO: Make this more efficient?
    for x in range(3):
        for y in range(3):
            # TODO: Use flatten + sum instead?
            region_sums[x*3+y] = prob_field[x*3:x*3+3, y*3:y*3+3].sum(axis=0).sum(axis=0)
    
    return row_sums, col_sums, region_sums


# Low collapse value means the choice is less likely to lead to a broken board state.
@njit
def collapse_value(prob_field: np.ndarray, x: int, y: int, i: int):
    xx = x // 3
    yy = y // 3
    return min(prob_field[xx*3:xx*3+3, yy*3:yy*3+3, i].sum(), prob_field[x, :, i].sum(), prob_field[:, y, i].sum())


# NOTE: All methods below are unused.

# Note that the value of x,y,i is counted once in the final sum:
#   row + col + (region - (row ∩ reg) - (col ∩ reg) = sum
#   cell + cell + (cell - cell - cell) = (cell) x 1
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

if __name__ == '__main__':
    import timeit
    iterations = 1000
    print(' > Precompile njit')
    generate_unified_heuristic_map(np.zeros((9, 9, 9), dtype=np.uint8), np.zeros((9, 9), dtype=np.uint8))
    
    t1 = timeit.timeit(lambda: generate_unified_heuristic_map(np.zeros((9, 9, 9), dtype=np.uint8), np.zeros((9, 9), dtype=np.uint8)), number=iterations)
    
    print(f'Time: {t1/iterations:0.7f}')