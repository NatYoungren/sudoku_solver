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
def generate_unified_heuristic_map(prob_field: np.ndarray, collapsed_cells: np.ndarray):
    # NOTE: This seems to save negligible/no time compared to generate_heuristic_maps.
    row_sums, col_sums, region_sums = get_sums(prob_field)
    
    # Max collapse value = 9
    # Max weight value = 9
    # Heuristic value = collapse value * 10 + weight value
    # Max heuristic value = 99
    # Default heuristic value = 100
    heuristic_map = np.full((9, 9, 9), fill_value=100, dtype=np.uint8)
    # heuristic_map.fill(100)
    for x in range(9):
        for y in range(9):
            if collapsed_cells[x, y]:
                continue
            r = (x//3)*3+y//3
            for i in range(9):
                if prob_field[x, y, i]:
                    heuristic_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[r, i]) * 10 + max(col_sums[x, i], row_sums[y, i], region_sums[r, i])
                
    return heuristic_map # TODO: Revisit these names, swap them?


@njit # NOTE: ATTEMPT TO SPEED UP UNIFIED HEURISTIC MAP
def generate_heuristic_map(prob_field: np.ndarray, collapsed_cells: np.ndarray):
    
    mask_arr = np.logical_or((1 - prob_field.flatten()), collapsed_cells.repeat(9)).reshape(9, 9, 9)
    
    sums_map = np.full((9, 9, 9, 3), fill_value=100, dtype=np.uint8)
    # sums_map = mask_arr.repeat(3).reshape((9, 9, 9, 3))
    # sums_map = np.empty((9, 9, 9, 3), dtype=np.uint8)
    row_sums, col_sums, region_sums = get_sums(prob_field) # NOTE: How few possibilities does it take to make this not worth doing?

    for k in range(9):
        sums_map[:, k, :, 0] = np.multiply(row_sums[k, :], mask_arr[:, k, :])
        sums_map[k, :, :, 1] = np.multiply(col_sums[k, :], mask_arr[k, :, :])
        sums_map[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :, 2] *= np.multiply(region_sums[k, :], mask_arr[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :,])
    
    print(sums_map)
    return sums_map

    heuristic_map = np.multiply((1 - mask_arr), np.add(np.multiply(sums_map[:, 0], 10), sums_map[:, 2]))
    
    # print(mask_arr.shape, heuristic_map.shape, sums_map.shape)
    
    # np.multiply((1 - mask_arr), (np_min_a0_3d(np.min, 0, sums_map) * 10 + np_apply_along_axis(np.max, 0, sums_map)), out=heuristic_map, casting='unsafe')
    heuristic_map = np.add(heuristic_map, mask_arr * 100)
    

    
    # NOTE: Slower than get_sums
    # sums_map = np.empty((9, 9, 9, 3), dtype=np.uint8)
    # for k in range(9): # TODO: Swap
    #     sums_map[:, k, :, 0] = prob_field[:, k, :].sum(axis=0)
    #     sums_map[k, :, :, 1] = prob_field[k, :, :].sum(axis=0)
    #     sums_map[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :, 2] = prob_field[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :].sum(axis=0).sum(axis=0)
    
        
    # heuristic_map = np.full((9, 9, 9), fill_value=100, dtype=np.uint8)
    # # heuristic_map.fill(100)
    # for x in range(9):
    #     for y in range(9):
    #         if collapsed_cells[x, y]:
    #             continue
    #         # r = (x//3)*3+y//3
    #         for i in range(9):
    #             if prob_field[x, y, i]:
    #                 heuristic_map[x, y, i] = min(sums_map[x, y, i, :]) * 10 + max(sums_map[x, y, i, :])
                
    # return heuristic_map # TODO: Revisit these names, swap them?

    
    # Make a mask which == True for all the indexes we do not want to select in the heuristic map
    #   These are all indexes which are either not marked as possible in prob field or are in a collapsed cells.
    
    print(mask_arr)
    print(mask_arr.shape)
    
    # print(np_min_a0_4d(sums_map))
    # print(np_max_a0_4d(sums_map))
    # print(np_min_a0_4d(sums_map) * 10 + np_max_a0_4d(sums_map))
    # print(mask_arr.dtype)
    
    # # print(sums_map.shape)
    # sums_map = sums_map.reshape((-1, 3)) # Test w/ flatten
    # # print(sums_map.shape)
    # for i in range(729):
    #     sums_map[i,:] = np.sort(sums_map[i, :])# = np.sort(sums_map[i, :])
        
    # print('sorted')
    # print(sums_map.shape)
    # print(mask_arr.shape)
    
    return heuristic_map

# @njit
# def compute_heuristic(arr):
    

@njit
def np_min_a0_4d(arr): # Get min along axis 0 of a 4d array. # NOTE: Njit workaround
    s = arr.shape
    out_arr = np.empty(s[1:], dtype=arr.dtype)
    
    for i1 in range(s[1]):
        for i2 in range(s[2]):
            for i3 in range(s[3]):
                out_arr[i1, i2, i3] = np.min(arr[:, i1, i2, i3])
            
    return out_arr

@njit
def np_max_a0_4d(arr): # Get max along axis 0 of a 4d array. # NOTE: Njit workaround
    s = arr.shape
    out_arr = np.empty(s[1:], dtype=arr.dtype)
    
    for i1 in range(s[1]):
        for i2 in range(s[2]):
            for i3 in range(s[3]):
                out_arr[i1, i2, i3] = np.min(arr[:, i1, i2, i3])
            
    return out_arr   

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

# @njit
# def get_sums2(prob_field: np.ndarray):
#     """ Returns the row, column, and region sums for each index in for a given probability field as a single array.

#     Args:
#         prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.

#     Returns:
#         np.ndarray: Returns a 9x9x9x3 array of summed values. [:, :, :, 0] is row sums, [:, :, :, 1] is column sums, [:, :, :, 2] is region sums.
#     """
#     sums_map = np.empty((9, 9, 9, 3), dtype=np.uint8)
#     for k in range(9): # TODO: Swap
#         sums_map[:, k, :, 0] = prob_field[:, k, :].sum(axis=0)
#         sums_map[k, :, :, 1] = prob_field[k, :, :].sum(axis=0)
#         sums_map[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :, 2] = prob_field[(k//3)*3:(k//3)*3+3, (k%3)*3:(k%3)*3+3, :].sum(axis=0).sum(axis=0)
#     return sums_map

@njit # TODO: Currently unused, remove?
def collapse_heuristic_map(h_map: np.ndarray, x: int, y: int, i: int, fillval=10): # TODO: Hardcode fillval?
    """ Removes a given value-index (i) from the heuristic map for all affected cells.
        Removes all value-indices from the heuristic map for the affected cell.
        
        Perpetuates that change by setting the value-index (i) to fillval for all other cells in the row, column, and region.
        Heuristic map in-place.
        
    Args:
        h_map (np.ndarray): 9x9x9 grid tracking the cell heuristics.
        x (int): X coordinate of the cell to collapse.
        y (int): Y coordinate of the cell to collapse.
        i (int): Index of the value to collapse to. (0-8)
    """
    h_map[x, :, i] = fillval         # Set option i to 0 for all cells in the x column.
    h_map[:, y, i] = fillval         # Set option i to 0 for all cells in the y row.
    
    xx = x // 3                     # Set option i to 0 for all cells in the region.
    yy = y // 3                     # (xx, yy) is the top-left corner of the 3x3 region containing x, y.
    h_map[xx*3:xx*3+3, yy*3:yy*3+3, i] = fillval
    
    h_map[x, y, :] = fillval         # Set all options for the x, y cell to 0.


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
    import time
    import json
    from setup_utils import generate_probability_field
    iterations = 1000
    PUZZLE_FILE = 'sudoku_puzzle.json'
    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)
    
    
    
    
    puzzle = np.array(puzzles['medium'])
    # print(puzzle)
    prob_field = generate_probability_field(puzzle)
    collapsed_cells = np.zeros((9, 9), dtype=np.uint8)
    collapsed_cells[0, 0] = 1
    collapsed_cells[5, 4] = 1
    
    # # Unified Heuristic Map
    # print(' > Precompiling njit...')
    # h_map1 = generate_unified_heuristic_map(prob_field, collapsed_cells)

    # print(' > Running w/ timeit...')
    # t1 = timeit.timeit(lambda: generate_unified_heuristic_map(prob_field, collapsed_cells), number=iterations)
    # print(f'Time: {t1/iterations:0.7f}')
    
    
    # New Heuristic Map
    print(' > Precompiling njit...')
    h_map2 = generate_heuristic_map(prob_field=prob_field, collapsed_cells=collapsed_cells)

    print(' > Running w/ timeit...')
    t2 = timeit.timeit(lambda: generate_heuristic_map(prob_field=prob_field, collapsed_cells=collapsed_cells), number=iterations)
    print(f'Time: {t2/iterations:0.7f}')
    
    
    # print(h_map1)
    # print(h_map2)
    
    # print(h_map1 == h_map2)
    # print((h_map1 == h_map2).all())
