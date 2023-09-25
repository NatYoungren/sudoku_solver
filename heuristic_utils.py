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
def generate_heuristic_maps(prob_field: np.ndarray): # TODO: Consider returning the 3 sets of sums. (row, col, region)
    """ Generates heuristics for each cell in the probability field.

    Args:
        prob_field (np.ndarray): 9x9x9 grid tracking the possibility of each cell containing each value.

    Returns:
        collapse_map (np.ndarray), value_map (np.ndarray): 9x9x9 heuristic grids.
            > collapse_map: 9x9x9 grid tracking the minimum number of competing cells for that value in a row/column/region.
            > weight_map: 9x9x9 grid tracking the number of cells affected any given collapse.
    """
    
    row_sums, col_sums, region_sums = get_sums(prob_field)
    
    # NOTE: Consider using prob_field.copy() multiplying the values.
    #       This would leave all impossible values as 0.
    collapse_map = np.zeros((9, 9, 9), dtype=np.uint8)
    weight_map = np.zeros((9, 9, 9), dtype=np.uint8)
    
    for x in range(9):
        for y in range(9):
            for i in range(9):
                # NOTE: Consider using sum?
                weight_map[x, y, i] = max(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
                collapse_map[x, y, i] = min(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
                
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
    region_sums = np.zeros((9, 9))
    
    # TODO: Make this more efficient?
    for x in range(3):
        for y in range(3):
            # TODO: Use flatten + sum instead?
            region_sums[x*3+y] = prob_field[x*3:x*3+3, y*3:y*3+3].sum(axis=0).sum(axis=0)
    
    return row_sums, col_sums, region_sums

# @njit
# def apply_collapse_scalar(prob_field: np.ndarray):
#     # Modifies probability field to contain probability values between 0 and 1
#     # Returns value field which contains the number of options
    
#     # Modifies the probability field in place.
#     row_sums = prob_field.sum(axis=0)
#     col_sums = prob_field.sum(axis=1)

#     region_sums = np.zeros((9, 9))
#     value_field = np.zeros((9, 9, 9))
    
#     # TODO: Make this more efficient?
#     for x in range(3):
#         for y in range(3):
#             region_sums[x*3+y] = prob_field[x*3:x*3+3, y*3:y*3+3].sum(axis=0).sum(axis=0)

#     for x in range(9):
#         for y in range(9):
#             for i in range(9):
#                 # Consider using sum?
#                 value_field[x, y, i] = max(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
#                 prob_field[x, y, i] /= min(col_sums[x, i], row_sums[y, i], region_sums[(x//3)*3+y//3, i])
                
#     return value_field


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

