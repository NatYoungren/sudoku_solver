# Nat Youngren
# September 25, 2023
#

import numpy as np
import random

# # # # # # #
# Setup Utilities
#

def generate_probability_field(puzzle: np.ndarray, dtype=np.uint8):
    """ Generates a probability field for a given puzzle.


    Args:
        puzzle (np.ndarray): A 9x9 numpy array representing a sudoku puzzle.
                            Filled cells contain their 1-9 value, empty cells are represented as 0.
        dtype (np.dtype): Data type of the output probability field. (Default: np.uint8

    Returns:
        np.ndarry: A 9x9x9 numpy array, the list at each cell represents the viable options for that cell (index = value-1).
                    Remaining options are represented as 1, removed options are represented as 0.
    """
    puzzle = np.array(puzzle)
    prob_field = np.zeros((9, 9, 9), dtype=dtype)
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


def prob_field_to_puzzle(prob_field: np.ndarray, dtype=np.uint8):
    """ Converts a 9x9x9 probability field into a 9x9 puzzle.

    Args:
        prob_field (np.ndarray): 9x9x9 probability field.
        dtype (np.dtype): Data type of the output puzzle. (Default: np.uint8

    Returns:
        np.array, optional: 9x9 numpy array holding cell values. Returns None if the prob_field was None.
                            0 represents an unsolved cell.
                            -1 represents an unsolvable cell.
    """
    if prob_field is None:
        return None
    
    out_puzzle = np.zeros((9, 9), dtype=dtype)
    for x, _ in enumerate(prob_field):
        for y, _ in enumerate(prob_field.T):
            s = np.count_nonzero(prob_field[x][y])
            if s == 1:
                out_puzzle[x][y] = np.argmax(prob_field[x][y]) + 1
            elif not s:
                out_puzzle[x][y] = -1
    return out_puzzle


def generate_puzzle(seed: int = None, difficulty=0.5, simple_and_solvable=True):
    # TODO: Complete this function.
    # TODO: Implement difficulty.
    if seed is None:
        seed = random.randint(0, np.iinfo(np.uint64).max)
        
    if simple_and_solvable:
        positions = list(range(9))
        random.seed(seed)
        random.shuffle(positions)
        return np.array([[(i + 1) if i == positions[j] else 0 for i in range(9)] for j in range(9)], dtype=np.uint8)
    else:
        pass # TODO: Implement more convoluted generator


def validate_solution(puzzle: np.ndarray, solution: np.array):
    out_text = []
    if solution is None:
        out_text.append('No solution found.')
        return False, out_text
    
    if solution.shape != puzzle.shape:
        out_text.append('Solution shape does not match puzzle shape.')
        return False, out_text
    
    valid = True
    for x, _ in enumerate(puzzle):
        for y, _ in enumerate(puzzle.T):
            if puzzle[x][y] == 0:
                continue
            if puzzle[x, y] != solution[x, y]:
                out_text.append(f'Solution does not match puzzle: {x}, {y}')
                valid = False
            
    valid_set = set([i for i in range(1, 10)])
    for x, row in enumerate(solution):
        if valid_set.difference(set(row)):
            out_text.append(f'Invalid row: {x}')
            valid = False
        
    for y, col in enumerate(solution.T):
        if valid_set.difference(set(col)):
            out_text.append(f'Invalid column: {y}')
            valid = False
    
    for x in range(1, 10, 3):
        for y in range(1, 10, 3):
            region = get_region(solution, x, y)
            if valid_set.difference(set(region.flatten())):
                out_text.append(f'Invalid region: {x}, {y}')
                valid = False
                
    return valid, out_text


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
    return puzzle[x*3:x*3+3, y*3:y*3+3].flatten()


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
