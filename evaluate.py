# Nat Youngren
# September 25, 2023
#

import numpy as np
import time
import timeit
import json

from setup_utils import generate_probability_field, prob_field_to_puzzle, validate_solution
from sudoku_solver import collapse_solve, masked_solve, recursive_solve, ripple_solve, simpler_collapse_solve

np.set_printoptions(linewidth=np.inf)




# # # # # # #
# Evaluation
#

def evaluate(puzzles, solvers, iterations=10, verbose_loop: bool = True, verbose_end: bool = True):
    """ Evaluates a set of solvers against a set of puzzles.
    Args:
        puzzles (dict): Dict of puzzle name -> 9x9 np.ndarray.
        solvers (list(func)): Solver functions which take and returns a 9x9x9 np.ndarray probability field.
        iterations (int, optional): Number of iterations performed during timing. Defaults to 10.
        verbose_loop (bool, optional): If True, print results of each puzzle during evaluation. Defaults to True.
        verbose_end (bool, optional): If True, print overall results after all puzzles are evaluated. Defaults to True.
    """
    stored_data = {}
    for solver in solvers:
        times = {}
        recursions = {}
        failed_recursions = {}
        collapses = {}
        was_correct = {}
        
        for name, puzzle in puzzles.items():
            if verbose_loop:
                print('\n Solving', name, 'with', solver.__name__, '\n')

            # Generate solver input
            puzzle = np.array(puzzle)
            prob_field = generate_probability_field(puzzle)

            # Solve once manually for solution and metrics
            start_time = time.time()
            solution, recursions[name], failed_recursions[name], collapses[name] = solver(prob_field.copy())
            end_time = time.time()
            
            # Solve repeatedly for average time
            t = timeit.timeit(lambda: solver(prob_field.copy()), number=iterations)
            times[name] = t / iterations
            
            # Validate solution
            solved_puzzle = prob_field_to_puzzle(solution)
            was_correct[name] = validate_solution(puzzle, solved_puzzle)
            
            # Print puzzle results
            if verbose_loop:
                solved_puzzle = solved_puzzle
                unsolved_nodes = np.count_nonzero(solved_puzzle == 0)
                print(solved_puzzle)
                print(f'Average time: {times[name]:0.7f} seconds.')
                print(f'Finished in {end_time - start_time:0.7f} seconds with {unsolved_nodes} unsolved nodes.')
                if not was_correct[name][0]:
                    print('Solution was invalid:')
                    for line in was_correct[name][1]:
                        print('>', line)
                        
        stored_data[solver.__name__] = (times, recursions, failed_recursions, collapses, was_correct)
    
    # Print overall results
    if verbose_end:
        
        # Iterate through all evaluated solvers
        for solver_name, data in stored_data.items():
            print(f'\n\n {iterations} iterations of {solver_name}:\n')
            times, recursions, failed_recursions, collapses, was_correct = data
            
            # Iterate through stored data for each puzzle
            for name, t in times.items():
                r = recursions.get(name, np.nan)
                failed_r = failed_recursions.get(name, np.nan)
                c_count = collapses.get(name, np.nan)
                print(f' {name:<12}: {t:0.7f} ({r:>4} total recursions - failed {failed_r:>4}) [{(100/r)*failed_r:0.2f}%] ({c_count} total collapses)')
                
                # Print any validation errors
                if not was_correct[name][0]:
                    print('  Solution was invalid:')
                    for line in was_correct[name][1]:
                        print('   >', line)
                        
            print(f'Average time : {sum(times.values()) / len(times):.7f}')


if __name__ == '__main__':
    PRECOMPILE = True

    PUZZLE_FILE = 'sudoku_puzzle.json'
    
    VERBOSE_LOOP = False
    VERBOSE_END = True
    ITERATIONS = 1000
    
    TEST_FUNCS = [ripple_solve, recursive_solve, masked_solve, collapse_solve, simpler_collapse_solve, heuristic_solve]

    with open(PUZZLE_FILE) as f:
        puzzles = json.load(f)
        # puzzles.pop('ai_escargot', None)
    
    if PREP_TIMEIT:
        # Presolve impossible puzzle:
        print(' > Prepping timeit')
        for f in TEST_FUNCS:
            print(f' >> {f.__name__}')
            timeit.timeit(lambda: f(np.zeros((9, 9, 9), dtype=np.uint8)), number=1)

    evaluate(puzzles, TEST_FUNCS, iterations=ITERATIONS, verbose_loop=VERBOSE_LOOP, verbose_end=VERBOSE_END)
