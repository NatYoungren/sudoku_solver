# Project Title

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Usage ](#usage-)
  - [Optimization Progress ](#optimization-progress-)
    - [Numba Speed Testing](#numba-speed-testing)
    - [Sorting by Collapse Value Heuristic](#sorting-by-collapse-value-heuristic)
    - [Minimizing Failed Recursions](#minimizing-failed-recursions)

## About <a name = "about"></a>

Write about 1-2 paragraphs describing the purpose of your project.

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## Usage <a name = "usage"></a>

Add notes about how to use the system.

## Optimization Progress <a name = "testing"></a>


### Numba Speed Testing  
> **Note:** See documentation on the [*@njit* decorator](https://numba.pydata.org/numba-doc/latest/user/5minguide.html#what-is-nopython-mode).
```
 > Averages from 1000 iterations of ripple_solve (no heuristic sorting):

Non-njit:
  easy        : 0.0012183
  medium      : 0.0021823
  evil        : 0.0045218
  evil2       : 0.0199788
  blank       : 0.0221743
  sudokupy    : 0.8660402
 Average time : 0.1526860

Njit:
  easy        : 0.0000409
  medium      : 0.0000895
  evil        : 0.0002152
  evil2       : 0.0010506
  blank       : 0.0015018
  sudokupy    : 0.0480540
 Average time : 0.0084920  (5.56% of non-njit)
```
**Conclusion:** Njit is about ***~18x faster*** than non-njit.
           

### Sorting by Collapse Value Heuristic

> **Notes:**  
> - C value is an estimate of how many likely a cell is to contain a given option, by evaluating the minimum number of competitors out of that cell's row, column, and 3x3 region.  
> - eg. C=9 shows that the evaluated cell option could theoretically be in any competing cell of that row, column, or region. C=1 means that the evaluated cell option has been solved, as it has no competitors in *either* its row, column, or region.
> - For these tests, once a cell with the lowest number of options is selected, those options are sorted and recursed on by their C values.
> - Collapsing options with lower C values impacts fewer cells per recursion. 
> - The more cells affected by a collapse, the higher the chances of an invalid board state.  

```
 > Averages from 1000 iterations of ripple_solve (using njit):

                                        % of unsorted time/recusions:
       Unsorted:                                     v
  easy        : 0.0000409 (   1 recursions)    (100% ~ 100%)
  medium      : 0.0001352 (   3 recursions)    (100% ~ 100%)
  evil        : 0.0002346 (   9 recursions)    (100% ~ 100%)
  evil2       : 0.0010838 (  56 recursions)    (100% ~ 100%)
  blank       : 0.0014291 ( 182 recursions)    (100% ~ 100%)
  sudokupy    : 0.0457672 (3541 recursions)    (100% ~ 100%)
 Average time : 0.0081151                          (100%)

       Reverse C sorting (high->low):
  easy        : 0.0000433 (   1 recursions)    (105.87% ~ 100%)
  medium      : 0.0002304 (  11 recursions)    (170.41% ~ 366.67%)
  evil        : 0.0007213 (  33 recursions)    (307.46% ~ 366.67%)
  evil2       : 0.0014490 (  75 recursions)    (133.70% ~ 133.93%)
  blank       : 0.0017153 ( 179 recursions)    (120.03% ~ 98.35%)
  sudokupy    : 0.1120580 (7602 recursions)    (244.84 ~ 214.69%)
 Average time : 0.0193696                          (238.69%)   

       C sorting (low->high):
  easy        : 0.0000401 (   1 recursions)    (98.04%% ~ 100%)
  medium      : 0.0001241 (   5 recursions)    (91.79% ~ 166.67%)
  evil        : 0.0002044 (   7 recursions)    (87.13% ~ 77.78%)
  evil2       : 0.0005253 (  27 recursions)    (48.47% ~ 48.21%)
  blank       : 0.0006736 (  49 recursions)    (47.13% ~ 26.92%)
  sudokupy    : 0.0049007 ( 335 recursions)    (10.71% ~ 9.46%)
 Average time : 0.0010780                          (13.28%)
```

**Conclusions:**
- C sorting averages ***~7.5x faster*** than unsorted.
- Njit + C sorting averages ***135~141x faster*** than Non-njit + Unsorted.
> - Selecting low C values results in signicantly fewer recursions in uncertain board states.
> - Selecting high C values has the potential to reduce more overall uncertainty per recursion, but this is outweighed by the frequency of invalid board states.


### Minimizing Failed Recursions
**Notes**:
- These 
- Sudokupy can potentially be solved in 34 recursions.
- Viewing 
```
       Recursion success rate:
 Unsorted              (3541 total recursions - failed 3527) [99.60%].
 Reverse C-Map Sorting (7602 total recursions - failed 7588) [99.82%].
 C-Map Sorting         ( 335 total recursions - failed  321) [95.82%].
```



```
Total Collapses:
 easy        : 76
 medium      : 75
 evil        : 95
 evil2       : 202
 blank       : 78
 ai_escargot : 1652
 sudokupy    : 1843
 ```