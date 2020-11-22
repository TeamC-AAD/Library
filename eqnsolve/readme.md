# Equation Solver

The equation solver solves a non-linear higher degree polynomial equations in order to get the most minimal root (if exists) or the closes real valued number of ```x```
which satisfied the given equation. The walkthrough has been provided below.

## Walkthrough

In order to solve the equation, first create a file and let us name it for now as ```eqn6.py```. Import numpy using the following command:

```python3
import numpy as np
```

Define the number of distinct variables in your equation. For now, let ```var=1```. Now, let us define the fitness as follows:

```python3
def eqnfit(chromosome):

    '''
    Equation 1: x^-2 + x^3 - x = 2
    '''

    eqn = chromosome[0]**-2 + chromosome[0]**3 - chromosome[0]
    val = 2
    return (1/(eqn-val))
```

In order to get the value of the minimal root (or nearest), define this function:

```python3
def value(chromosome):

    return chromosome[0]**-2 + chromosome[0]**3 - chromosome[0]
```
