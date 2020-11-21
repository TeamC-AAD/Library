'''
Equation Solver
'''

##TSP Solver 
from typing import Sequence
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import math
sys.path.append("..")
from geneticalg.core.ContinuousSolver import ContinuousGenAlgSolver

# 2*x^5 + 3*x^3 + 10x = 14

powers = np.array([5, 3, 1])
weights = np.array([2, 3, 10])

def solveqn(powers=powers, weights=weights, var=1):

    solver = ContinuousGenAlgSolver(
        gene_size=var,
        fitness_func=lambda chromosome: eqnfit(chromosome, powers, weights),
        pop_cnt=4000, # population size (number of individuals)
        max_gen=200, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="linear_rank",
        crossover_type="uniform",
        mutation_type="insert",
        verbose=True
    )
    for curr_data in solver.solve(): 
        print(curr_data ,"    ", value(curr_data["best_ind"], powers, weights))
        yield curr_data
    return

def value(chromosome, powers, weights):
    return np.dot(weights, chromosome ** powers)

def eqnfit(chromosome, powers, weights):
    output = np.dot(weights, chromosome ** powers)
    error = max(1e-30, math.fabs(output))
    return 1/error

# list(solveqn())

# 2x^5 + 3x^3 + 10x