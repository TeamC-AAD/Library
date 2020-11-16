'''
Equation Solver
'''

##TSP Solver 
from typing import Sequence
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
sys.path.append("..")
from geneticalg.core.ContinuousSolver import ContinuousGenAlgSolver
from eqnsolve.eqn6 import *
import pandas as pd


def solveqn():
    solver = ContinuousGenAlgSolver(
        gene_size=var,
        fitness_func=lambda chromosome: eqnfit(chromosome),
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
        print(curr_data ,"    ", value(curr_data["best_ind"]))
    return

solveqn()