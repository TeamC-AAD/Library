''' Base solver class '''
import datetime
import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from geneticalg.helper import mutation
from geneticalg.helper import selection
from geneticalg.helper import crossover



class AbstractSolver:
    def __init__(
        self,
        gene_size: int=100,
        fitness_func=None,
        pop_cnt: int=100,
        max_gen: int=1000,
        mutation_ratio: float=0.2,
        selection_ratio: float=0.2,
        selection_type: str="",
        crossover_points: int=1,
    ):
        seed = np.random.randint(0,10)
        np.random.seed(seed)

        self.max_gen = max_gen
        self.fitness_func = fitness_func
        self.pop_cnt = pop_cnt
        self.mutation_ratio = mutation_ratio
        self.selection_ratio = selection_ratio
        self.selection_type = selection_type
        self.crossover_points = crossover_points
        self.gene_size = gene_size

        '''
        Base Tests
        '''
        if fitness_func is None:
            '''
            Generate Error function
            '''
        min_pop = math.floor(selection_ratio*pop_cnt)
        min_parents = math.floor((pop_cnt - min_pop)/2)
        mut_number = math.ceil((pop_cnt-1)*n_genes*mutation_ratio)

        if min_pop < 2 or min_parents < 1 or mut_number < 0:
             




    def solve(self):
        """Runs the genetic algorithm for the number of iterations
        and optimizes for the given problem.
        """

        average_fitness = np.ndarray(shape=(1, 0))
        max_fitness = np.ndarray(shape=(1, 0))