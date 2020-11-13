##TSP Solver 
from typing import Sequence
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
sys.path.append("..")
from geneticalg.core.AbstractSolver import AbstractSolver
import pandas as pd


class TSPSolver(AbstractSolver):
    def __init__(
        self,
        problem_type=float,
        fitness_func=None,
        pop_cnt: int = 100,
        gene_size: int = 100,
        max_gen: int = 1000,
        mutation_ratio: float = 0.2,
        selection_ratio: float = 0.2,
        selection_type: str = "",
        mutation_type: str = "interchanging",
        crossover_type: str = "",
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        verbose: bool = False,
        cv=0,
        **kwargs
    ):
        """
        :param fitness_function: can either be a fitness function or
        a class implementing a fitness function + methods to override
        the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        :param variables_limits: limits for each variable [(x1_min, x1_max), (x2_min, x2_max), ...].
        If only one tuple is provided, then it is assumed the same for every variable
        :param problem_type: whether problem is of float or integer type
        """

        AbstractSolver.__init__(
            self,
            problem_type=problem_type,
            gene_size=gene_size,
            fitness_func=fitness_func,
            pop_cnt=pop_cnt,
            max_gen=max_gen,
            mutation_ratio=mutation_ratio,
            selection_ratio=selection_ratio,
            selection_type=selection_type,
            mutation_type=mutation_type,
            crossover_type=crossover_type,
            excluded_genes=excluded_genes,
            verbose=verbose,
            cv=cv,
            **kwargs
        )
        
    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).

        :return: a numpy array with a randomized initialized population
        """
        population = np.empty(shape=(self.pop_cnt, self.gene_size))
        for i in range(0,self.pop_cnt):
            a = []
            for j in range(0,self.gene_size):
                a.append((np.random.random(1)[0],j+2))
            a = np.array(a)
            dt = [('col1' , a.dtype) , ('col2' , a.dtype)]
            assert a.flags['C_CONTIGUOUS']
            b = a.ravel().view(dt)
            b.sort(order=['col1','col2'])
            for j in range(0,self.gene_size):
                population[i,j] = b[j][1]


        return population

INT_MAX = float('inf')

def tsp_fitness(chromosome , map):
    sum = map[0][np.int(chromosome[0])-1]
    N = len(chromosome)
    for i in range(1 , len(chromosome)):
        sum = sum+map[np.int(chromosome[i]-1)][np.int(chromosome[i-1]-1)]
    sum = sum +map[0][np.int(chromosome[N-1])-1]
    return 1/sum

def test_tsp(map):
    with open(map) as file:
        info = file.read()
    info = info.split("\n")
    info = [ x.split() for x in info ]
    scores = pd.DataFrame(info)
    scores = scores.to_numpy()
    scores = scores.astype(np.float)
    solver = TSPSolver(
        gene_size=len(scores)-1,
        fitness_func=lambda a : tsp_fitness(a , scores),
        pop_cnt=1100, # population size (number of individuals)
        max_gen=400, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="tournament",
        crossover_type="one_point",
        mutation_type="insert",
        verbose=True,
        cv=0
    )

    solver.solve()

## Test maps
test_tsp("map7.txt")







