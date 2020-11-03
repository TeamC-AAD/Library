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
        gene_size: int = 100,
        fitness_func=None,
        pop_cnt: int = 100,
        max_gen: int = 1000,
        mutation_ratio: float = 0.2,
        selection_ratio: float = 0.2,
        selection_type: str = "",
        crossover_points: int = 1,
        excluded_genes: Sequence = None,
    ):
        seed = np.random.randint(0, 10)
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
        if self.fitness_func is None:
            '''
            Generate Error function
            '''
        self.min_pop = math.floor(self.selection_ratio*self.pop_cnt)
        self.min_parents = math.floor((self.pop_cnt - self.min_pop)/2)
        self.mut_number = math.ceil((self.pop_cnt-1)*self.gene_size*self.mutation_ratio)
        self.excluded_genes = excluded_genes

        if self.min_pop < 2 or self.min_parents < 1 or self.mut_number < 0:
            '''
            Generate Error
            '''
            pass

        self.allowed_selection_strat = ["rowlette_wheel", "tournament",
                                   "linear_transform", "stochastic"]

        if self.selection_type not in self.allowed_selection_strat:
            '''
            Generate Error
            '''
            pass

        if excluded_genes is not None:
            self.excluded_genes = np.array(self.excluded_genes)
            self.all_genes = np.arange(0,self.gene_size)
            self.allowed_genes = []
            for i in self.all_genes:
                if i in self.excluded_genes:
                    continue
                else:
                    self.allowed_genes.append(i)

        if excluded_genes is None:
            '''
            Raise error
            '''
            pass

    def calculate_fitness(self, population):
        '''Calculates fitness of the population
        Args:
            population (list): Array of state for each individual
        Returns:
            fitness of current population
        '''

        # Apply the provided fitness function
        # to each of the individual
        result = list(map(self.fitness_func, population))

        # Convert the results to numpy array and return
        return np.array(result)


    def solve(self):
        """Runs the genetic algorithm for the number of iterations
        and optimizes for the given problem.
        """
        generation = 0
        start_time = datetime.datetime.now()

        average_fitness = np.ndarray(shape=(1, 0))
        max_fitness = np.ndarray(shape=(1, 0))

        # Randomly initialize population
        population = self.initialize_population()
        # Compute fitness for current population
        fitness = self.calculate_fitness(population)

        fitness, population = self.sort_by_fitness(fitness, population)

        gen_interval = max(round(self.max_gen / 10), 1)

        # TODO
        while True:
            pass

    @staticmethod
    def sort_by_fitness(fitness, population):
        """Sorts population according to fitness
        Args:
            fitness(list): fitness of current population
            population(list): state of current population
        Returns:
            Sorted population array according to fitness
        """
        # Sort
        sorted_ind = np.argsort(fitness)
        # Descending order
        sorted_ind = sorted_ind[::-1]

        # Pick relevant individuals and their fitnesses
        population = population[sorted_ind, :]
        fitness = fitness[sorted_ind]

        return fitness, population

    def select_parents(self, fitness):
        """
        Selects two parents ma and pa 
        Args:
            fitness(list): fitness of current population
            self: Object instance
        Returns:
            Two parents ma and pa
        Allowed Strategies:
            allowed_selection_strat = 
        ["rowlette_wheel", "tournament","linear_transform", "stochastic"]
        """

        ma,pa = None
        
        if self.selection_type == "tournament":
            '''
            Call tournament
            '''
            ma = tournament_selection(fitness , np.len(fitness))

        
        if self.selection_type == "rowlette_wheel":
            '''
            Call Rowlette Wheel
            '''
            pass

        if self.selection_type == "linear_transform":
            '''
            Call Linear transform
            '''
            pass
        
        if self.selection_type == "stochastic":
            '''
            Call Stochastic
            '''
            pass

        return ma,pa
        
        

        
        

    @abstractmethod
    def initialize_population(self):
        """Sets up population array
        with randomized individual states

        returns:
            numpy array that models initial population
        """
        pass