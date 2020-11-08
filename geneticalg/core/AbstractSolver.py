''' Base solver class '''
import datetime
import logging
import math
import random
from abc import ABCMeta, abstractmethod
from typing import Sequence

import statistics

import numpy as np
import matplotlib.pyplot as plt

from geneticalg.helper import mutation
from geneticalg.helper import selection
from geneticalg.helper import crossover


# TODO: Check if ma == pa

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
        mutation_type: str = "",
        crossover_type: str = "",
        excluded_genes: Sequence = None,
        verbose: bool = False,
        **kwargs
    ):
        seed = np.random.randint(0, 10)
        np.random.seed(seed)

        self.max_gen = max_gen
        self.fitness_func = fitness_func
        self.pop_cnt = pop_cnt
        self.mutation_ratio = mutation_ratio
        self.selection_ratio = selection_ratio
        self.selection_type = selection_type
        self.gene_size = gene_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.verbose = verbose

        self.n_mutations = self.get_number_mutations()

        '''
        Base Tests
        '''
        if self.fitness_func is None:
            '''
            Generate Error function
            '''
        self.min_pop = math.floor(self.selection_ratio*self.pop_cnt)
        self.n_matings = math.floor((self.pop_cnt - self.min_pop)/2)
        self.mut_number = math.ceil((self.pop_cnt-1)*self.gene_size*self.mutation_ratio)
        self.excluded_genes = excluded_genes

        if self.min_pop < 2 or self.n_matings < 1 or self.mut_number < 0:
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
        
        self.kwargs = kwargs

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

        average_fitness = []
        max_fitness = []

        # Randomly initialize population
        population = self.initialize_population()
        # Compute fitness for current population
        fitness = self.calculate_fitness(population)

        # Order individuals by their fitness (desc)
        fitness, population = self.sort_by_fitness(fitness, population)

        gen_interval = max(round(self.max_gen / 15), 1)

        while True:
            generation += 1

            if generation % gen_interval == 0 and self.verbose:
                logging.info(f"Iter number: {generation}")
                logging.info(f"Best fitness: {fitness[0]}")

            curr_avg_fitness = statistics.mean(average_fitness)

            # Track average and max, fitness is sorted
            average_fitness.append(curr_avg_fitness)
            max_fitness.append(fitness[0])

            ma, pa = self.select_parents(fitness)

            ix = np.arange(0, self.pop_cnt - self.min_pop - 1, 2)

            xp = np.array(
                list(map(lambda _: self.get_crossover_points(), range(self.n_matings)))
            )

            # Generate the next population
            for i in range(xp.shape[0]):
                ma_ind = population[ma[i], :]
                pa_ind = population[pa[i], :]
                f_child, s_child = self.create_offspring(
                    ma_ind, pa_ind, xp[i]
                )
                population[-1 - ix[i], :] = f_child
                population[-1 - ix[i] - 1, :] = s_child

            # Mutate population
            population = self.mutate_population(population, self.n_mutation)
            # Compute fitness for current population
            fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))
            # Order fitness and population
            fitness, population = self.sort_by_fitness(fitness, population)
            
            if generation >= self.max_gen:
                break

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
        """

        ma,pa = None

        ma = selection.selection_strats[self.selection_type](fitness, n_matings)
        pa = selection.selection_strats[self.selection_type](fitness, n_matings)

        return ma,pa
    
    def get_number_mutations(self):
        return math.ceil((self.pop_size - 1) * self.n_genes * self.mutation_rate)

    @staticmethod
    @abstractmethod
    def create_offspring(self,first_parent, sec_parent):
        """docstring
        """
        if child_no != 1 or child_no != 2:
            '''
            Error
            '''
            pass
        beta = np.random.rand(1)[0]
        child1 , child2 = None
        child1, child2 = crossover.crossover_strats[self.crossover_type](first_parent, sec_parent , beta)

        return child1,child2

    def mutate_population(self, population, n_mutation):
        """docstring
        """
        chosen_indivs = random.sample(population, n_mutation)

        for indiv in chosen_indivs:
            pass

    @abstractmethod
    def initialize_population(self):
        """Sets up population array
        with randomized individual states

        returns:
            numpy array that models initial population
        """
        pass