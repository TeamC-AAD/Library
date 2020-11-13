''' Base solver class '''
# import datetime
import logging
import math
import random
from abc import abstractmethod
from typing import Sequence

import numpy as np

from geneticalg.helper import mutation
from geneticalg.helper import selection
from geneticalg.helper import crossover


# TODO: Check if ma == pa

class AbstractSolver:
    def __init__(
        self,
        problem_type: float,
        gene_size: int = 100,
        fitness_func=None,
        pop_cnt: int = 100,
        max_gen: int = 1000,
        mutation_ratio: float = 0.2,
        selection_ratio: float = 0.2,
        selection_type: str = "",
        mutation_type: str = "insert",
        crossover_type: str = "one_point",
        excluded_genes: Sequence = None,
        verbose: bool = False,
        cv=1,
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
        self.problem_type = float
        self.n_mutations = self.get_number_mutations()
        self.cv = cv

        '''
        Base Tests
        '''
        if self.fitness_func is None:
            '''
            Generate Error function
            '''
        self.min_pop = math.floor(self.selection_ratio*self.pop_cnt)
        self.n_matings = math.floor((self.pop_cnt - self.min_pop)/2)
        self.mut_number = math.ceil((self.pop_cnt-1) * self.gene_size *
                                    self.mutation_ratio)
        self.excluded_genes = excluded_genes

        if self.min_pop < 2 or self.n_matings < 1 or self.mut_number < 0:
            '''
            Generate Error
            '''
            pass

        if excluded_genes is not None:
            self.excluded_genes = np.array(self.excluded_genes)
            self.all_genes = np.arange(0, self.gene_size)
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
        # start_time = datetime.datetime.now()

        average_fitness = []
        max_fitness = []

        # Randomly initialize population
        population = self.initialize_population()
        # Compute fitness for current population
        fitness = self.calculate_fitness(population)

        # Order individuals by their fitness (desc)
        fitness, population = self.sort_by_fitness(fitness, population)

        gen_interval = max(round(self.max_gen / 15), 1)
        # gen_interval = 1
        while True:
            generation += 1

            if generation % gen_interval == 0 and self.verbose:
                logging.info(f"#Iter: {generation}")
                logging.info(f"#Best_fit: {fitness[0]}")

                print(f"Iter number: {generation}")
                print(f"Best fitness: {1/fitness[0]}")
                print(f"best individual: {population[0]}")

            # curr_avg_fitness = np.mean(np.array(average_fitness))

            # # Track average and max, fitness is sorted
            # average_fitness.append(curr_avg_fitness)
            # max_fitness.append(fitness[0])

            average_fitness = np.append(average_fitness, fitness.mean())
            max_fitness = np.append(max_fitness, fitness[0])

            ma, pa = self.select_parents(fitness)

            ix = np.arange(0, self.pop_cnt - self.min_pop - 1, 2)

            # Generate the next population
            for i in range(self.n_matings):
                ma_ind = population[ma[i], :]
                pa_ind = population[pa[i], :]
                f_child, s_child = self.create_offspring(
                    ma_ind, pa_ind
                )
                population[-1 - ix[i], :] = f_child
                population[-1 - ix[i] - 1, :] = s_child

            # Mutate population
            population = self.mutate_population(population, self.n_mutations)
            # Compute fitness for current population
            curr_fitness = self.calculate_fitness(population[1:, :])
            fitness = np.hstack((fitness[0], curr_fitness))
            # Order fitness and population
            fitness, population = self.sort_by_fitness(fitness, population)

            if generation >= self.max_gen:
                break

        self.generations_ = generation
        self.best_individual_ = population[0, :]
        self.best_fitness_ = fitness[0]
        self.population_ = population
        self.fitness_ = fitness

        print(f"Best individual: {self.best_individual_}")
        print(f"Best fitness: {self.best_fitness_}")

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

        ma = pa = None

        ma = selection.selection_strats[self.selection_type](fitness,
                                                             self.n_matings)
        pa = selection.selection_strats[self.selection_type](fitness,
                                                             self.n_matings)

        return ma, pa

    def get_number_mutations(self):
        return math.ceil(self.gene_size * self.mutation_ratio)

    def create_offspring(self, first_parent, sec_parent):
        """docstring
        """
        child1 = child2 = None
        child1, child2 = crossover.\
            crossover_strats[self.crossover_type](first_parent, sec_parent , self.cv)

        return child1, child2

    def mutate_population(self, population, n_mutation):
        """docstring
        """
        chosen_indices = random.sample(range(len(population)), n_mutation)

        for index in chosen_indices:
            population[index] = mutation.\
                mutation_strats[self.mutation_type](population[index],
                                                    **self.kwargs)

        return population

    @abstractmethod
    def initialize_population(self):
        """Sets up population array
        with randomized individual states

        returns:
            numpy array that models initial population
        """
        pass
