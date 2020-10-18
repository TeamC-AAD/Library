from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver


class BinaryGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        n_bits: int = 1,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = True,
        show_stats: bool = True,
        plot_results: bool = True,
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
    ):
        """
        :param fitness_function: can either be a fitness function or a class implementing a fitness function +
        methods to override the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param n_bits: number of bits representing each gene
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes * n_bits,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            selection_strategy=selection_strategy,
            verbose=verbose,
            show_stats=show_stats,
            plot_results=plot_results,
            excluded_genes=excluded_genes,
            n_crossover_points=n_crossover_points,
            random_state=random_state,
        )

    def initialize_population(self):
        """
        Input:
        Output: Numpy array with randomized initialized population
        """

        pop = []

        for i in range(self.pop_size):
            bits = np.zeros(self.n_genes)
            bits[: np.random.randint(0, self.n_genes)] = 1
            np.random.shuffle(bits)

            pop.append(bits)

        return np.array(pop)

    @staticmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, _):
        """
        Input: Parent1 , Parent2 , crossover_pt
        Output: Offspring 
        """

        return np.hstack(
            (first_parent[: crossover_pt[0]], sec_parent[crossover_pt[0] :])
        )

    def mutate_population(self, population, n_mutations):
        """
        Input: Population at a given iter , number of mutations to be performed
        Output: Mutated Population
        """

        mutation_rows, mutation_cols = super(
            BinaryGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = np.abs(population - 1)[
            mutation_rows, mutation_cols
        ]

        return population