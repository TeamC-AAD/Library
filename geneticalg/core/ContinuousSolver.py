from typing import Sequence

import numpy as np

from geneticalg.core.AbstractSolver import AbstractSolver


def get_input_dimensions(lst, n_dim=0):
    if isinstance(lst, (list, tuple)):
        return get_input_dimensions(lst[0], n_dim + 1) if len(lst) > 0 else 0
    else:
        return n_dim


class ContinuousGenAlgSolver(AbstractSolver):
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
        mutation_type: str = "",
        crossover_type: str = "",
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        verbose: bool = False,
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
            **kwargs
        )

        if not variables_limits:
            min_max = np.iinfo(np.int64)
            variables_limits = [(min_max.min, min_max.max) for _ in range(gene_size)]

        if get_input_dimensions(variables_limits) == 1:
            variables_limits = [variables_limits for _ in range(gene_size)]

        self.variables_limits = variables_limits
        self.problem_type = problem_type

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).

        :return: a numpy array with a randomized initialized population
        """

        population = np.empty(shape=(self.pop_cnt, self.gene_size))

        for i, variable_limits in enumerate(self.variables_limits):
            if self.problem_type == float:
                population[:, i] = np.random.uniform(
                    variable_limits[0], variable_limits[1], size=self.pop_cnt
                )
            else:
                population[:, i] = np.random.randint(
                    variable_limits[0], variable_limits[1] + 1, size=self.pop_cnt
                )

        return population