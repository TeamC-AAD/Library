from typing import Sequence

import numpy as np

from geneticalg.core.AbstractSolver import AbstractSolver


def get_input_dimensions(lst, n_dim=0):
    if isinstance(lst, (list, tuple)):
        if len(lst) > 0:
            return get_input_dimensions(lst[0], n_dim + 1)
        else:
            return 0
    else:
        return n_dim

def set_limits(g):
    # Retrieve machine min/max values
    min_max = np.iinfo(np.int64)
    # Generate the ranges
    limits = [(min_max.min, min_max.max) for _ in range(g)]
    return limits


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
        """Solves for floating point/integer based problems
        Arguments
            fitness_func: The function/class that the solver uses to optimize solution
            problem_type: Used to initialize population (can be float/int)
            gene_size: number of genes (variables) to have in each chromosome
            max_gen: maximum number of generations to perform the optimization
            pop_cnt: Number of individuals in current population
            mutation_ratio: rate at which random mutations occur
            selection_ratio: percentage of the population to be selected for crossover
            selection_type: type of selection algorithm to be employed
            mutation_type: Mutation algorithm to be employed
            verbose: To track/log or not
            variables_limits: limits for each variable of the form (min, max)
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
            variables_limits = set_limits(gene_size)

        if get_input_dimensions(variables_limits) == 1:
            variables_limits = [variables_limits for _ in range(gene_size)]

        self.variables_limits = variables_limits
        self.problem_type = problem_type

    def initialize_population(self):
        """Creates the population pool by generating random values
        based on the problem type (float or integer)

        Returns: a numpy array with a randomized initialized population
        """

        population = np.empty(shape=(self.pop_cnt, self.gene_size))

        if self.problem_type == float:
            for i, variable_limits in enumerate(self.variables_limits):
                population[:, i] = np.random.uniform(
                    variable_limits[0], variable_limits[1], size=self.pop_cnt
                )
        else:
            for i, variable_limits in enumerate(self.variables_limits):
                population[:, i] = np.random.randint(
                    variable_limits[0], variable_limits[1] + 1, size=self.pop_cnt
                )

        return population
