import math
from typing import Sequence

import numpy as np

# ! Need to refactor this ASAP
allowed_selection_strategies = {}

class AbstractSolver:
    def __init__(
        self,
        gene_size: int,
        population_size: int = 100,
        generations: int = 1000,
        fitness_function = None,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        excluded_genes: Sequence = None,
        crossover_points: int = 1,
        random: int = None,
        verbose: bool = True,
        show_exit_stats: bool = True,
        plot_results: bool = True,
    ):
        """
        Initialization function for the AbstractSolver class. When a child class is being created that is a solver, 
        it needs to be initialized by a call to this function.

        Args:
            gene_size (int): The number of genes in a chromosome.
            population_size (int, optional): The size of the population for a single generation of the run. Defaults to 100.
            generations (int, optional): Number of generations to run for in the simulation. Defaults to 1000.
            fitness_function (function, optional): A functor or a class that has overriding functions that needs to be provided that 
            calculates the fitness of a genome and returns the fitness. Defaults to None.
            mutation_rate (float, optional): The rate at which mutation will occur in a genome. Defaults to 0.15.
            selection_rate (float, optional): How likely is a genome to get selected. Defaults to 0.5.
            selection_strategy (str, optional): The strategy used for selection. Defaults to "roulette_wheel".
            excluded_genes (Sequence, optional): Genes that are excluded in a run. Defaults to None.
            crossover_points (int, optional): Number of points to consider when performing k-point crossover. Defaults to 1.
            random (int, optional): Initial state of the simulation. This should be used to continue training of a model. Defaults to None.
            verbose (bool, optional): If set to True, gives verbose messages. Defaults to True.
            show_exit_stats (bool, optional): If set to True, shows population statistics of the last generation. Defaults to True.
            plot_results (bool, optional): If True, plots the results of the run at the end. Defaults to True.
        """
        
        if isinstance(random, int):
            np.random.seed(random)

        self.gene_size = gene_size
        # ? What is this random allowed_mutation_genes thingy doing?
        self.allowed_mutation_genes = np.arrange(self.gene_size)

        # TODO: some check base function        

        self.selection_strategy = selection_strategy

        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.crossover_points = crossover_points
        self.verbose = verbose
        self.show_exit_stats = show_exit_stats
        self.plot_results = plot_results

        self.population_retain = math.floor(selection_rate * population_size)

        if self.population_retain < 2:
            self.population_retain = 2
        
        # TODO: Add the get_selection_probabilities stuff

        self.mating_count = math.floor((self.population_size - self.population_retain) / 2)
        # TODO: Add the get_number_mutations stuff

        self._generation = 0
        self._best_individual = None
        self._best_fitness = 0
        self._population = None
        self._fitness = None
    

    def check_input_base(
        self, 
        fitness_function, 
        selection_strategy, 
        population_size, 
        excluded_genes,
    ):
        # TODO: Add the docstring for this function
        if not fitness_function:
            try:
                getattr(self, "fitness_function")
            except AttributeError:
                # TODO: here goes the exception handling code - need for new file
                pass
        else:
            self.fitness_function = fitness_function
        
        if selection_strategy not in allowed_selection_strategies:
            # TODO: Need to handle the exception here
            pass

        if population_size < 2:
            # TODO: Need to handle the corresponding exception
            pass

        if isinstance(excluded_genes, (list, tuple, np.ndarray)):
            self.allowed_mutation_genes = [
                item
                for item in self.allowed_mutation_genes
                if item not in excluded_genes
            ]

        elif excluded_genes is not None:
            # TODO: Raise needed exceptions here
            pass
        
