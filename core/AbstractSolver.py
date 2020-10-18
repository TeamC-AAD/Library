from typing import Sequence

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
        crossober_points: int = 1,
        start_state: int = None,
        verbose: bool = True,
        show_exit_stats: bool = True,
        plot_results: bool = True,
    ):
        """Initialization function for the AbstractSolver class. When a child
        class is being created that is a solver, it needs to be initialized
        by a call to this function.

        Args:
            gene_size (int): The number of genes in a chromosome.
            population_size (int, optional): The size of the population for a single generation of the run. Defaults to 100.
            generations (int, optional): Number of generations to run for in the simulation. Defaults to 1000.
            fitness_function (function, optional): A functor that needs to be provided that calculates the fitness of a genome and returns the fitness. Defaults to None.
            mutation_rate (float, optional): [description]. Defaults to 0.15.
            selection_rate (float, optional): [description]. Defaults to 0.5.
            selection_strategy (str, optional): [description]. Defaults to "roulette_wheel".
            excluded_genes (Sequence, optional): [description]. Defaults to None.
            crossober_points (int, optional): [description]. Defaults to 1.
            start_state (int, optional): [description]. Defaults to None.
            verbose (bool, optional): [description]. Defaults to True.
            show_exit_stats (bool, optional): [description]. Defaults to True.
            plot_results (bool, optional): [description]. Defaults to True.
        """
        pass