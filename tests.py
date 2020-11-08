from geneticalg.core.ContinuousSolver import *


solver = ContinuousGenAlgSolver(
    gene_size=3,
    fitness_func=lambda chromosome: 1 / (sum(chromosome**2)-4),
    pop_cnt=2000, # population size (number of individuals)
    max_gen=300, # maximum number of generations
    mutation_ratio=0.4, # mutation rate to apply to the population
    selection_ratio=0.6, # percentage of the population to select for mating
    selection_type="linear_rank",
    crossover_type="uniform",
    mutation_type="insert",
    verbose=True
)


solver.solve()
