##TSP Solver 
from typing import Sequence
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
from TSP import TSPSolver, tsp_fitness
sys.path.append("..")
from geneticalg.core.AbstractSolver import AbstractSolver
import pandas as pd


def test_tsp(map, selection_func, mutation_func):
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
        pop_cnt=400, # population size (number of individuals)
        max_gen=300, # maximum number of generations
        mutation_ratio=0.5, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type=selection_func,
        crossover_type="one_point",
        mutation_type=mutation_func,
        verbose=True,
        cv=0
    )

    x = []
    y = []
    for curr_data in solver.solve():
        x.append(curr_data['iter'])
        y.append(curr_data['fitness'])
    return x, y


import matplotlib.pyplot as plt
import itertools

strat = ['roulette_wheel', 'stochastic', 'linear_rank']
mut = ['insert', 'flip', 'interchanging', 'reversing', 'uniform', 'creep']
for s, m in list(itertools.product(strat, mut)):
    print(s, m)
    try:
        x, y = test_tsp('map7.txt', s, m)
        plt.plot(x, y, label=(s, m))
    except:
        continue
    print("Done " + s + ' ' + m)

plt.xlabel('iter')
# Set the y axis label of the current axis.
plt.ylabel('1/fitness')
# Set a title of the current axes.
plt.title('TSPSolver with different parameters')
# show a legend on the plot
plt.legend()
plt.show()
plt.savefig('comparision.png')
# test_tsp("map7.txt", strat)

# plt.legend()
# plt.show()
