import numpy as np
import random
from misc import *
"""Defines all the available selection methods"""

def proportional_roulette_wheel(population):
    """ Performs roulette wheel selection:
    every individual can become a parent with a 
    probability which is proportional to its fitness.
    Selects one parent.
    """

    # Compute the total fitness of population
    sum_fitness = sum([entity_fitness for entity_fitness in population])

    # Each entity in population is given a probability
    # to become a parent proportional to the fitness of the
    # individual
    selection_probabilities = [entity_fitness/sum_fitness for entity_fitness in population]

    # Select the index of the parent chosen using the
    # probabilities computed for each individual in the population
    parent_ind = np.random.choice(len(population), p=selection_probabilities)
    return parent_ind


def stochastic_universal_sampling(population, N):
    """SUS uses a single random value to sample all of the solutions by 
    choosing them at evenly spaced intervals, giving
    weaker members of the population (according to their fitness) 
    chance to be chosen.
    Produces a parent pool of size N
    """

    # https://stackoverflow.com/questions/22749132/stochastic-universal-sampling-ga-in-python
    # https://en.wikipedia.org/wiki/Stochastic_universal_sampling
    wheel = makeWheel(population)
    stepSize = 1.0/N
    answer = []
    r = random.random()

    answer.append(binSearch(wheel, r))

    # step through
    while len(answer) < N:
        r += stepSize
        if r>1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer


def classic_linear_rank(population_fitness):
    """ RWS will have problems when the finesses differs very much. 
    Outstanding individuals will introduce a bias in the beginning 
    of the search that may cause a premature convergence and a loss 
    of diversity.

    http://www.ijmlc.org/papers/146-C00572-005.pdf
    """

    # TODO: this doesn't even select lol
    
    # Get the number of individuals in the population.
    n = len(population_fitness)

    # (sum of integers 1 to N).
    rank_sum = n * (n + 1) / 2

    fitness_intermediary = list()

    # Sort and go through all individual fitnesses; enumerate ranks from 1.
    for rank, ind_fitness in enumerate(sorted(population_fitness), 1):
        fitness_intermediary.append(float(rank) / rank_sum)
    
    # print(fitness_intermediary)
    # qq = input()
    
    return proportional_roulette_wheel(fitness_intermediary)

def linear_rank_with_selective_pressure():
    # TODO: Fill function
    pass

def tournament_selection():
    # TODO: Fill function
    pass

def transform_ranking_selection():
    # TODO: Fill function
    pass


# Very simple tests to view workings
def selection_test(f_func):
    p_fitness = [2, 4, 6, 1, 7, 10, 1, 20]
    cnt = dict()
    total_iters = 100000
    for i in range(total_iters):
        ind = f_func(p_fitness)
        if ind in cnt:
            cnt[ind] += 1
        else:
            cnt[ind] = 1

    print("key  fitness  percent_selected")
    for key, value in cnt.items():
        print(key, ' ', p_fitness[key], ' ', value/total_iters)

def sus_test():
    p_fitness = [2, 4, 6, 1, 7, 10, 1, 13]

    # selection_test(proportional_roulette_wheel)
    # total_iters = len(p_fitness)
    total_iters = 10000
    res = stochastic_universal_sampling(p_fitness, total_iters)
    for i in set(res):
        print(i, ' ', p_fitness[i], ' ', res.count(i)/total_iters)

# sus_test()
# selection_test(classic_linear_rank)