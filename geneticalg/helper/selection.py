"""Defines all the available selection methods"""
import numpy as np
import random
from .misc import binSearch, makeWheel
import sys
import math


def proportional_roulette_wheel(population, N):
    """ Performs roulette wheel selection:
    every individual can become a parent with a
    probability which is proportional to its fitness.
    Selects one parent.
    https://stackoverflow.com/questions/177271/roulette-selection-in-genetic-algorithms/5315710#5315710

    Args:
        N(int): Number of parents to choose
        population(list): List containing fitness values of individuals

    Returns:
        int: Index of parent chosen
    """

    update = min(population)
    population = [fitness + math.fabs(update) for fitness in population]

    # Compute the total fitness of population
    sum_fitness = sum([entity_fitness for entity_fitness in population])

    # Each entity in population is given a probability
    # to become a parent proportional to the fitness of the
    # individual
    selection_probabilities = [entity_fitness/sum_fitness for
                               entity_fitness in population]

    # Select the index of the parent chosen using the
    # probabilities computed for each individual in the population
    parents = []
    for i in range(N):
        parent_ind = np.random.choice(len(population),
                                      p=selection_probabilities)
        parents.append(parent_ind)
    return parents


def stochastic_universal_sampling(population, N):
    """SUS uses a single random value to sample all of the solutions by
    choosing them at evenly spaced intervals, giving
    weaker members of the population (according to their fitness)
    chance to be chosen.
    Produces a parent pool of size N

    Args:
        population (List): List containing fitness values of individuals
        N (int): Number of parents

    Returns:
        list: The indices of parents
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
        if r > 1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer


def classic_linear_rank(population_fitness, N):
    """ RWS will have problems when the finesses differs very much.
    Outstanding individuals will introduce a bias in the beginning
    of the search that may cause a premature convergence and a loss
    of diversity.

    http://www.ijmlc.org/papers/146-C00572-005.pdf

    Args:
        population_fitness(list): numpy array containing fitness values of individuals

    Returns:
        int: Index of parent chosen
    """

    # Create ranks
    ranks = [0 for x in population_fitness]
    sorted_ind = np.argsort(population_fitness)

    pos = 0
    for ind in sorted_ind:
        ranks[ind] = pos
        pos += 1

    return proportional_roulette_wheel(ranks, N)


def tournament_selection(fitness, n_parents):
    '''
    Input: population,fitness and a number k.
    Output: the function conducts tournaments between k individuals randomly and selects the best
    '''
    k = np.random.randint(1, len(fitness))
    N = len(fitness)
    indx = []
    for _ in range(n_parents):
        best = -1
        fit = -sys.maxsize - 1
        for _ in range(1, k):
            ind = np.random.randint(0, N)
            if (fit == -1) or fitness[ind] > fit:
                best = ind
                fit = fitness[ind]
        indx.append(best)
    # print(indx)
    return indx


def selection_test(func):
    p_fitness = np.array([2, 4, 6, 1, 7, 10, 5, 20])
    cnt = dict()
    total_iters = 10000
    print(p_fitness)
    res = func(p_fitness, total_iters)
    for ind in res:
        if ind in cnt:
            cnt[ind] += 1
        else:
            cnt[ind] = 1

    print("key  fitness  percent_selected")
    for key, value in sorted(cnt.items(), key=lambda x: x[1]/total_iters):
        print(key, ' ', p_fitness[key], ' ', value/total_iters)

'''
Tests
'''

'''
print("Testing proportional roulette wheel")
selection_test(proportional_roulette_wheel)

print("Testing SUG")
selection_test(stochastic_universal_sampling)

print("Testing linear rank: ")
selection_test(classic_linear_rank)

print("Testing tournament selection")
selection_test(tournament_selection)
'''

selection_strats = {
    "roulette_wheel"        :   proportional_roulette_wheel,
    "stochastic"            :   stochastic_universal_sampling,
    "linear_rank"           :   classic_linear_rank,
    "tournament"            :   tournament_selection
}
