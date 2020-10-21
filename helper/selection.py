"""Defines all the available selection methods"""
import sys
import numpy as np

def proportional_roulette_wheel():
    # TODO: Fill function
    pass

def stochastic_universal_sampling():
    # TODO: Fill function
    pass

def classic_linear_rank():
    # TODO: Fill function
    pass

def linear_rank_with_selective_pressure():
    # TODO: Fill function
    pass

def tournament_selection(pop,fitness,k):
    '''
    Input: population,fitness and a number k.
    Output: the function conducts tournaments between k individuals randomly and selects the best
    ''' 
    N = np.size(pop)
    best = -1
    fit  = -sys.maxsize-1
    for _ in range(1,k):
        ind = pop[np.random.randint(0, N)]
        if (fit == -1) or fitness[ind] > fitness[best]:
            best = ind
            fit = fitness[ind]
    return best

def boltzmann_selection(temp_gene , fitness , pop):
    #TODO: Fill function
    pass

