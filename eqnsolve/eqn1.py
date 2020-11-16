import numpy as np

var = 3

def eqnfit(chromosome):

    '''
    Equation 1: x^2-x + y^3-y = -1
    '''

    eqn = chromosome[0]**2 - chromosome[0] + chromosome[1]**3 - chromosome[1]
    val = -1
    return (1/(eqn-val))

def value(chromosome):

    return chromosome[0]**2 - chromosome[0] + chromosome[1]**3 - chromosome[1]