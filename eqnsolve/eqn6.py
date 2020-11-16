import numpy as np

var = 1

def eqnfit(chromosome):

    '''
    Equation 1: x^-2 + x^3 - x = 2
    '''

    eqn = chromosome[0]**-2 + chromosome[0]**3 - chromosome[0]
    val = 2
    return (1/(eqn-val))

def value(chromosome):

    return chromosome[0]**-2 + chromosome[0]**3 - chromosome[0]