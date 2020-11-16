import numpy as np

var = 3

def eqnfit(chromosome):

    '''
    Equation 1: x^3 + y^8 - z^2 = 2
    '''

    eqn = chromosome[0]**3 + chromosome[1]**8 - chromosome[2]**2
    val = 2
    return (1/(eqn-val))

def value(chromosome):

    return chromosome[0]**3 + chromosome[1]**8 - chromosome[2]**2