import numpy as np

var = 3

def eqnfit(chromosome):

    '''
    Equation 1: x^-2 + y^-3 - z^-1 = 0.4
    '''

    eqn = chromosome[0]**-2 + chromosome[1]**-3 - chromosome[2]**-1
    val = 0.4
    return (1/(eqn-val))

def value(chromosome):

    return chromosome[0]**-2 + chromosome[1]**-3 - chromosome[2]**-1