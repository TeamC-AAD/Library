import numpy as np

var = 1

def eqnfit(chromosome):

    '''
    Equation 1: x^143 + 2x^324 - x^13 = 10362
    '''

    eqn = chromosome[0]**143  + 2*chromosome[0]**324 - chromosome[0]**13 
    val = 10362
    return (1/(eqn-val))

def value(chromosome):

    return  chromosome[0]**143  + 2*chromosome[0]**324 - chromosome[0]**13 