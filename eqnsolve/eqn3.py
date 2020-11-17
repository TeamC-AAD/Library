import numpy as np

var = 3

def eqnfit(chromosome):

    '''
    Equation 1: x^53 + y^42 - z^76 = 0
    '''

    eqn = chromosome[0]**53 + chromosome[1]**42 - chromosome[2]**76
    val = 10
    return (1/(eqn-val))

def value(chromosome):

    return  chromosome[0]**53 + chromosome[1]**42 - chromosome[2]**76