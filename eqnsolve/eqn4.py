import numpy as np

var = 6

def eqnfit(chromosome):

    '''
    Equation 1: x^53 + y^42 - z^76 + a^106 + b^25 - c^46 = 0
    '''

    eqn = chromosome[0]**53 + chromosome[1]**42 - chromosome[2]**76 + chromosome[3]**106 + chromosome[4]**25 - chromosome[5]**46
    val = 226
    return (1/(eqn-val))

def value(chromosome):

    return  chromosome[0]**53 + chromosome[1]**42 - chromosome[2]**76 + chromosome[3]**106 + chromosome[4]**25 - chromosome[5]**46