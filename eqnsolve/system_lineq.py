import numpy as np
import math

var = 2

n_eq = 3
n_un = 2

weights = np.array([[2, 3], [3, -5]])
outputs = np.array([4, 7])


def eqnfit(chromosome):
    '''
    Equation 1:
    '''

    output_model = np.dot(weights, np.array(chromosome))

    error = np.sum(np.power(output_model - outputs, 2))

    if error == 0:
        error = 1

    return 1 / error



def value(chromosome):
    return np.dot(weights, np.array(chromosome))