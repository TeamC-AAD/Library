'''Crossover Algorithms'''
import numpy as np

def one_point_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    N = np.size(p1)
    div = np.random.randint(0,N)
    child1 = np.zeros(N)
    child2 = np.zeros(N)
    for i in range(0,N):
        if i<div:
            child1[i] = p1[i]
            child2[i] = p2[i]
        else:
            child1[i] = p2[i]
            child2[i] = p1[i]
    return child1,child2

def two_point_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    N = np.size(p1)
    div1 = 0
    div2 = 0
    child1 = np.zeros(N)
    child2 = np.zeros(N)
    while div1==div2:
        div1 = np.random.randint(0,N)
        div2 = np.random.randint(0,N)
    for i in range(0,N):
        if i<div1 or i>=div2:
            child1[i] = p1[i]
            child2[i] = p2[i]
        else:
            child1[i] = p2[i]
            child2[i] = p1[i]
    return child1,child2

def uniform_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    N = np.size(p1)
    child1 = np.zeros(N)
    child2 = np.zeros(N)
    for i in range(0,N):
        cp = np.random.choice([1,2])
        if cp==1:
            child1[i] = p1[i]
            child2[i] = p2[i]
        else:
            child1[i] = p2[i]
            child2[i] = p1[i]
    return child1,child2

            
