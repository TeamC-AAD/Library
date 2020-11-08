'''Crossover Algorithms'''
import numpy as np

def one_point_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    beta = np.random.rand(1)[0]
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
    child1[div] = p1[div] + beta*(p2[div] - p1[div])
    child2[div] = p1[div] - beta*(p2[div] - p1[div])
    return child1,child2

def two_point_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    beta = np.random.rand(1)[0]
    gamma = np.random.rand(1)[0]
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
    child1[div1] = p1[div1] + beta*(p2[div1] - p1[div1])
    child2[div1] = p1[div1] - beta*(p2[div1] - p1[div1])

    child1[div1] = p1[div1] + gamma*(p2[div1] - p1[div1])
    child2[div1] = p1[div1] - gamma*(p2[div1] - p1[div1])

    return child1,child2

def uniform_crossover(p1 , p2):
    '''
    Given two chromosomes, returns the child chromosomes
    '''
    N = np.size(p1)
    child1 = np.zeros(N)
    child2 = np.zeros(N)
    beta = np.random.rand(1)[0]
    div1 = np.random.randint(0,N)
    for i in range(0,N):
        cp = np.random.choice([1,2])
        if cp==1:
            child1[i] = p1[i]
            child2[i] = p2[i]
        else:
            child1[i] = p2[i]
            child2[i] = p1[i]
    child1[div1] = p1[div1] + beta*(p2[div1] - p1[div1])
    child2[div1] = p1[div1] - beta*(p2[div1] - p1[div1])
    return child1,child2

def test_crossover(func):
    p1 = [1,3,2,5,4,2,1]
    p2 = [3,5,2,6,1,4,3]

    for i in range(0,100):
        p1,p2 = func(p1,p2)

    print(p1)
    print(p2)

'''
print("Testing one point crossover")
test_crossover(one_point_crossover)

print("Testing two point crossover")
test_crossover(two_point_crossover)

print("Testing uniform crossover")
test_crossover(uniform_crossover)
'''

crossover_strats = {
    "one_point"     :   one_point_crossover,
    "two_point"     :   two_point_crossover,
    "uniform"       :   uniform_crossover
}
