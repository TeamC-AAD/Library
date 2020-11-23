# Genesis

[![PyPI version](https://badge.fury.io/py/pypi.svg)](https://badge.fury.io/py/pypi)
[![Github All Releases](https://img.shields.io/github/downloads/nickinack/Library/total.svg)]()


## Table of contents

* [Introduction](#Introduction)
* [Installation](#Installation)
* [Usage](#Usage)
* [Genetic Algorithms](#Genetic-Algorithms)
  * [Selection](#Selection)
  * [Crossover](#Crossover)
  * [Mutation](#Mutation)
* [Additional Features](#Additional-Features)
* [Applications](#Applications)
* [Creators](#Creators)

## Introduction

Genesis is a library created using Python in order to implement Genetic Algorithms. Genetic Algorithms(GAs) were inspired by the Charles Darwin theory of natural selection and we have added functionalities for an abstract GA solver and used it to optimize the travelling salesman problem (TSP), approximate the minimal roots of a non-linear equation and feature selection in the field of Machine Learning. 

## Installation

```bash
git clone https://github.com/TeamC-AAD/Library
```
Installation using ```pip``` will be updated soon.

After installation, use the following command to download all the packages:

```bash
 pip install -r requirements.txt
 ```
 
 ## Usage
 
 In order to use the Library, you must first create a fitness function. A sample fitness function has been shown below:
 
 ```python
 
 def fitness_func(chromosome):
    
    return 1/np.sum(chromosome)
```

Once the fitness function has been created, you must import the library and create an object of the ```AbstractSolver``` in the following way:

```python
solver = AbstractSolver(
        gene_size=var,
        fitness_func=lambda chromosome: fitness_func(chromosome), #Use your fitness function here
        pop_cnt=4000, # population size (number of individuals)
        max_gen=200, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="linear_rank",
        crossover_type="uniform",
        mutation_type="insert",
        verbose=True
    )
```

In order to start the optimisation process, use the following piece of code:

```python
for curr_data in solver.solve():
    print(curr_data ,"    ", value(curr_data["best_ind"]))
```       

## Genetic Algorithms 

#### Selection

- Rowlette Wheel
- Classical Linear Rank
- Stochastic Universal Sampling
- Tournament Selection

#### Crossover

- One-point crossover with value change
- One-point crossover without value change
- Two point crossover
- Uniform crossover

#### Mutation

- Insert mutation 
- Flip mutation
- Interchanging mutation
- Reversion mutation
- Uniform mutation
- Creep Mutation

## Additional Features

For problems like the Travelling salesman problem, you may need to initialize a population of your own / avoid value changes of the populations during mutations. In order to avoid value changes, you may set ```cv=0``` as illustrated in the following piece of code:

```python
TSPSolver(
        gene_size=len(scores)-1,
        fitness_func=lambda a : tsp_fitness(a , scores),
        pop_cnt=600, # population size (number of individuals)
        max_gen=500, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="roulette_wheel",
        crossover_type="one_point",
        mutation_type="insert",
        verbose=True,
        cv=0
    )
````

In order to create your own population array, we have added a functionality through which you may override the population method defined in the abstract solver. In order to do this, you must create your own solver class by inheriting the Abstract solver. After initialising it, you may override the population initialisation process using this function:

```python
  def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).
        :return: a numpy array with a randomized initialized population
        """
        pass
```

## Applications

Genesis has been applied for solving TSP, finding the minimal root (if exists) in order to satisfy a non-linear equation and feature selection in Machine Learning. In case of the TSP, Genesis works very well without premature convergence for nodes less than or equal to 30 (we used rowlette wheel selection , one point crossover without change values and interchanging mutation). The equation solver finds the most minimalistic root for a given equation (if exists); or optimizes the value of x if no real roots exists within 5 generations (we tried the equation solver for an equation with a degree of 243). For feature selection, a coding walkthrough of the entire algorithm has been provided in the Jupyter notebook. We have also wrapped these applications with frontend applications so you can have a comprehensive look on convergence values and get more insights from the walkthrough we provided (for Feature Selection).

## Creators

- <a href="https://github.com/ArenaGrenade">Rohan Asokan</a>
- <a href="https://github.com/abhishekh2001">Abhishekh Sivakumar</a>
- <a href="https://github.com/nickinack">Karthik Viswanathan</a>
