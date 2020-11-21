from flask import Flask, Response, render_template, request, session
# from TSP import TSPSolver, tsp_fitness
import json
import random
import numpy as np
import pandas as pd
import random
import time

import sys
sys.path.append("..")
from tspsolve.TSP import TSPSolver, tsp_fitness
from eqnsolve.EQN import solveqn, value

app = Flask(__name__)
app.secret_key = "TEST_KEY"
map_str = "map1.txt"

powers = np.array([5, 3, 1])
weights = np.array([2, 3, 10])

@app.route('/')
def mainpage():
    return render_template("index.html")

@app.route('/report')
def reportpage():
    return render_template("report.html")

@app.route('/eqnsolve', methods=['GET', 'POST'])
def eqnsolve():
    global powers
    global weights 

    # Number of non-zero coefficients
    n_powers = random.randint(20, 30)
    # choose powers
    powers = np.random.choice(np.arange(-50, 50), replace=False, size=n_powers)
    # coefficients for each term
    weights = np.random.uniform(-100, 100, n_powers)

    session['powers'] = powers.tolist()
    session['weights'] = weights.tolist()
    
    
    print("powers: ")
    print(powers)
    print("weights: ")
    print(weights)

    # return render_template('eqnsolve.html')
    # list(solveqn(powers, weights))
    return render_template('eqnsolve.html')

@app.route('/eqn-chart-data')
def eqn_chart_data():
    global powers
    global weights
    print("Called")
    solver = solveqn(powers, weights)
    def generate_data(solver):
        best_ind = []
        for curr_data in solver.solve():
            print(curr_data)
            best_ind = curr_data['best_ind']
            curr = {
                'iter': curr_data['iter'],
                'fitness': curr_data['fitness'],
                'best_ind': best_ind.tolist(),
                'value': value(curr_data['best_ind'], powers, weights)
            }

            print(curr['iter'], curr['best_ind'], value(curr_data['best_ind'], powers, weights))

            json_data = json.dumps(curr)
            yield f"data:{json_data}\n\n"

        # Done here, generate fake data
        done_signal = {
            'iter': -1,
            'best_ind': best_ind.tolist(),
        }

        json_data = json.dumps(done_signal)
        yield f"data:{json_data}\n\n"

        return None
    
    return Response(generate_data(solver), mimetype='text/event-stream')



@app.route('/tsp', methods=['GET', 'POST'])
def tsp():
    mp = request.args.get("map")
    if mp is None:
        return "Map file required"

    global map_str
    map_str = mp
    return render_template('tsp.html')


@app.route('/chart-data')
def chart_data():
    print("Called")
    global map_str
    solver, adjacency_matrix = test_tsp('assets/' + map_str)

    def generate_data(solver):
        best_ind = []
        for curr_data in solver.solve():
            print(curr_data)
            best_ind = curr_data['best_ind']
            curr = {
                'iter': curr_data['iter'],
                'fitness': curr_data['fitness'],
                'best_ind': best_ind.tolist(),
                'graph': adjacency_matrix
            }

            print({curr['iter'], curr['fitness']})

            json_data = json.dumps(curr)
            yield f"data:{json_data}\n\n"
        

        # Done here, generate fake data
        done_signal = {
            'iter': -1,
            'best_ind': best_ind.tolist(),
            'graph': adjacency_matrix
        }

        json_data = json.dumps(done_signal)
        yield f"data:{json_data}\n\n"


        return None
    
    return Response(generate_data(solver), mimetype='text/event-stream')


def test_tsp(map):
    with open(map) as file:
        info = file.read()
    info = info.split("\n")
    info = [ x.split() for x in info ]
    scores = pd.DataFrame(info)
    scores = scores.to_numpy()
    scores = scores.astype(np.float)
    solver = TSPSolver(
        gene_size=len(scores),
        fitness_func=lambda a : tsp_fitness(a , scores),
        pop_cnt=600, # population size (number of individuals)
        max_gen=270, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="stochastic",
        crossover_type="one_point",
        mutation_type="insert",
        verbose=True,
        cv=0
    )

    return solver, scores.tolist()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
