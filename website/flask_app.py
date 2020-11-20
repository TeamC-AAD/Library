from flask import Flask, Response, render_template, request
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
from eqnsolve.EQN import solveqn

app = Flask(__name__)
map_str = "map1.txt"
weights = np.array([[2, 3], [3, -5]])
outputs = np.array([4, 7])

@app.route('/')
def mainpage():
    return render_template("index.html")

@app.route('/report')
def reportpage():
    return render_template("report.html")

@app.route('/eqnsolve', methods=['GET', 'POST'])
def eqnsolve():
    global weights
    global outputs
    n_eq = random.randint(2, 6)
    n_eq = 2
    weights = np.random.randint(low=0, high=20, size=(n_eq, n_eq))
    outputs = np.random.randint(low=10, high=50, size=n_eq)
    
    print("weights: ")
    print(weights)
    print("outputs: ")
    print(outputs)

    return render_template('eqnsolve.html')
    # solveqn(weights, outputs)
    solveqn(weights, outputs)
    return f"{weights}, {outputs}"


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
