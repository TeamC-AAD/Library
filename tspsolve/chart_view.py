from flask import Flask, Response, render_template
import numpy as np
import pandas as pd
from .TSP import TSPSolver, tsp_fitness
import json
import random
from time import sleep

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/chart-data')
def chart_data():
    print("Called")
    # solver = test_tsp('map7.txt')
    def generate_data():
        solver = test_tsp('map7.txt')

        for curr_data in solver.solve():
            print(curr_data)
            curr = {
                'iter': curr_data['iter'],
                'fitness': curr_data['fitness']
            }
            print(f"ACTUAL: {curr}")
            json_data = json.dumps(curr)
            yield f"data:{json_data}\n\n"

    return Response(generate_data(), mimetype='text/event-stream')


@app.route('/chart_view_test')
def run_TSP_tester():
    solver = test_tsp('map7.txt')
    for curr_data in solver.solve():
        print(curr_data)
    return "Done with test"

def test_tsp(map):
    with open(map) as file:
        info = file.read()
    info = info.split("\n")
    info = [ x.split() for x in info ]
    scores = pd.DataFrame(info)
    scores = scores.to_numpy()
    scores = scores.astype(np.float)
    solver = TSPSolver(
        gene_size=len(scores)-1,
        fitness_func=lambda a : tsp_fitness(a , scores),
        pop_cnt=600, # population size (number of individuals)
        max_gen=500, # maximum number of generations
        mutation_ratio=0.4, # mutation rate to apply to the population
        selection_ratio=0.6, # percentage of the population to select for mating
        selection_type="stochastic",
        crossover_type="one_point",
        mutation_type="insert",
        verbose=True,
        cv=0
    )

    return solver


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

