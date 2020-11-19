from flask import Flask, Response, render_template
import numpy as np
import pandas as pd
from .TSP import TSPSolver, tsp_fitness
import json
import random
from time import sleep
import networkx as nx
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt



app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/chart-data')
def chart_data():
    print("Called")
    solver, adjacency_matrix = test_tsp('map7.txt')

    def generate_data(solver):
        best_ind = []
        for curr_data in solver.solve():
            print(curr_data)
            curr = {
                'iter': curr_data['iter'],
                'fitness': curr_data['fitness']
            }

            best_ind = curr_data['best_ind']

            print(f"ACTUAL: {curr}")
            json_data = json.dumps(curr)
            yield f"data:{json_data}\n\n"

        # draw_graph(adjacency_matrix, best_ind)

        # Done here, generate fake data
        done_signal = {
            'iter': -1,
            'best_ind': best_ind.tolist(),
            'graph': adjacency_matrix
        }

        json_data = json.dumps(done_signal)
        yield f"data:{json_data}\n\n"

        while True:  # Stop client from redrawing graph
            True

    return Response(generate_data(solver), mimetype='text/event-stream')


@app.route('/chart_view_test')
def run_TSP_tester():
    solver, matrix = test_tsp('map7.txt')
    for curr_data in solver.solve():
        print(curr_data)
    return "Done with test"


def draw_graph(adjacency_matrix, path):
    """ Draws the network matrix
    """
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    # nx.drawing.nx_pylab.draw_networkx_nodes(gr, )
    plt.plot()
    plt.savefig('map.png')




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
    app.run(host='127.0.0.1', port=8080, debug=True, threaded=True)

