import numpy as np
n_matings = 4

def np_ar_test():
    return np.array(
                list(map(lambda _: get_crossover_points(), range(n_matings)))
            )

def get_crossover_points():
    """
    Retrieves random crossover points
    :return: a numpy array with the crossover points
    """
    return np.sort(
        np.random.choice(
            np.arange(n_genes + 1), n_crossover_points, replace=False
        )
    )

n_crossover_points = 1
n_genes = 5
print(get_crossover_points())
print(np_ar_test())