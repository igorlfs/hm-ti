from pathlib import Path

import networkx as nx
import numpy as np
import tsplib95

from src.algorithms import Algorithm
from src.helpers import calculate_distances, match_algorithm, plot_path

PROBLEMS_EUC_2D_PATH = "EUC_2D"


def run() -> None:
    problems_euc_2d = Path(PROBLEMS_EUC_2D_PATH).glob("*")
    # TODO df: Nome | Algoritmo | Tempo | Custo
    for problem_path in problems_euc_2d:
        results = []
        problem = tsplib95.load(problem_path)
        x = np.array([problem.node_coords[i][0] for i in problem.get_nodes()])
        y = np.array([problem.node_coords[i][1] for i in problem.get_nodes()])
        distances = calculate_distances(x, y)
        graph = nx.Graph(distances)
        algorithms = ["Christofides", "TATT", "MonteCarlo"]
        for algorithm in algorithms:
            result = match_algorithm(algorithm, graph)
            if result is not None:
                cost, path = result
                results.append(Algorithm(path, cost, algorithm))
        plot_path(x, y, results)
