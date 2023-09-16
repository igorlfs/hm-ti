import time as ti
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import tsplib95
from scipy.spatial import distance

from src.algorithms import match_algorithm

PROBLEMS_EUC_2D_PATH = "EUC_2D"


def run() -> None:
    data: list[list] = []
    problems_euc_2d = Path(PROBLEMS_EUC_2D_PATH).glob("*")
    for problem_path in problems_euc_2d:
        problem = tsplib95.load(problem_path)
        coords = np.array([problem.node_coords[i] for i in problem.get_nodes()])
        distances = distance.cdist(coords, coords, "euclidean")
        graph = nx.Graph(distances)
        algorithms = ["Christofides", "TATT", "MonteCarlo"]
        for algorithm in algorithms:
            start = ti.time()
            result = match_algorithm(algorithm, graph)
            end = ti.time()
            time = end - start
            if result is not None:
                cost, _ = result
                data.append([problem.name, algorithm, time, cost])
    df = pd.DataFrame(data, columns=["Name", "Algorithm", "Time", "Cost"])
    df.to_csv("results.csv", index=False)
