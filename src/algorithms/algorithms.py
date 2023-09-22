import time as ti
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import tsplib95
from scipy.spatial import distance

from .helpers import monte_carlo_step

# Hiperparâmetros do simulated annealing
MONTE_CARLO_STEPS = 20000


@dataclass
class Algorithm:
    path: list[int]
    cost: float
    name: str


def get_time_and_cost(problem_path: Path, data: list[list]) -> None:
    """Calcula o tempo e o custo de aproximar a instância do TSP em `problem_path`,
    usando difentes algoritmos e adicionando os resultados em `data`."""
    problem = tsplib95.load(problem_path)
    coords = np.array([problem.node_coords[i] for i in problem.get_nodes()])
    distances = distance.cdist(coords, coords, "euclidean")
    if problem_path.name.find("att") != -1:
        distances = np.trunc(distances)  # corte nas instâncias não euclidianas
    graph = nx.Graph(distances)
    algorithms = ["Christofides", "TATT", "Simulated Annealing"]
    for algorithm in algorithms:
        start = ti.time()
        result = match_algorithm(algorithm, graph)
        end = ti.time()
        time = end - start
        if result is not None:
            cost, _ = result
            data.append([problem.name, algorithm, time, cost])


def match_algorithm(name: str, graph: nx.Graph) -> tuple[float, list[int]] | None:
    """Casa um nome com uma função de aproximação para o TSP."""
    match name:
        case "TATT":
            return twice_around_the_tree(graph)
        case "Christofides":
            return christofides(graph)
        case "Simulated Annealing":
            return simulated_annealing(graph, 3 * graph.number_of_nodes())


def twice_around_the_tree(graph: nx.Graph) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando o `Twice Around The Tree`."""
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    cycle: list[int] = list(nx.dfs_preorder_nodes(mst, 0))
    cycle.append(0)
    return nx.path_weight(graph, cycle, "weight"), cycle


def christofides(graph: nx.Graph) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando o algoritmo de `Christofides`."""
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    assert not isinstance(mst.degree, int)
    odd_degree_vertices = [node for (node, val) in mst.degree if val % 2 == 1]
    odd_graph: nx.Graph = nx.induced_subgraph(graph, odd_degree_vertices)
    matching = nx.min_weight_matching(odd_graph)
    eulerian_multigraph = nx.MultiGraph()
    eulerian_multigraph.add_edges_from(mst.edges)
    eulerian_multigraph.add_edges_from(matching)
    edges: list[tuple[int, int]] = list(nx.eulerian_circuit(eulerian_multigraph, 0))
    cycle: list[int] = [0]
    [cycle.append(v) for _, v in edges if v not in cycle]
    cycle.append(0)
    return nx.path_weight(graph, cycle, "weight"), cycle


def simulated_annealing(
    graph: nx.Graph, initial_temperature: float
) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando `Simulated Annealing`."""
    distances = nx.to_numpy_matrix(graph)
    path = np.arange(len(distances))
    cost = nx.path_weight(graph, [*path, 0], "weight")
    best_cost = cost
    best_path = path
    beta = initial_temperature
    for _ in range(MONTE_CARLO_STEPS):
        cost, path, best_cost, best_path = monte_carlo_step(
            beta, cost, path, best_cost, best_path, distances
        )
    best_path = best_path.tolist()
    best_path.append(best_path[0])
    return best_cost, best_path
