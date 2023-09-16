from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.algorithms.helpers import monte_carlo_step

# Hiperparâmetros do simulated annealing
MONTE_CARLO_STEPS = 20000
DECAY_RATIO = 0.999


@dataclass
class Algorithm:
    path: list[int]
    cost: float
    name: str


def twice_around_the_tree(graph: nx.Graph) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando o `Twice Around The Tree`."""
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    cycle: list[int] = list(nx.dfs_preorder_nodes(mst, 0))
    cycle.append(0)
    return nx.path_weight(graph, cycle, "weight"), cycle


def christofides(graph: nx.Graph) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando o algoritmo de `Christofides`."""
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    odd_degree_vertices = [node for (node, val) in mst.degree if val % 2 == 1]
    odd_graph: nx.Graph = nx.induced_subgraph(graph, odd_degree_vertices)
    matching = nx.min_weight_matching(odd_graph)
    eulerian_multigraph = nx.MultiGraph()
    eulerian_multigraph.add_edges_from(mst.edges)
    eulerian_multigraph.add_edges_from(matching)
    edges: list[tuple[int, int]] = list(nx.eulerian_circuit(eulerian_multigraph, 0))
    cycle: list[int] = [0]
    for _, v in edges:
        if v in cycle:
            continue
        cycle.append(v)
    cycle.append(0)
    return nx.path_weight(graph, cycle, "weight"), cycle


def monte_carlo(graph: nx.Graph, initial_temperature: float) -> tuple[float, list[int]]:
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
        beta = beta * DECAY_RATIO
    best_path = best_path.tolist()
    best_path.append(best_path[0])
    return best_cost, best_path
