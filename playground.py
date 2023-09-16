from dataclasses import dataclass
from functools import reduce

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tsplib95
from matplotlib.lines import Line2D
from numpy import float64, int64
from numpy.typing import NDArray

# |%%--%%| <TyichS2zRM|xpM9ojUbAd>


@dataclass
class Algorithm:
    path: list[int]
    cost: float
    name: str


# |%%--%%| <xpM9ojUbAd|E36rET3VZI>


def calculate_distances(
    n: int, x: NDArray[float64], y: NDArray[float64]
) -> NDArray[float64]:
    """Calcula as distâncias entre todas as `n` cidades."""
    distance = np.zeros((n, n), dtype=float64)
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.sqrt(
                np.power((x[i] - x[j]), 2) + np.power((y[i] - y[j]), 2)
            )
    return distance


# |%%--%%| <E36rET3VZI|e64qA0lNRP>


def twice_around_the_tree(graph: nx.Graph) -> tuple[float, list[int]]:
    """Aproxime o caixeiro viajante para `graph` usando o `twice_around_the_tree`."""
    mst: nx.Graph = nx.minimum_spanning_tree(graph)
    cycle: list[int] = list(nx.dfs_preorder_nodes(mst, 0))
    cycle.append(0)
    return nx.path_weight(graph, cycle, "weight"), cycle


# |%%--%%| <e64qA0lNRP|vvfixhkV68>


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


# |%%--%%| <vvfixhkV68|YqNOkE3pR4>


def calculate_new_path(path: NDArray[int64]) -> tuple[NDArray[int64], int, int]:
    """Calcula o novo caminho, trocando dois índices."""
    num_cities = len(path)
    # define uma nova caminhada
    rng = np.random.default_rng()

    idx: NDArray = rng.choice(np.arange(num_cities), 2, replace=False)

    # o índice inicial deve ser o menor
    beg, end = (idx[0], idx[1]) if idx[0] < idx[1] else (idx[1], idx[0])

    # inverte o sentido em que percorre o caminho entre os indices escolhidos
    new_path = np.zeros(num_cities, dtype=int64)
    for k in range(num_cities):
        new_path[k] = path[end - k + beg] if k >= beg and k <= end else path[k]

    return new_path, beg, end


# |%%--%%| <YqNOkE3pR4|y8ZEyIk3aB>


def monte_carlo_step(
    beta: float,
    cost: float,
    path: NDArray[int64],
    best_cost: float,
    best_path: NDArray[int64],
    dist: NDArray[float64],
) -> tuple[float, NDArray[int64], float, NDArray[int64]]:
    """Rode um passo de Monte Carlo para o TSP."""
    num_cities = len(path)
    new_path = np.zeros(num_cities, dtype=int64)

    # propõe um novo caminho
    new_path, beg, end = calculate_new_path(path)

    # determina a diferença de custo
    left_from_beg = (beg - 1) % num_cities
    right_from_end = (end + 1) % num_cities
    delta: float = (
        -dist[path[left_from_beg], path[beg]]
        - dist[path[right_from_end], path[end]]
        + dist[new_path[left_from_beg], new_path[beg]]
        + dist[new_path[right_from_end], new_path[end]]
    )

    # se o custo diminuir, a gente sempre troca
    if delta < 0:
        cost += delta
        path = new_path
        if cost < best_cost:
            best_cost = cost
            best_path = path
    # aplica o criterio de Metropolis
    elif np.random.random() < np.exp(-beta * delta):
        cost += delta
        path = new_path

    return cost, path, best_cost, best_path


# |%%--%%| <y8ZEyIk3aB|qziE6R7OWR>


def monte_carlo(
    distances: NDArray[float64],
    num_cities: int,
    temperature: float,
    num_steps: int,
    decay: float,
) -> tuple[float, list[int]]:
    path = np.array(list(range(num_cities)))
    beta = temperature
    cost = reduce(
        lambda a, b: a + b,
        [
            distances[path[i % num_cities]][path[(i + 1) % num_cities]]
            for i in range(num_cities)
        ],
    )
    best_cost = cost
    best_path = path
    for _ in range(num_steps):
        cost, path, best_cost, best_path = monte_carlo_step(
            beta, cost, path, best_cost, best_path, distances
        )
        beta = beta * decay
    best_path = best_path.tolist()
    best_path.append(best_path[0])
    return best_cost, best_path


# |%%--%%| <qziE6R7OWR|czLS8aiJfd>


def plot_path(x: NDArray, y: NDArray, results: list[Algorithm]) -> None:
    colors = ["blue", "red", "green", "yellow", "purple", "orange"]
    handles = []
    labels = []
    for k, alg in enumerate(results):
        for i, j in zip(alg.path, alg.path[1:]):
            plt.plot([x[i], x[j]], [y[i], y[j]], color=colors[k], label=k)
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[k],
                label=f"{alg.name} {alg.cost}",
            )
        )
        labels.append(f"{alg.name} {alg.cost}")

    plt.legend(handles=handles, labels=labels)
    # plt.show(block=False)


# |%%--%%| <czLS8aiJfd|kCX3SBVH6k>

PROBLEM_PATH = "../EUC_2D/berlin52.tsp"
problem = tsplib95.load(PROBLEM_PATH)

# |%%--%%| <kCX3SBVH6k|PEvwJA6A8o>

x = np.array([problem.node_coords[i][0] for i in problem.get_nodes()])
y = np.array([problem.node_coords[i][1] for i in problem.get_nodes()])
num_cities = len(x)
distances = calculate_distances(num_cities, x, y)

# |%%--%%| <PEvwJA6A8o|zZodlbaXYS>

G = nx.Graph(distances)

# |%%--%%| <zZodlbaXYS|wPzWwCC8k6>

results = []

# |%%--%%| <wPzWwCC8k6|QYLG74krmh>

cost, path = christofides(G)
results.append(Algorithm(path, cost, "Christofides"))

# |%%--%%| <QYLG74krmh|x8QKVQDQgQ>

cost, path = twice_around_the_tree(G)
results.append(Algorithm(path, cost, "TATT"))

# |%%--%%| <x8QKVQDQgQ|n4gVQLrghw>


MC_STEPS = 20000
TEMPERATURE = 3 * num_cities
DECAY = 0.9999


# |%%--%%| <n4gVQLrghw|BHamLSEWwH>

cost, path = monte_carlo(distances, num_cities, TEMPERATURE, MC_STEPS, DECAY)
results.append(Algorithm(path, cost, "MonteCarlo"))

# |%%--%%| <BHamLSEWwH|opoHV5VOFb>

plot_path(x, y, results)

# |%%--%%| <opoHV5VOFb|1W0bXpxQ5a>
