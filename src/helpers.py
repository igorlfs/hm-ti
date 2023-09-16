import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from numpy import float64
from numpy.typing import NDArray

from src.algorithms import Algorithm, christofides, monte_carlo, twice_around_the_tree

mpl.use("QtCairo")


def match_algorithm(name: str, graph: nx.Graph) -> tuple[float, list[int]] | None:
    """Casa um nome com uma função de aproximação para o TSP."""
    match name:
        case "TATT":
            return twice_around_the_tree(graph)
        case "Christofides":
            return christofides(graph)
        case "MonteCarlo":
            return monte_carlo(graph, 3 * graph.number_of_nodes())


def calculate_distances(x: NDArray[float64], y: NDArray[float64]) -> NDArray[float64]:
    """Calcula a distância euclidiana entre todas as `n` cidades."""
    distance = np.zeros((x.size, y.size), dtype=float64)
    for i in range(x.size):
        for j in range(y.size):
            distance_squared = np.power((x[i] - x[j]), 2) + np.power((y[i] - y[j]), 2)
            distance[i, j] = np.sqrt(distance_squared)
    return distance


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
                label=f"{alg.name} {alg.cost:.2f}",
            )
        )
        labels.append(f"{alg.name} {alg.cost:.2f}")

    plt.legend(handles=handles, labels=labels)
    plt.show()
