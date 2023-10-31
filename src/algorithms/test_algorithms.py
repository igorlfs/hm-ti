import networkx as nx
import networkx.algorithms.approximation as nx_app
import numpy as np
import pytest
from scipy.spatial import distance

from .algorithms import christofides, twice_around_the_tree


def test_twice_around_the_tree() -> None:
    """Teste o twice around the tree usando o exemplo de Algoritmos II."""
    matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [8, 6, 0, 0, 0],
            [9, 8, 10, 0, 0],
            [12, 9, 11, 7, 0],
        ]
    )
    graph = nx.from_numpy_array(matrix)
    cost, _ = twice_around_the_tree(graph)
    assert cost == 39  # noqa: PLR2004


@pytest.fixture
def _input_graph() -> nx.Graph:
    rng = np.random.default_rng()
    sample = rng.random((10, 2))
    matrix = distance.cdist(sample, sample, "euclidean")
    graph: nx.Graph = nx.from_numpy_array(matrix)
    return graph


def test_christofides(_input_graph: nx.Graph) -> None:
    """Compare o resultado com a implementação do networkx."""
    expected_cycle = nx_app.christofides(_input_graph)
    expected_cost = nx.path_weight(_input_graph, expected_cycle, "weight")
    actual_cost, actual_cycle = christofides(_input_graph)

    assert expected_cost == actual_cost
    assert expected_cycle == actual_cycle
