import numpy as np
from numpy import float64
from numpy.typing import NDArray


def monte_carlo_swap_path(path: list[int], distances: NDArray) -> tuple[list[int], float]:
    """Calcula o novo caminho de `path` a partir da troca de dois vértices."""
    num_cities = len(path)
    rng = np.random.default_rng()

    # i deve ser menor que j
    i = rng.choice(np.arange(num_cities - 1))
    j = rng.choice(np.arange(i + 1, num_cities))

    new_path = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]

    left = (i - 1) % num_cities
    right = (j + 1) % num_cities
    delta: float = (
        distances[path[left], path[j]]
        + (distances[path[i], path[right]])
        - (distances[path[left], path[i]])
        - (distances[path[j], path[right]])
    )

    return new_path, delta


def monte_carlo_step(
    beta: float,
    cost: float,
    path: list[int],
    best_cost: float,
    best_path: list[int],
    distances: NDArray[float64],
) -> tuple[float, list[int], float, list[int]]:
    """Rode um passo de Monte Carlo para o TSP."""
    new_path, delta = monte_carlo_swap_path(path, distances)

    # Se o custo diminuir, a gente sempre troca
    if delta < 0:
        cost += delta
        path = new_path
        if cost < best_cost:
            best_cost = cost
            best_path = path
    # Aplica o critério de Metropolis se o custo aumentar
    elif np.random.random() < np.exp(-beta * delta):
        cost += delta
        path = new_path

    return cost, path, best_cost, best_path


def two_opt(
    tour: list[int], distances: NDArray, i: int, j: int
) -> tuple[float, list[int]]:
    # Nós não precisamos nos preocupar com fazer o módulo
    # pois Python aceita indexação com índices negativos.
    new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
    num_cities = len(tour)
    left = (i - 1) % num_cities
    right = (j + 1) % num_cities
    # Já aqui... É necessário usar o módulo.
    delta: float = (
        distances[tour[left], tour[j]]
        + distances[tour[i], tour[right]]
        - distances[tour[left], tour[i]]
        - distances[tour[j], tour[right]]
    )
    return delta, new_tour


def three_opt(
    tour: list[int], distances: NDArray, i: int, j: int, k: int
) -> tuple[float, list[int]]:
    new_tour = (
        tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 : k + 1][::-1] + tour[k + 1 :]
    )
    num_cities = len(tour)
    i_prev = (i - 1) % num_cities
    j_next = (j + 1) % num_cities
    k_next = (k + 1) % num_cities
    delta = (
        distances[tour[i_prev], tour[j]]
        + distances[tour[i], tour[k]]
        + distances[tour[j_next], tour[k_next]]
        - (
            distances[tour[i_prev], tour[i]]
            + distances[tour[j], tour[j_next]]
            + distances[tour[k], tour[k_next]]
        )
    )
    return delta, new_tour
