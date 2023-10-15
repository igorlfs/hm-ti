import numpy as np
from numpy import float64, int64
from numpy.typing import NDArray

# Hiperparâmetros do simulated annealing
DECAY_RATIO = 0.999


def monte_carlo_swap_path(path: NDArray[int64]) -> tuple[NDArray[int64], int, int]:
    """Calcula o novo caminho de `path` a partir da troca de dois vértices."""
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
    new_path, beg, end = monte_carlo_swap_path(path)

    # determina a diferença de custo
    left_from_beg = (beg - 1) % num_cities
    right_from_end = (end + 1) % num_cities
    delta = (
        float(dist[new_path[left_from_beg], new_path[beg]])
        + float(dist[new_path[right_from_end], new_path[end]])
        - float(dist[path[left_from_beg], path[beg]])
        - float(dist[path[right_from_end], path[end]])
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

    beta = beta * DECAY_RATIO

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
    delta = (
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
