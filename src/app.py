from pathlib import Path

import pandas as pd

from .algorithms import get_time_and_cost, row

PROBLEMS_EUC_2D_PATH = "EUC_2D"
PROBLEM_ATT_PATH = "ATT/att48.tsp"


def run() -> None:
    data: list[row] = []
    problems = Path(PROBLEMS_EUC_2D_PATH).glob("*")
    for problem_path in problems:
        get_time_and_cost(problem_path, data)
    get_time_and_cost(Path(PROBLEM_ATT_PATH), data)
    df = pd.DataFrame(data, columns=["Name", "Algorithm", "Time", "Cost"])
    df.to_csv("results.csv", index=False)
