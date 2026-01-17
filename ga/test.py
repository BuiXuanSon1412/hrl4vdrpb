from utils import (
    decode,
    init_population,
    repair,
)
from problem import Problem
import random


if __name__ == "__main__":
    """
    dir = Path("../data/generated/data/N50/")
    count = 0
    for file in dir.rglob("S*.json"):
        if file.is_file():
            res = test_repair(file)
            count += res

    print("overload case: ", count)
    """
    random.seed(42)
    data_path = "../data/generated/data/N10/S042_N10_C_R50.json"
    problem = Problem(data_path=data_path)

    init_popu = init_population(5, problem)

    print("DECODED: ")
    for idx, indi in enumerate(init_popu):
        repair(indi.chromosome, problem)
        tardiness_solution, cost_solution = decode(indi, problem)

        print("Tardines Solution ", idx)
        for idx, (route, trips) in enumerate(tardiness_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)

        print("Cost Solution ", idx)
        for idx, (route, trips) in enumerate(cost_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)
