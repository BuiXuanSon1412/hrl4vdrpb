from utils import (
    decode,
    init_population,
    repair,
)
from problem import Problem


if __name__ == "__main__":
    data_path = "../data/generated/data/N10/S042_N10_C_R50.json"
    problem = Problem(data_path=data_path)

    init_popu = init_population(5, 41, problem)

    for idx, indi in enumerate(init_popu):
        print("INDIVIDUAL ", idx)
        print("\n===REPAIRED===\n")
        repair(indi.chromosome, problem)
        tardiness_solution, cost_solution = decode(indi, problem)
        print("\n===DECODED===\n")
        print("Tardiness Solution")
        for idx, (route, trips) in enumerate(tardiness_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)

        print("Cost Solution")
        for idx, (route, trips) in enumerate(cost_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)

        print("\n\n")
