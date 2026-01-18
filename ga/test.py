from math import cos
from utils import (
    cal_cost,
    cal_tardiness,
    decode,
    init_population,
    repair,
)
from problem import Problem

"""
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
        print(f"Tardiness Solution: {cal_tardiness(tardiness_solution, problem):.2f}")
        for idx, (route, trips) in enumerate(tardiness_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)

        print(f"Cost Solution: {cal_cost(cost_solution, problem):.2f}")
        for idx, (route, trips) in enumerate(cost_solution.routes):
            print("Fleet ", idx)
            print(route)
            print(trips)

        print("\n\n")

"""
if __name__ == "__main__":
    import json
    import numpy as np
    from pymoo.indicators.hv import HV

    # 1. Load your result file
    with open("./result/PFG_MOEA/N50/S046_N50_R_R50.json", "r") as f:
        data = json.load(f)

    history = data["history"]

    # 2. Define your reference point
    # (Must be worse than the max values in your data for minimization)
    ref_point = np.array([50.0, 80000.0])

    # 3. Initialize HV calculator
    ind = HV(ref_point=ref_point)

    # 4. Calculate HV for each generation
    hv_progress = {}
    for gen, points in history.items():
        # Convert points to a numpy array
        points_array = np.array(points)

        # Calculate HV for the current generation
        # Note: Ensure these are the non-dominated points (Pareto Front)
        current_hv = ind(points_array)
        hv_progress[int(gen)] = current_hv

    # 5. Output results
    for gen in sorted(hv_progress.keys()):
        print(f"Generation {gen}: Hypervolume = {hv_progress[gen]:.4f}")
