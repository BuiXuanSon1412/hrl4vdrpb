import enum
import numpy as np
from solver import (
    decode,
    dronable,
    init_population,
    partition,
    balance,
    overload,
    unset,
)
from problem import VRPBTWProblem
import random


def test_init_population(num_indi, problem):
    init_popu = init_population(num_indi, problem)

    return init_popu


def test_repair(chro, problem):
    unset(chro, problem)
    if overload(chro, problem):
        chro = balance(chro, problem)

    chro = dronable(chro, problem)

    seqs = partition(chro, problem)
    for seq in seqs:
        print(seq)

    return chro


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
    problem = VRPBTWProblem(data_path=data_path)

    init_popu = test_init_population(5, problem)

    print("INITIALIZED: ")
    for idx, indi in enumerate(init_popu):
        print("individual ", idx)
        print(indi.chromosome)

    print("REPAIRED: ")
    for idx, indi in enumerate(init_popu):
        print("individual ", idx)
        chro = test_repair(indi.chromosome, problem)
        indi.chromosome = chro

    print("DECODED: ")
    for idx, indi in enumerate(init_popu):
        print("Solution ", idx)
        solution = decode(indi, problem)
        if not solution:
            continue
        routes = solution.routes
        for idx, (route, trips) in enumerate(solution.routes):
            print("fleet ", idx)
            print(route)
            print(trips)
