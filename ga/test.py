import numpy as np
from solver import split, unset, balance
from problem import VRPBTWProblem


def test_split():
    data_path = "../data/generated/data/N400/S042_N400_C_R50.json"
    problem = VRPBTWProblem(data_path=data_path)
    num_fleet = problem.num_fleet
    num_nodes = len(problem.nodes) - 1

    chro = []
    nodes = np.random.permutation(np.arange(1, num_nodes + num_fleet))
    masks = np.random.choice([0, 1], size=(num_nodes + num_fleet - 1), p=[0.8, 0.2])
    chro = np.array([nodes, masks])

    print("initial:")
    print(chro)

    print("unset:")
    unset(chro, problem)
    print(chro)

    print("balance:")
    chro = balance(chro, problem)
    print(chro)

    print("split:")
    seqs = split(chro, problem)
    for seq in seqs:
        print(seq)


if __name__ == "__main__":
    test_split()
