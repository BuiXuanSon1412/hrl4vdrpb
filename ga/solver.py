from typing import List, Optional

from matplotlib import rc_params
from problem import VRPBTWProblem, Solution, Route
import numpy as np
from collections import deque


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = float("inf")


def init_population(num_indi):
    popu = []
    for _ in range(num_indi):
        chromosome = []
        indi = Individual(chromosome)
        popu.append(indi)

    return popu


def split(chro, problem):
    delimiter = chro[0] >= len(problem.nodes)

    delimiter_idx = np.where(delimiter)[0] + 1

    seq = np.split(chro, delimiter_idx, axis=1)

    seq = [part[:, :-1] for part in seq]

    return seq


def unset(chro, problem: VRPBTWProblem):
    demands = np.array([abs(node.demand) for node in problem.nodes])
    is_node = chro[0] < len(problem.nodes)

    chro_demands = np.zeros(chro.shape[1])
    node_indices = chro[0, is_node]
    chro_demands[is_node] = demands[node_indices]

    overload = (is_node) & (chro_demands > problem.drone_capacity)
    chro[1, overload] = 0


def balance(chro, problem: VRPBTWProblem, deviation=0.0):
    demands = np.array([node.demand for node in problem.nodes])

    genes = chro.T.tolist()
    delimiter_queue = deque()
    node_queue = deque()

    for gene in genes:
        if gene[0] >= len(problem.nodes):
            delimiter_queue.append(gene)
        else:
            node_queue.append(gene)

    avg_count = (len(problem.nodes) - 1) / problem.num_fleet
    bload = 0
    lload = 0
    count = 0
    new_genes = []
    while node_queue:
        gene = node_queue.popleft()
        if demands[gene[0]] > 0:
            if lload + demands[gene[0]] <= problem.truck_capacity and count < avg_count:
                lload += demands[gene[0]]
                count = count + 1
            else:
                new_genes.append(delimiter_queue.popleft())
                lload = 0
                bload = 0
                count = 0
        else:
            if (
                bload + abs(demands[gene[0]]) <= problem.truck_capacity
                and count < avg_count
            ):
                bload += abs(demands[gene[0]])
                count = count + 1
            else:
                new_genes.append(delimiter_queue.popleft())
                lload = 0
                bload = 0
                count = 0
        new_genes.append(gene)

    return np.array(new_genes).T


def route(seq, problem: VRPBTWProblem):
    return None, None


def decode(indi: Individual, problem: VRPBTWProblem) -> Optional[Solution]:
    chro = indi.chromosome

    seqs = split(chro, problem)
    # unset(seqs, problem)
    # balance(seqs, problem)

    # for seq in seqs:
    #    truck_route, drone_trips = route(seq, problem)
    return None


class Solver:
    def __init__(self, problem: VRPBTWProblem, config):
        self.problem = problem
        self.config = config

    def solve(self, problem):
        pass
