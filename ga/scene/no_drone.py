import random
from population import Individual


def init_population(num_indi, seed, problem):
    random.seed(seed)
    popu = []
    for _ in range(num_indi):
        chromosome = []

        num_fleet = problem.num_fleet
        num_nodes = len(problem.nodes) - 1

        chromosome = []

        population = list(range(1, num_nodes + num_fleet))
        nodes = random.sample(population, len(population))

        mask = [0 for _ in range(num_nodes + num_fleet - 1)]
        chromosome = [nodes, mask]

        indi = Individual(chromosome)
        popu.append(indi)

    return popu
