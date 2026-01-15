from typing import List, Optional, Tuple

from problem import VRPBTWProblem, Solution, Route
import numpy as np
from collections import deque


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = float("inf")


def calculate_fitness(solution, problem, weights):
    return float("inf")


def init_population(num_indi, problem):
    popu = []
    for _ in range(num_indi):
        chromosome = []

        num_fleet = problem.num_fleet
        num_nodes = len(problem.nodes) - 1

        chromosome = []
        nodes = np.random.permutation(np.arange(1, num_nodes + num_fleet))
        mask = np.random.choice(
            [0, 1, -1], size=(num_nodes + num_fleet - 1), p=[0.7, 0.15, 0.15]
        )
        chromosome = np.array([nodes, mask])

        indi = Individual(chromosome)
        popu.append(indi)

    return popu


def partition(chro, problem):
    delimiter = chro[0] >= len(problem.nodes)

    delimiter_idx = np.where(delimiter)[0] + 1

    seqs = np.split(chro, delimiter_idx, axis=1)

    seqs = [seq[:, :-1] for seq in seqs[:-1]] + [seqs[-1]]

    return seqs


def unset(chro, problem: VRPBTWProblem):
    demands = np.array([abs(node.demand) for node in problem.nodes])
    is_node = chro[0] < len(problem.nodes)

    chro_demands = np.zeros(chro.shape[1])
    node_indices = chro[0, is_node]
    chro_demands[is_node] = demands[node_indices]

    overload = (is_node) & (chro_demands > problem.drone_capacity)
    chro[1, overload] = 0


def overload(chro, problem: VRPBTWProblem):
    demands = np.array([abs(node.demand) for node in problem.nodes])
    genes = chro.T.tolist()
    load = 0
    for gene in genes:
        if gene[0] >= len(problem.nodes):
            load = 0
            if load > problem.truck_capacity:
                return True
        else:
            load += demands[gene[0]]

    return False


def calculate_route_distance(route, problem: VRPBTWProblem):
    distance = 0
    for i in range(1, len(route)):
        distance += problem.distance_matrix[route[i]][route[-1]]

    return distance


def dronable(chro, problem: VRPBTWProblem):
    # unset mask layer if its node capacity overload the drone capacity itself
    unset(chro, problem)

    genes = chro.T.tolist()
    # two adjacent drone nodes must have same sign
    for i in range(1, len(genes)):
        if (
            not genes[i][1]
            or genes[i][0] >= len(problem.nodes)
            or genes[i - 1][0] >= len(problem.nodes)
        ):
            continue
        if genes[i - 1][1] + genes[i][1] == 0:
            genes[i][1] = genes[i - 1][1]

    # re-distribute nodes into trips
    trip = []
    demands = np.array([node.demand for node in problem.nodes])
    for i in range(len(genes)):
        if genes[i][0] >= len(problem.nodes):
            trip.clear()

        elif genes[i][1]:
            if not trip or trip[-1][1] + genes[i][1] == 0:
                trip.clear()
                trip.append(genes[i])
            else:
                # verify sum of interior edges
                temp_trip = trip + [genes[i]]
                temp_in_distance = calculate_route_distance(
                    [node[0] for node in temp_trip], problem
                )

                # verify drone load
                overload = False
                load = sum(
                    [demands[node[0]] for node in temp_trip if demands[node[0]] > 0]
                )
                if load > problem.drone_capacity:
                    overload = True
                else:
                    for node in temp_trip:
                        load -= demands[node[0]]
                        if load > problem.drone_capacity:
                            overload = True
                            break

                # violation
                if (
                    temp_in_distance > problem.drone_speed * problem.drone_trip_duration
                    or overload
                ):
                    genes[i][1] = 0
                    for j in range(i + 1, len(genes)):
                        genes[j][1] = genes[j][1] * (-1)
                    trip.clear()
                trip.append(genes[i])

    return np.array(genes).T


def balance(chro, problem: VRPBTWProblem, deviation=0.0):
    demands = np.array([abs(node.demand) for node in problem.nodes])

    genes = chro.T.tolist()
    delimiter_queue = deque()
    node_queue = deque()

    for gene in genes:
        if gene[0] >= len(problem.nodes):
            delimiter_queue.append(gene)
        else:
            node_queue.append(gene)

    avg_count = (len(problem.nodes) - 1) / problem.num_fleet
    load = 0
    count = 0
    new_genes = []
    while node_queue:
        gene = node_queue.popleft()
        if load + demands[gene[0]] <= problem.truck_capacity and count < avg_count:
            load += demands[gene[0]]
            count = count + 1
        else:
            new_genes.append(delimiter_queue.popleft())
            load = 0
            count = 0
        new_genes.append(gene)

    return np.array(new_genes).T


def gemini_schedule(
    route, trips, problem: "VRPBTWProblem"
) -> Tuple[Optional[Route], Optional[List[Route]], float]:
    n_truck = len(route)

    # Initialize Route objects
    t_route = Route(
        nodes=route,
        arrival=[0.0] * n_truck,
        departure=[0.0] * n_truck,
        service=[0.0] * n_truck,
    )

    d_trips = []
    for trip_nodes in trips:
        d_trips.append(
            Route(
                nodes=trip_nodes,
                arrival=[0.0] * 3,
                departure=[0.0] * 3,
                service=[0.0] * 3,
            )
        )

    # --- Forward Pass ---
    for i in range(n_truck):
        curr_node = route[i]

        # 1. Truck Arrival Time
        if i == 0:
            # Cast TW start to float to avoid NumPy type leakage
            t_route.arrival[i] = float(problem.nodes[curr_node].time_window[0])
        else:
            prev_node = route[i - 1]
            dist = problem.distance_matrix[prev_node, curr_node].item()
            travel_time = dist / problem.truck_speed
            t_route.arrival[i] = t_route.departure[i - 1] + travel_time

        # 2. Service Time (Earliest possible start)
        tw_start = float(problem.nodes[curr_node].time_window[0])
        t_route.service[i] = max(t_route.arrival[i], tw_start)

        # 3. Synchronization: Landing Drones
        for trip_idx, trip_nodes in enumerate(trips):
            if trip_nodes[2] == curr_node:
                # Wait for drone + recovery time
                drone_arrival = d_trips[trip_idx].arrival[2]
                t_route.service[i] = max(t_route.service[i], drone_arrival)
                t_route.service[i] += problem.land_time

        # 4. Departure Time
        is_depot = i == 0 or i == n_truck - 1
        # Avoid assigning None; use 0.0 for calculations
        current_service_duration = 0.0 if is_depot else problem.service_time
        t_route.departure[i] = t_route.service[i] + current_service_duration

        # 5. Drone Launch logic
        for trip_idx, trip_nodes in enumerate(trips):
            if trip_nodes[0] == curr_node:
                l_idx, c_idx, arr_idx = trip_nodes

                # Setup launch
                d_launch_start = t_route.service[i] + problem.launch_time
                d_trips[trip_idx].departure[0] = d_launch_start

                # Travel to Customer
                d_dist_to_cust = problem.distance_matrix[l_idx, c_idx].item()
                d_trips[trip_idx].arrival[1] = d_launch_start + (
                    d_dist_to_cust / problem.drone_speed
                )

                # Customer Window
                d_tw_start = float(problem.nodes[c_idx].time_window[0])
                d_trips[trip_idx].service[1] = max(
                    d_trips[trip_idx].arrival[1], d_tw_start
                )
                d_trips[trip_idx].departure[1] = (
                    d_trips[trip_idx].service[1] + problem.service_time
                )

                # Return to Truck
                d_dist_to_land = problem.distance_matrix[c_idx, arr_idx].item()
                d_trips[trip_idx].arrival[2] = d_trips[trip_idx].departure[1] + (
                    d_dist_to_land / problem.drone_speed
                )

                # Ensure truck doesn't leave before launch is complete
                t_route.departure[i] = max(t_route.departure[i], d_launch_start)

    # --- Max Tardiness ---
    total_max_tardiness = 0.0

    # Check Truck Customers
    for i in range(1, n_truck - 1):
        due_date = float(problem.nodes[route[i]].time_window[1])
        tardiness = max(0.0, (t_route.service[i] + problem.service_time) - due_date)
        total_max_tardiness = max(total_max_tardiness, tardiness)

    # Check Drone Customers
    for d_trip in d_trips:
        due_date = float(problem.nodes[d_trip.nodes[1]].time_window[1])
        tardiness = max(0.0, (d_trip.service[1] + problem.service_time) - due_date)
        total_max_tardiness = max(total_max_tardiness, tardiness)

    return t_route, d_trips, total_max_tardiness


def tardiness_route(seq, problem: VRPBTWProblem):
    return None, None


def routing(seq, problem: VRPBTWProblem):
    end_depot = len(problem.nodes)
    nodes = [[0, 0]] + seq.T.tolist() + [[end_depot, 0]]

    # leftmost-right drone node
    lr_drone = [0 for _ in range(len(nodes))]
    temp = 0
    for idx, node in enumerate(reversed(nodes)):
        if node[1]:
            temp = idx
        else:
            lr_drone[idx] = temp

    # separate truck route and drone trips in one sequence
    route = []
    trips = []
    trip = []
    for idx, node in enumerate(nodes):
        if not node[1]:
            route.append(node[0])
        elif node[0] < len(problem.nodes):
            if trip and nodes[trip[-1]][1] != node[1]:
                trips.append(trip)
                trip.clear()
            trip.append(idx)
    if trip:
        trips.append(trip)

    if trips:
        stack = []
        opt_val = float("inf")

        opt_route = None
        opt_trips = None

        # push initial state into stack
        first = trip[0][0]
        last = trip[0][-1]
        for launch in range(0, first):
            for land in range(last + 1, lr_drone[last + 1]):
                temp_trip = [launch] + trip + [land]

                d = calculate_route_distance([nodes[idx][0] for idx in trip], problem)
                if d <= problem.drone_speed * problem.drone_trip_duration:
                    stack.append([[temp_trip]])

        while stack:
            ll_trips = stack.pop()

            if len(ll_trips) == len(trips):
                temp_route, temp_trips, temp_val = gemini_schedule(
                    route, [nodes[idx] for trip in ll_trips for idx in trip], problem
                )
                if opt_val > temp_val:
                    opt_route, opt_trips, opt_val = temp_route, temp_trips, temp_val
                    continue

            last_trip = ll_trips[-1]
            last_land = last_trip[-1]
            next_trip = trips[len(ll_trips)]
            for launch in range(last_land, next_trip[0]):
                for land in range(next_trip[-1], len(nodes)):
                    if nodes[land][1]:
                        break
                    temp_trip = [launch] + next_trip + [land]
                    d = calculate_route_distance(
                        [nodes[idx][0] for idx in temp_trip], problem
                    )
                    if d <= problem.drone_speed * problem.drone_trip_duration:
                        ll_trips.append(temp_trip)
                        stack.append(ll_trips)
    else:
        opt_route, opt_trips, opt_val = gemini_schedule(route, [], problem)
    return opt_route, opt_trips


def decode(indi: Individual, problem: VRPBTWProblem) -> Optional[Solution]:
    chro = indi.chromosome

    seqs = partition(chro, problem)
    routes = []

    for seq in seqs:
        route, trips = routing(seq, problem)
        routes.append((route, trips))

    return Solution(routes)


class Solver:
    def __init__(self, problem: VRPBTWProblem, config):
        self.problem = problem
        self.config = config

    def solve(self, problem):
        pass
