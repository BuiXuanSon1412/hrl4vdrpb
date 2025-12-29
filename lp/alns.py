"""

import numpy as np
import pandas as pd
import os
import json
import random
import copy

data_path = "E:/bkai/VRPB/hrl4vdrpbtw/data/generated/data/N10/S101_N10_C_3G_R50.json"
with open(data_path, 'r') as f:
    data = json.load(f)

config = data['Config']
nodes = data['Nodes']

num_customers = config['General']['NUM_CUSTOMERS']
num_nodes = config['General']['NUM_NODES']
MAX_COORD = config['General']['MAX_COORD_KM']
T_max = config['General']['T_MAX_SYSTEM_H']

V_TRUCK = config['Vehicles']['V_TRUCK_KM_H']
V_DRONE = config['Vehicles']['V_DRONE_KM_H']
Q = config['Vehicles']['CAPACITY_TRUCK']
Q_tilde = config['Vehicles']['CAPACITY_DRONE']
num_vehicles = config['Vehicles']['NUM_TRUCKS']
num_drones = config['Vehicles']['NUM_DRONES']

tau_l = config['Vehicles']['DRONE_TAKEOFF_MIN'] / 60.0
tau_r = config['Vehicles']['DRONE_LANDING_MIN'] / 60.0
service_time = config['Vehicles']['SERVICE_TIME_MIN'] / 60.0

depot_info = config['Depot']
depot_idx = depot_info['id']
depot_coord = np.array(depot_info['coord'])
depot_tw = depot_info['time_window_h']

coords = [depot_coord]
demands = {depot_idx: 0}
time_windows = {depot_idx: depot_tw}
service_times = {depot_idx: 0}

linehaul_indices = []
backhaul_indices = []

for node in nodes:
    node_id = node['id']
    coords.append(np.array(node['coord']))
    demands[node_id] = node['demand']
    time_windows[node_id] = node['tw_h']
    service_times[node_id] = service_time

    if node['type'] == 'LINEHAUL':
        linehaul_indices.append(node_id)
    else:
        backhaul_indices.append(node_id)

coords = np.array(coords)

L = linehaul_indices
B = backhaul_indices
C = L + B
N = [depot_idx] + C

n_nodes = len(coords)
d = np.zeros((n_nodes, n_nodes))
d_tilde = np.zeros((n_nodes, n_nodes))

for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            d[i, j] = np.linalg.norm(coords[i] - coords[j], ord=1)
            d_tilde[i, j] = np.linalg.norm(coords[i] - coords[j], ord=2)

t = d / V_TRUCK
t_tilde = d_tilde / V_DRONE

q = demands
t_start = {i: time_windows[i][0] for i in range(n_nodes)}
t_end = {i: time_windows[i][1] for i in range(n_nodes)}
s = service_times

K = list(range(1, num_vehicles + 1))
num_drone_routes = num_drones
R = list(range(1, num_drone_routes + 1))

c = 1.0
c_tilde = 0.2
M = 10000.0

w1 = 1.0
w2 = 0.2

# def check_time_windows():


# def evaluate_solution():

#
def create_initial_solution():
    solution = {
        'truck_routes': {},
        'drone_trips': {},
        'cost': 0.0,
        'spanning_time': 0.0,
        'feasible': True
    }

    for k in K:
        solution['truck_routes'][k] = [0, 0]
        solution['drone_trips'][k] = {}
        for r in R:
            solution['drone_trips'][k][r] = {'launch': None, 'customers': [], 'land': None}

    unserved = C.copy()
    current_vehicle = 1

    while unserved:
        route = solution['truck_routes'][current_vehicle]
        current_node = 0
        current_load = 0

        while True:
            best_customer = None
            best_distance = float('inf')

            for customer in unserved:
                if current_load + q[customer] > Q:
                    continue

                temp_route = route[:-1] + [customer, 0]
                if check_time_windows(temp_route):
                    dist = d[current_node][customer]
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer

            if best_customer is None:
                break

            route.insert(-1, best_customer)
            unserved.remove(best_customer)
            current_node = best_customer
            current_load += q[best_customer]

        current_vehicle += 1
        if current_vehicle > num_vehicles:
            current_vehicle = 1
            if unserved:
                for customer in list(unserved):
                    solution['truck_routes'][1].insert(-1, customer)
                    unserved.remove(customer)
                break

    evaluate_solution(solution)
    return solution

# xóa ngẫu nhiên khác
def random_removal(solution, num_remove):
    new_solution = copy.deepcopy(solution)

    all_customers = []
    for k, route in new_solution['truck_routes'].items():
        for node in route[1:-1]:
            all_customers.append((k, 'truck', node))

    for k, trips in new_solution['drone_trips'].items():
        for r, trip in trips.items():
            for customer in trip['customers']:
                all_customers.append((k, 'drone', r, customer))

    num_remove = min(num_remove, len(all_customers))
    removed = random.sample(all_customers, num_remove)
    removed_customers = []

    for item in removed:
        if item[1] == 'truck':
            k, _, node = item
            new_solution['truck_routes'][k].remove(node)
            removed_customers.append(node)
        else:
            k, _, r, customer = item
            new_solution['drone_trips'][k][r]['customers'].remove(customer)
            removed_customers.append(customer)

    return new_solution, removed_customers

# xóa khách có chi phí cao nhất
def worst_removal(solution, num_remove):
    new_solution = copy.deepcopy(solution)

    customer_costs = []

    for k, route in new_solution['truck_routes'].items():
        for i in range(1, len(route) - 1):
            node = route[i]
            cost_contribution = d[route[i-1]][node] + d[node][route[i+1]] - d[route[i-1]][route[i+1]]
            customer_costs.append((cost_contribution, k, 'truck', node))

    for k, trips in new_solution['drone_trips'].items():
        for r, trip in trips.items():
            for customer in trip['customers']:
                cost_contribution = d_tilde[trip['launch']][customer] + d_tilde[customer][trip['land']]
                customer_costs.append((cost_contribution, k, 'drone', r, customer))

    customer_costs.sort(reverse=True)

    num_remove = min(num_remove, len(customer_costs))
    removed_customers = []

    for i in range(num_remove):
        item = customer_costs[i]
        if item[2] == 'truck':
            _, k, _, node = item
            new_solution['truck_routes'][k].remove(node)
            removed_customers.append(node)
        else:
            _, k, _, r, customer = item
            new_solution['drone_trips'][k][r]['customers'].remove(customer)
            removed_customers.append(customer)

    return new_solution, removed_customers

# xóa các khách tương tự nhau
def shaw_removal(solution, num_remove):
    new_solution = copy.deepcopy(solution)

    all_customers = []
    for k, route in new_solution['truck_routes'].items():
        for node in route[1:-1]:
            all_customers.append(node)

    for k, trips in new_solution['drone_trips'].items():
        for r, trip in trips.items():
            all_customers.extend(trip['customers'])

    if not all_customers:
        return new_solution, []

    seed = random.choice(all_customers)
    removed_customers = [seed]

    relatedness = []
    for customer in all_customers:
        if customer == seed:
            continue

        dist_related = d[seed][customer]

        time_related = abs(t_start[seed] - t_start[customer])

        demand_related = abs(q[seed] - q[customer])

        related = dist_related + time_related + demand_related * 0.1
        relatedness.append((related, customer))

    relatedness.sort()

    num_remove = min(num_remove, len(relatedness) + 1)
    for i in range(num_remove - 1):
        removed_customers.append(relatedness[i][1])

    for customer in removed_customers:
        for k, route in new_solution['truck_routes'].items():
            if customer in route:
                route.remove(customer)

        for k, trips in new_solution['drone_trips'].items():
            for r, trip in trips.items():
                if customer in trip['customers']:
                    trip['customers'].remove(customer)

    return new_solution, removed_customers

# Xóa cả route
def route_removal(solution, num_routes):
    new_solution = copy.deepcopy(solution)

    routes_to_remove = random.sample(list(K), min(num_routes, len(K)))
    removed_customers = []

    for k in routes_to_remove:
        route = new_solution['truck_routes'][k]
        for node in route[1:-1]:
            removed_customers.append(node)
        new_solution['truck_routes'][k] = [0, 0]

        for r in R:
            removed_customers.extend(new_solution['drone_trips'][k][r]['customers'])
            new_solution['drone_trips'][k][r] = {'launch': None, 'customers': [], 'land': None}

    return new_solution, removed_customers

"""

