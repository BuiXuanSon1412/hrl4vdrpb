import pulp as pl
import numpy as np
import pandas as pd
import os
import sys
import json


# data_path = "E:/bkai/VRPB/hrl4vdrpbtw/data/benchmark/vrpbtw-solomon-100-derived/n15/c101-n15-b4-k25.csv"

# df = pd.read_csv(data_path)

# print("Các cột trong CSV:")
# print(df.columns.tolist())
# print(df.head())

# # Depot là node đầu tiên (ID = 0)
# depot_idx = 0

# linehaul_indices = df[df['DEMAND'] > 0].index.tolist()

# backhaul_indices = df[df['DEMAND'] < 0].index.tolist()

# L = linehaul_indices  
# B = backhaul_indices  
# C = L + B             
# N = [depot_idx] + C   

# # Tạo ma trận khoảng cách từ XCOORD và YCOORD
# n_nodes = len(df)
# d = np.zeros((n_nodes, n_nodes))
# d_tilde = np.zeros((n_nodes, n_nodes))

# for i in range(n_nodes):
#     for j in range(n_nodes):
#         if i != j:
#             xi, yi = df.loc[i, 'XCOORD.'], df.loc[i, 'YCOORD.']
#             xj, yj = df.loc[j, 'XCOORD.'], df.loc[j, 'YCOORD.']
#             d[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
#             d_tilde[i, j] = d[i, j] * 1.2  

# # Thời gian di chuyển
# vehicle_speed = 1.0  
# drone_speed = 1.5    
# t = d / vehicle_speed
# t_tilde = d_tilde / drone_speed

# q = df['DEMAND'].to_dict()

# s = df['SERVICE TIME'].to_dict()

# t_start = df['READY TIME'].to_dict()
# t_end = df['DUE DATE'].to_dict()

# T_max = df.loc[depot_idx, 'DUE DATE']

# Q = 200.0          
# Q_tilde = 50.0     

# tau_l = 5.0        
# tau_r = 5.0        

# c = 1.0            
# c_tilde = 0.2      
# M = 10000.0        

# w1 = 1.0
# w2 = 0.2

# num_customers = len(C)
# num_vehicles = max(2, min(5, num_customers // 3 + 1))
# num_drone_routes = max(1, min(3, num_customers // 5 + 1))

# K = list(range(1, num_vehicles + 1))  
# R = list(range(1, num_drone_routes + 1))  

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

model = pl.LpProblem('VRPBD', pl.LpMinimize)

x = pl.LpVariable.dicts('x', ((k, i) for k in K for i in N), cat='Binary')
y = pl.LpVariable.dicts('y', ((k, i, j) for k in K for i in N for j in N), cat='Binary')
z = pl.LpVariable.dicts('z', ((k, i) for k in K for i in N), lowBound=0, cat='Integer')
p = pl.LpVariable.dicts('p', ((k, i) for k in K for i in N), lowBound=0, cat='Continuous')
a = pl.LpVariable.dicts('a', ((k, i) for k in K for i in N), lowBound=0, cat='Continuous')
b = pl.LpVariable.dicts('b', ((k, i) for k in K for i in N), lowBound=0, cat='Continuous')

x_tilde = pl.LpVariable.dicts('x_tilde', ((k, r, i) for k in K for r in R for i in N), cat='Binary')
y_tilde = pl.LpVariable.dicts('y_tilde', ((k, r, i, j) for k in K for r in R for i in N for j in N), cat='Binary')
z_tilde = pl.LpVariable.dicts('z_tilde', ((k, r, i) for k in K for r in R for i in N), lowBound=0, cat='Integer')
lambda_var = pl.LpVariable.dicts('lambda', ((k, r, i) for k in K for r in R for i in N), cat='Binary')
varrho = pl.LpVariable.dicts('varrho', ((k, r, i) for k in K for r in R for i in N), cat='Binary')
p_tilde = pl.LpVariable.dicts('p_tilde', ((k, r, i) for k in K for r in R for i in N), lowBound=0, cat='Continuous')
a_tilde = pl.LpVariable.dicts('a_tilde', ((k, i) for k in K for i in N), lowBound=0, cat='Continuous')
b_tilde = pl.LpVariable.dicts('b_tilde', ((k, i) for k in K for i in N), lowBound=0, cat='Continuous')
h = pl.LpVariable.dicts('h', ((k, r, i) for k in K for r in R for i in N), lowBound=0, cat='Continuous')

Z_lambda = pl.LpVariable.dicts('Z_lambda', ((k, r, i) for k in K for r in R for i in N), lowBound=0, cat='Continuous')
Z_varrho = pl.LpVariable.dicts('Z_varrho', ((k, r, i) for k in K for r in R for i in N), lowBound=0, cat='Continuous')

xi = pl.LpVariable.dicts('xi', (i for i in N), lowBound=0, cat='Continuous')

spanning_time = pl.LpVariable('spanning_time', lowBound=0, cat='Continuous')

cost = pl.lpSum([y[k, i, j] * c * d[i][j] for k in K for i in N for j in N if i != j]) + \
       pl.lpSum([y_tilde[k, r, i, j] * c_tilde * d_tilde[i][j] for k in K for r in R for i in N for j in N])

model += w1 * cost + w2 * spanning_time

for i in C:
    model += spanning_time >= xi[i] + s[i] - t_end[i], f'spanning_{i}'

# 3
for i in C:
    model += pl.lpSum([x[k, i] for k in K]) + pl.lpSum([x_tilde[k, r, i] for k in K for r in R]) == 1

# 4-5
for k in K:
    model += pl.lpSum([y[k, 0, j] for j in C]) == 1
    model += pl.lpSum([y[k, i, 0] for i in C]) == 1

for k in K:
    for i in C:
        model += pl.lpSum([y[k, j, i] for j in N if j != i]) == x[k, i]
        model += pl.lpSum([y[k, i, j] for j in N if j != i]) == x[k, i]

# 6-8
for k in K:
    for r in R:
        model += pl.lpSum([lambda_var[k, r, i] for i in N]) <= 1
        model += pl.lpSum([varrho[k, r, j] for j in N]) <= 1
        model += pl.lpSum([lambda_var[k, r, i] for i in N]) == pl.lpSum([x_tilde[k, r, i] for i in C])
        model += pl.lpSum([varrho[k, r, j] for j in N]) == pl.lpSum([x_tilde[k, r, i] for i in C])

for k in K:
    for r in R:
        for i in C:  
            model += lambda_var[k, r, i] <= x[k, i]
            model += varrho[k, r, i] <= x[k, i]


# 9
for k in K:
    for r in R:
        for i in N:
            for j in N:
                if i != j:
                    model += z_tilde[k, r, i] <= z_tilde[k, r, j] + M * (1 - varrho[k, r, j])

# 10
for k in K:
    for r in R:
        for i in N:
            model += pl.lpSum(y_tilde[k, r, j, i] for j in N if j != i)  - x_tilde[k, r, i] <= M * (lambda_var[k, r, i] + varrho[k, r, i])
            model += pl.lpSum(y_tilde[k, r, j, i] for j in N if j != i)  - x_tilde[k, r, i] >= -M * (lambda_var[k, r, i] + varrho[k, r, i])

            model += pl.lpSum(y_tilde[k, r, i, j] for j in N if j != i)  - x_tilde[k, r, i] <= M * (lambda_var[k, r, i] + varrho[k, r, i])
            model += pl.lpSum(y_tilde[k, r, i, j] for j in N if j != i)  - x_tilde[k, r, i] >= -M * (lambda_var[k, r, i] + varrho[k, r, i])

# 11-12
for k in K:
    for r in R: 
        for i in C:
            model += lambda_var[k, r, i] <= pl.lpSum([y_tilde[k, r, i, j] for j in N if j != i])
            model += lambda_var[k, r, i] >= x[k, i] + pl.lpSum([y_tilde[k, r, i, j] for j in N if j != i]) - 1


# 13-14
for k in K:
    for r in R: 
        for i in C:
            model += varrho[k, r, i] <= pl.lpSum([y_tilde[k, r, j, i] for j in N if j != i])
            model += varrho[k, r, i] >= x[k, i] + pl.lpSum([y_tilde[k, r, j, i] for j in N if j != i]) - 1

# 15-16
for k in K:
    for i in C:
        for j in C:
            if i != j:
                model += (z[k, i] - z[k, j] + 1 <= M * (1 - y[k, i, j]))

for k in K:
    for r in R:
        for i in C:
            for j in C:
                if i != j:
                    model += (z_tilde[k, r, i] - z_tilde[k, r, j] + 1 <= M * (1 - y_tilde[k, r, i, j]))

# 17
for k in K:
    initial_load = pl.lpSum([q[u] * x[k, u] for u in L]) + pl.lpSum([q[u] * x_tilde[k, r, u] for u in L for r in R]) 
    model += p[k, 0] == initial_load

# 18-19
epsilon = 0.01
for k in K:
    for i in N:
        for j in C:
            if i != j:
                load_change = - q[j] - pl.lpSum([Z_lambda[k, r, j] for r in R]) + pl.lpSum([Z_varrho[k, r, j] for r in R])
                model += p[k, j] <= p[k, i] + load_change + M * (1 - y[k, i, j]) + epsilon
                model += p[k, j] >= p[k, i] + load_change - M * (1 - y[k, i, j]) - epsilon

# 20-30
for k in K:
    for i in N:
        model += p[k, i] <= Q
        model += p[k, i] >= 0

for k in K:
    for r in R:
        for j in N:
            model += Z_lambda[k, r, j] <= p_tilde[k, r, j] + Q_tilde * (1 - lambda_var[k, r, j])
            model += Z_lambda[k, r, j] <= Q_tilde * lambda_var[k, r, j]
            model += Z_lambda[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - lambda_var[k, r, j])
            
            model += Z_varrho[k, r, j] <= p_tilde[k, r, j] + Q_tilde * (1 - varrho[k, r, j])
            model += Z_varrho[k, r, j] <= Q_tilde * varrho[k, r, j]
            model += Z_varrho[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - varrho[k, r, j])

# 31-32
for k in K:
    for r in R:
        drone_pickup = pl.lpSum([q[u] * x_tilde[k, r, u] for u in L])
        for i in N:
            model += p_tilde[k, r, i] <= drone_pickup + M * (1 - lambda_var[k, r, i])
            model += p_tilde[k, r, i] >= drone_pickup - M * (1 - lambda_var[k, r, i])

# 33-34
for k in K:
    for r in R:
        for j in N:
            model += p_tilde[k, r, j] <= M * (1 - varrho[k, r, j])

# 35-36
for k in K:
    for r in R:
        for i in N:
            for j in C:
                model += p_tilde[k, r, j] <= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] + M * (1 - y_tilde[k, r, i, j])
                model += p_tilde[k, r, j] >= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] - M * (1 - y_tilde[k, r, i, j])

# 37
for k in K:
    for r in R:
        for i in N:
            model += p_tilde[k, r, i] <= Q_tilde

# 38-40
for i in C:
    model += xi[i] >= t_start[i]

for k in K:
    for i in C: 
        model += xi[i] >= a[k, i] - M * (1 - x[k, i])

for k in K:
    for r in R:
        for i in C:
            model += xi[i] >= a_tilde[k, i] + tau_r - M * (1 - x_tilde[k, r, i] )

# 41
for k in K:
    for i in N:
        for j in C:
            if i != j:
                model += a[k, j] >= b[k, i] + t[i][j] - M * (1 - y[k, i, j]) 
                model += a[k, j] <= b[k, i] + t[i][j] + M * (1 - y[k, i, j])

# 42
for k in K:
    for r in R:
        for i in N:
            for j in C:
                model += a_tilde[k, j] >= b_tilde[k, i] + tau_l + t_tilde[i][j] - M * (1 - y_tilde[k, r, i, j])
                model += a_tilde[k, j] <= b_tilde[k, i] + tau_l + t_tilde[i][j] + M * (1 - y_tilde[k, r, i, j])

# 43-45
for k in K:
    for i in C:
        model += b[k, i] >= xi[i] + s[i] # - M * (1 - x[k, i])

for k in K:
    for r in R:
        for i in C:
            model += b[k, i] >= b_tilde[k, i] + tau_l - M * (1 - lambda_var[k, r, i])
            model += b[k, i] >= a_tilde[k, i] + tau_r - M * (1 - varrho[k, r, i])

# 46
for k in K:
    for r in R:
        for i in C:
            model += b_tilde[k, i] >= xi[i] + s[i] - M * (1 - x_tilde[k, r, i]) - M * lambda_var[k, r, i] - M * varrho[k, r, i]

# 47-48
for k in K:
    for r in R:
        for i in C:
            model += b_tilde[k, i] >= a[k, i] - M * (1 - lambda_var[k, r, i])
            model += b_tilde[k, i] <= b[k, i] + M * (1 - lambda_var[k, r, i])

# 49-50
for k in K:
    for r in R:
        for j in N:
            model += a_tilde[k, j] + h[k, r, j] >= a[k, j] - M * (1 - varrho[k, r, j])
            model += a_tilde[k, j] + h[k, r, j] + tau_r <= b[k, j] + M * (1 - varrho[k, r, j])

# 51-54
for k in K:
    for r in R:
        for i in N:
            model += h[k, r, i] >= a[k, i] - a_tilde[k, i] - M * (1 - varrho[k, r, i])
            model += h[k, r, i] >= xi[i] - a_tilde[k, i] - M * (1 - x_tilde[k, r, i])
            model += h[k, r, i] <= xi[i] - a_tilde[k, i] + M * (1 - x_tilde[k, r, i])
            model += h[k, r, i] >= 0

# 55
for k in K:
    for r in R:
        flight_time = pl.lpSum([y_tilde[k, r, i, j] * t_tilde[i][j] for i in C for j in C if i != j])
        launch_time = pl.lpSum([lambda_var[k, r, i] * tau_l for i in C])
        land_time = pl.lpSum([varrho[k, r, j] * tau_r for j in C])
        service_time = pl.lpSum([x_tilde[k, r, i] * s[i] for i in C])
        wait_time = pl.lpSum([h[k, r, i] for i in C])
        
        total_time = flight_time + launch_time + land_time + service_time + wait_time
        model += total_time <= T_max

# 56
for k in K:
    for r in R[:-1]:
        trip_r_active = pl.lpSum([x_tilde[k, r, i] for i in C])
        trip_r_next_active = pl.lpSum([x_tilde[k, r+1, i] for i in C])
        model += trip_r_next_active <= trip_r_active * len(C)
        
for k in K:
    for r in R[:-1]:
        for i in N:
            for j in N:
                model += z[k, j] >= z[k, i] - M * (2 - varrho[k, r, i] - lambda_var[k, r+1, j])
                
        for i in N:
            for j in N:
                model += a_tilde[k, j] >= b_tilde[k, i] + tau_l - M * (2 - varrho[k, r, i] - lambda_var[k, r+1, j])


print(f"Model created with {len(model.variables())} variables and {len(model.constraints)} constraints")
print("\nSolving the model")

solver = pl.PULP_CBC_CMD(timeLimit=36000, gapRel=0.05, msg=1)

model.solve(solver)

if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
    print("\n")
    print("SOLUTION FOUND!")

    print(f"\nObjective value: {pl.value(model.objective):.2f}")
    print(f"Total cost: {pl.value(cost):.2f}")
    print(f"Spanning time: {pl.value(spanning_time):.2f}")
    print(f"Solution status: {pl.LpStatus[model.status]}")
    
    print("\n")
    print("DETAILED SOLUTION:")

    for k in K:
        route = []
        current = 0
        visited = set([0])
        route.append(0)
        
        max_iter = len(N) * 2
        iter_count = 0
        while iter_count < max_iter:
            found = False
            for j in N:
                if j not in visited and pl.value(y[k, current, j]) > 0.5:
                    route.append(j)
                    visited.add(j)
                    current = j
                    found = True
                    break
            if not found:
                if current != 0 and pl.value(y[k, current, 0]) > 0.5:
                    route.append(0)
                break
            iter_count += 1
        
        if len(route) > 1:
            truck_serves = [i for i in C if pl.value(x[k, i]) > 0.5]
            
            print(f"\n")
            print(f"VEHICLE {k}:")
            print(f"Route: {' -> '.join(map(str, route))}")
            print(f"Truck serves: {truck_serves}")
            
            has_drone = False
            drone_ended = False

            for r in R:
                if drone_ended:
                    break
                served = [i for i in C if pl.value(x_tilde[k, r, i]) > 0.5]
                if served:
                    if not has_drone:
                        print(f"\nDrone {k} trips:")
                        has_drone = True
                    
                    launch = [i for i in N if pl.value(lambda_var[k, r, i]) > 0.5]
                    land = [i for i in N if pl.value(varrho[k, r, i]) > 0.5]
                    
                    launch_node = launch[0] if launch else 'N/A'
                    land_node = land[0] if land else 'N/A'

                    if land_node == 0:
                        drone_ended = True
                    
                    drone_route = []
                    drone_current = launch_node
                    drone_visited = set()
                    
                    if launch_node != 'N/A':
                        for _ in range(len(N)):
                            drone_route.append(drone_current)
                            drone_visited.add(drone_current)
                            
                            found_next = False
                            for j in N:
                                if j not in drone_visited and pl.value(y_tilde[k, r, drone_current, j]) > 0.5:
                                    drone_current = j
                                    found_next = True
                                    break
                            if not found_next:
                                break
                        
                        if land_node != 'N/A' and land_node not in drone_route:
                            drone_route.append(land_node)
                    
                    drone_route_str = ' -> '.join(map(str, drone_route)) if drone_route else 'N/A'
                    
                    print(f"  Trip {r}: departs at node {launch_node} -> "
                          f"serves {served} -> returns to node {land_node}")
                    if drone_route:
                        print(f"    Full route: {drone_route_str}")
            
            if not has_drone:
                print(f"\nDrone {k} trips: None")
    
    print("\n")
                
else:
    print("\n")
    print("NO SOLUTION FOUND!")
    print(f"Status: {pl.LpStatus[model.status]}")