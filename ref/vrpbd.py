import pulp as pl
import numpy as np

L = [1, 2, 3, 4]  
B = [5, 6]     
C = L + B      
N = [0] + C    
K = [1, 2]    
R = list(range(1, 3))  

Q = 30.0           
Q_tilde = 20.0      
tau_l = 0.5      
tau_r = 0.5         
c = 1.0             
c_tilde = 0.2      
T_max = 500.0       
M = 10000.0         

np.random.seed(42)
n_nodes = len(N)
d = np.random.uniform(10, 50, (n_nodes, n_nodes))  
d_tilde = np.random.uniform(12, 60, (n_nodes, n_nodes))
np.fill_diagonal(d, 0)
np.fill_diagonal(d_tilde, 0)

t = d / 30.0  
t_tilde = d_tilde / 40.0  

q = {0: 0}
for i in L:
    q[i] = np.random.uniform(8, 12)  
for i in B:
    q[i] = -np.random.uniform(8, 12)  

s = {i: np.random.uniform(2, 5) for i in N}
s[0] = 0  

t_start = {i: 0 for i in N}
t_end = {i: 500.0 for i in N}  

w1 = 1.0   
w2 = 0.1  

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
        model += pl.lpSum([y[k, j, i] for j in N if j != i]) == x[k, i], f'flow_in_k{k}_i{i}'
        model += pl.lpSum([y[k, i, j] for j in N if j != i]) == x[k, i], f'flow_out_k{k}_i{i}'

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

for k in K:
    for r in R:
        model += lambda_var[k, r, 0] <= 1
        model += varrho[k, r, 0] <= 1

# 9
for k in K:
    for r in R:
        for i in N:
            for j in N:
                if i != j:
                    model += z[k, i] + lambda_var[k, r, i] <= z[k, j] + M * (1 - varrho[k, r, j])

# 10-11
for k in K:
    for r in R:
        for i in C:
            model += pl.lpSum([y_tilde[k, r, j, i] for j in N if j != i]) + lambda_var[k, r, i] == x_tilde[k, r, i] + varrho[k, r, i]
            model += pl.lpSum([y_tilde[k, r, i, j] for j in N if j != i]) + varrho[k, r, i] == x_tilde[k, r, i] + lambda_var[k, r, i]

# 12-13
for k in K:
    for i in C:
        for j in C:
            if i != j:
                model += z[k, i] - z[k, j] + 1 <= M * (1 - y[k, i, j])

for k in K:
    for r in R:
        for i in C:
            for j in C:
                if i != j:
                    model += z_tilde[k, r, i] - z_tilde[k, r, j] + 1 <= M * (1 - y_tilde[k, r, i, j])

# 14
for k in K:
    initial_load = pl.lpSum([q[u] * x[k, u] for u in L]) + pl.lpSum([q[u] * x_tilde[k, r, u] for u in L for r in R]) 
    model += p[k, 0] == initial_load

# 15-17
epsilon = 0.01
for k in K:
    for i in N:
        for j in C:
            if i != j:
                load_change = - q[j] - pl.lpSum([Z_lambda[k, r, j] for r in R]) + pl.lpSum([Z_varrho[k, r, j] for r in R])
                model += p[k, j] <= p[k, i] + load_change + M * (1 - y[k, i, j]) + epsilon
                model += p[k, j] >= p[k, i] + load_change - M * (1 - y[k, i, j]) - epsilon

for k in K:
    for i in N:
        model += p[k, i] <= Q
        model += p[k, i] >= 0

# 18-25
for k in K:
    for r in R:
        for j in N:
            model += Z_lambda[k, r, j] <= p_tilde[k, r, j]
            model += Z_lambda[k, r, j] <= Q_tilde * lambda_var[k, r, j]
            model += Z_lambda[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - lambda_var[k, r, j])
            
            model += Z_varrho[k, r, j] <= p_tilde[k, r, j]
            model += Z_varrho[k, r, j] <= Q_tilde * varrho[k, r, j]
            model += Z_varrho[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - varrho[k, r, j])

# 26-27
for k in K:
    for r in R:
        drone_pickup = pl.lpSum([q[u] * x_tilde[k, r, u] for u in L])
        for i in N:
            model += p_tilde[k, r, i] <= drone_pickup + M * (1 - lambda_var[k, r, i])
            model += p_tilde[k, r, i] >= drone_pickup - M * (1 - lambda_var[k, r, i])

# 28-29
for k in K:
    for r in R:
        for j in N:
            model += p_tilde[k, r, j] <= M * (1 - varrho[k, r, j])

# 30-31
for k in K:
    for r in R:
        for i in N:
            for j in C:
                model += p_tilde[k, r, j] <= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] + M * (1 - y_tilde[k, r, i, j])
                model += p_tilde[k, r, j] >= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] - M * (1 - y_tilde[k, r, i, j])

# 32
for k in K:
    for r in R:
        for i in N:
            model += p_tilde[k, r, i] <= Q_tilde

# 33-35
for i in C:
    model += xi[i] >= t_start[i]

for k in K:
    for i in C: 
        model += xi[i] >= a[k, i] - M * (1 - x[k, i])

for k in K:
    for i in C:
        model += xi[i] >= a_tilde[k, i] + tau_l - M * (1 - pl.lpSum([x_tilde[k, r, i] for r in R]))

# 36
for k in K:
    for i in N:
        for j in C:
            if i != j:
                model += a[k, j] >= b[k, i] + t[i][j] - M * (1 - y[k, i, j]) - epsilon

# 37
for k in K:
    for r in R:
        for i in N:
            for j in C:
                model += a_tilde[k, j] >= b_tilde[k, i] + tau_l + t_tilde[i][j] - M * (1 - y_tilde[k, r, i, j])

# 38-40
for k in K:
    for i in C:
        model += b[k, i] >= xi[i] + s[i] - M * (1 - x[k, i])

for k in K:
    for r in R:
        for i in C:
            model += b[k, i] >= b_tilde[k, i] + tau_l - M * (1 - lambda_var[k, r, i])
            model += b[k, i] >= a_tilde[k, i] + tau_r - M * (1 - varrho[k, r, i])

# 41
for k in K:
    for r in R:
        for i in C:
            model += b_tilde[k, i] >= xi[i] + s[i] - M * (1 - x_tilde[k, r, i]) - M * lambda_var[k, r, i] - M * varrho[k, r, i]

# 42-43
for k in K:
    for r in R:
        for i in C:
            model += b_tilde[k, i] >= a[k, i] - M * (1 - lambda_var[k, r, i])
            model += b_tilde[k, i] <= b[k, i] + M * (1 - lambda_var[k, r, i])

# 44-45
for k in K:
    for r in R:
        for j in N:
            model += a_tilde[k, j] + h[k, r, j] >= a[k, j] - M * (1 - varrho[k, r, j])
            model += a_tilde[k, j] + h[k, r, j] + tau_r <= b[k, j] + M * (1 - varrho[k, r, j])

# 46-49
for k in K:
    for r in R:
        for i in N:
            model += h[k, r, i] >= a[k, i] - a_tilde[k, i] - M * (1 - varrho[k, r, i])
            model += h[k, r, i] >= xi[i] - a_tilde[k, i] - M * (1 - x_tilde[k, r, i])
            model += h[k, r, i] <= xi[i] - a_tilde[k, i] + M * (1 - x_tilde[k, r, i])
            model += h[k, r, i] >= 0

# 50
for k in K:
    for r in R:
        flight_time = pl.lpSum([y_tilde[k, r, i, j] * t_tilde[i][j] for i in C for j in C if i != j])
        launch_time = pl.lpSum([lambda_var[k, r, i] * tau_l for i in C])
        land_time = pl.lpSum([varrho[k, r, j] * tau_r for j in C])
        service_time = pl.lpSum([x_tilde[k, r, i] * s[i] for i in C])
        wait_time = pl.lpSum([h[k, r, i] for i in C])
        
        total_time = flight_time + launch_time + land_time + service_time + wait_time
        model += total_time <= T_max

print(f"Model created with {len(model.variables())} variables and {len(model.constraints)} constraints")
print("\nSolving the model")

solver = pl.PULP_CBC_CMD(timeLimit=600, gapRel=0.05, msg=1)

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
            print(f"Route: {' → '.join(map(str, route))}")
            print(f"Truck serves: {truck_serves}")
            
            has_drone = False
            for r in R:
                served = [i for i in C if pl.value(x_tilde[k, r, i]) > 0.5]
                if served:
                    if not has_drone:
                        print(f"\nDrone trips:")
                        has_drone = True
                    
                    launch = [i for i in N if pl.value(lambda_var[k, r, i]) > 0.5]
                    land = [i for i in N if pl.value(varrho[k, r, i]) > 0.5]
                    
                    launch_node = launch[0] if launch else 'N/A'
                    land_node = land[0] if land else 'N/A'
                    
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
                    
                    drone_route_str = ' → '.join(map(str, drone_route)) if drone_route else 'N/A'
                    
                    print(f"  Drone {r}: departs from Vehicle {k} at node {launch_node} → "
                          f"serves {served} → returns to node {land_node}")
                    if drone_route:
                        print(f"    Full route: {drone_route_str}")
            
            if not has_drone:
                print(f"\nDrone trips: None")
    
    print("\n")
    print("DRONE TRIPS:")
    for k in K:
        for r in R:
            served = [i for i in C if pl.value(x_tilde[k, r, i]) > 0.5]
            launch = [i for i in N if pl.value(lambda_var[k, r, i]) > 0.5]
            land = [i for i in N if pl.value(varrho[k, r, i]) > 0.5]
            
            if served:
                print(f"Drone {k}, Trip {r}: Launch at {launch[0] if launch else 'N/A'}, "
                      f"Serve {served}, Land at {land[0] if land else 'N/A'}")

else:
    print("\n")
    print("NO SOLUTION FOUND!")
    print(f"Status: {pl.LpStatus[model.status]}")