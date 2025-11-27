import gurobipy as gp
from gurobipy import GRB
import numpy as np

L = [1, 2, 3]  
B = [4, 5]     
C = L + B      
N = [0] + C    
K = [1, 2]    
MaxTrips = 2 
R = list(range(1, MaxTrips + 1))  

Q = 100.0           
Q_tilde = 20.0      
tau_l = 1.0         
tau_r = 1.0         
c = 1.0             
c_tilde = 0.5       
T_max = 200.0       
M = 10000.0         

np.random.seed(42)
n_nodes = len(N)
d = np.random.uniform(10, 50, (n_nodes, n_nodes))  
d_tilde = np.random.uniform(12, 60, (n_nodes, n_nodes))
# d = np.array([[0, 20, 30], [20, 0, 25], [30, 25, 0]])  
# d_tilde = np.array([[0, 25, 35], [25, 0, 30], [35, 30, 0]])
np.fill_diagonal(d, 0)
np.fill_diagonal(d_tilde, 0)


t = d / 30.0  
t_tilde = d_tilde / 40.0  

q = {0: 0}
for i in L:
    q[i] = np.random.uniform(5, 15)  
for i in B:
    q[i] = -np.random.uniform(5, 15)  
# q = {0: 0, 1: 10.0, 2: 15.0}

s = {i: np.random.uniform(2, 5) for i in N}
s[0] = 0  
# s = {0: 0, 1: 2.0, 2: 2.0}

t_start = {i: 0 for i in N}
t_end = {i: 500.0 for i in N}  

w1 = 1.0   
w2 = 0.0   

model = gp.Model('VRPBD')

x = model.addVars(K, N, vtype=GRB.BINARY, name='x')
y = model.addVars(K, N, N, vtype=GRB.BINARY, name='y')
z = model.addVars(K, N, vtype=GRB.INTEGER, lb=0, name='z')
p = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, name='p')
a = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, name='a')
b = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, name='b')

x_tilde = model.addVars(K, R, N, vtype=GRB.BINARY, name='x_tilde')
y_tilde = model.addVars(K, R, N, N, vtype=GRB.BINARY, name='y_tilde')
z_tilde = model.addVars(K, R, N, vtype=GRB.INTEGER, lb=0, name='z_tilde')
lambda_var = model.addVars(K, R, N, vtype=GRB.BINARY, name='lambda')
varrho = model.addVars(K, R, N, vtype=GRB.BINARY, name='varrho')
p_tilde = model.addVars(K, R, N, vtype=GRB.CONTINUOUS, lb=0, name='p_tilde')
a_tilde = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, name='a_tilde')
b_tilde = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0, name='b_tilde')
h = model.addVars(K, R, N, vtype=GRB.CONTINUOUS, lb=0, name='h')

Z_lambda = model.addVars(K, R, N, vtype=GRB.CONTINUOUS, lb=0, name='Z_lambda')
Z_varrho = model.addVars(K, R, N, vtype=GRB.CONTINUOUS, lb=0, name='Z_varrho')

xi = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name='xi')

spanning_time = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='spanning_time')

model.update()

cost = gp.quicksum(y[k, i, j] * c * d[i][j] for k in K for i in N for j in N if i != j) + \
       gp.quicksum(y_tilde[k, r, i, j] * c_tilde * d_tilde[i][j] for k in K for r in R for i in N for j in N)

for i in C:
    model.addConstr(spanning_time >= xi[i] + s[i] - t_end[i], name=f'spanning_{i}')

model.setObjective(w1 * cost + w2 * spanning_time, GRB.MINIMIZE)


# 1 
for i in C:
    model.addConstr(gp.quicksum(x[k, i] for k in K) + gp.quicksum(x_tilde[k, r, i] for k in K for r in R) == 1)

# 2-3
for k in K:
    model.addConstr(gp.quicksum(y[k, 0, j] for j in C) == 1)
    model.addConstr(gp.quicksum(y[k, i, 0] for i in C) == 1)

for k in K:
    for i in C:
        model.addConstr(gp.quicksum(y[k, j, i] for j in N if j != i) == x[k, i])
        model.addConstr(gp.quicksum(y[k, i, j] for j in N if j != i) == x[k, i])

# 4-6
for k in K:
    for r in R:
        model.addConstr(gp.quicksum(lambda_var[k, r, i] for i in N) <= 1)
        model.addConstr(gp.quicksum(varrho[k, r, j] for j in N) <= 1)
        model.addConstr(gp.quicksum(lambda_var[k, r, i] for i in N) == gp.quicksum(varrho[k, r, j] for j in N))

for k in K:
    for r in R:
        for i in N:
            model.addConstr(lambda_var[k, r, i] <= x[k, i])
            model.addConstr(varrho[k, r, i] <= x[k, i])

# 7
for k in K:
    for r in R:
        for i in N:
            for j in N:
                model.addConstr(z[k, i] <= z[k, j] + M * (2 - lambda_var[k, r, i] - varrho[k, r, j]))

# 8-9
for k in K:
    for r in R:
        for i in C:
            model.addConstr(gp.quicksum(y_tilde[k, r, j, i] for j in N if j != i) + lambda_var[k, r, i] 
                == x_tilde[k, r, i] + varrho[k, r, i])
            model.addConstr(gp.quicksum(y_tilde[k, r, i, j] for j in N if j != i) + varrho[k, r, i] 
                == x_tilde[k, r, i] + lambda_var[k, r, i])

# 10-11
for k in K:
    for i in C:
        for j in C:
            if i != j:
                model.addConstr(z[k, i] - z[k, j] + 1 <= M * (1 - y[k, i, j]))

for k in K:
    for r in R:
        for i in C:
            for j in C:
                if i != j:
                    model.addConstr(z_tilde[k, r, i] - z_tilde[k, r, j] + 1 <= M * (1 - y_tilde[k, r, i, j]))

# 12
for k in K:
    model.addConstr(p[k, 0] == gp.quicksum(q[u] * x[k, u] for u in C if q[u] > 0) +
                   gp.quicksum(q[u] * x_tilde[k, r, u] for u in C for r in R if q[u] > 0))

# 13-15
for k in K:
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(p[k, j] <= p[k, i] - q[j] - gp.quicksum(Z_lambda[k, r, j] for r in R) + 
                                gp.quicksum(Z_varrho[k, r, j] for r in R) + M * (1 - y[k, i, j]))
                model.addConstr(p[k, j] >= p[k, i] - q[j] - gp.quicksum(Z_lambda[k, r, j] for r in R) + 
                                gp.quicksum(Z_varrho[k, r, j] for r in R) - M * (1 - y[k, i, j]))

for k in K:
    for i in N:
        model.addConstr(p[k, i] <= Q, name=f'truck_capacity_{k}_{i}')

# 16-23
for k in K:
    for r in R:
        for j in N:
            model.addConstr(Z_lambda[k, r, j] <= p_tilde[k, r, j])
            model.addConstr(Z_lambda[k, r, j] <= Q_tilde * lambda_var[k, r, j])
            model.addConstr(Z_lambda[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - lambda_var[k, r, j]))
            
            model.addConstr(Z_varrho[k, r, j] <= p_tilde[k, r, j])
            model.addConstr(Z_varrho[k, r, j] <= Q_tilde * varrho[k, r, j])
            model.addConstr(Z_varrho[k, r, j] >= p_tilde[k, r, j] - Q_tilde * (1 - varrho[k, r, j]))

# 24-25
for k in K:
    for r in R:
        for i in N:
            model.addConstr(p_tilde[k, r, i] >= gp.quicksum(q[u] * x_tilde[k, r, u] for u in C if q[u] > 0) - 
                            M * (1 - lambda_var[k, r, i]))
            model.addConstr(p_tilde[k, r, i] <= gp.quicksum(q[u] * x_tilde[k, r, u] for u in C if q[u] > 0) + 
                            M * (1 - lambda_var[k, r, i]))

# 26-27
for k in K:
    for r in R:
        for j in N:
            model.addConstr(p_tilde[k, r, j] <= M * (1 - varrho[k, r, j]))

# 28-29
for k in K:
    for r in R:
        for i in N:
            for j in N:
                model.addConstr(p_tilde[k, r, j] <= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] + 
                                M * (1 - y_tilde[k, r, i, j]))
                model.addConstr(p_tilde[k, r, j] >= p_tilde[k, r, i] - q[j] * x_tilde[k, r, j] - 
                                M * (1 - y_tilde[k, r, i, j]))

# 30
for k in K:
    for r in R:
        for i in N:
            model.addConstr(p_tilde[k, r, i] <= Q_tilde, name=f'drone_capacity_{k}_{r}_{i}')

# 31-33
for i in C:
    model.addConstr(xi[i] >= t_start[i])

for k in K:
    for i in N:
        model.addConstr(xi[i] >= a[k, i] - M * (1 - x[k, i]))

for k in K:
    for i in N:
        model.addConstr(xi[i] >= a_tilde[k, i] + tau_l - M * (1 - gp.quicksum(x_tilde[k, r, i] for r in R)))

# 34
for k in K:
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(a[k, j] >= b[k, i] + t[i][j] - M * (1 - y[k, i, j]))

# 35
for k in K:
    for r in R:
        for i in N:
            for j in N:
                model.addConstr(a_tilde[k, j] >= b_tilde[k, i] + tau_l + t_tilde[i][j] - M * (1 - y_tilde[k, r, i, j]))

# 36-38
for k in K:
    for i in C:
        model.addConstr(b[k, i] >= xi[i] + s[i])

for k in K:
    for r in R:
        for i in C:
            model.addConstr(b[k, i] >= b_tilde[k, i] + tau_l - M * (1 - lambda_var[k, r, i]))
            model.addConstr(b[k, i] >= a_tilde[k, i] + tau_r - M * (1 - varrho[k, r, i]))

# 39
for k in K:
    for r in R:
        for i in C:
            model.addConstr(b_tilde[k, i] >= xi[i] + s[i] - M * (1 - x_tilde[k, r, i]) - 
                            M * lambda_var[k, r, i] - M * varrho[k, r, i])

# 40-41
for k in K:
    for r in R:
        for i in C:
            model.addConstr(b_tilde[k, i] >= a[k, i] - M * (1 - lambda_var[k, r, i]))
            model.addConstr(b_tilde[k, i] <= b[k, i] + M * (1 - lambda_var[k, r, i]))

# 42-43
for k in K:
    for r in R:
        for j in N:
            model.addConstr(a_tilde[k, j] + h[k, r, j] >= a[k, j] - M * (1 - varrho[k, r, j]))
            model.addConstr(a_tilde[k, j] + h[k, r, j] + tau_r <= b[k, j] + M * (1 - varrho[k, r, j]))

# 44-47
for k in K:
    for r in R:
        for i in N:
            model.addConstr(h[k, r, i] >= a[k, i] - a_tilde[k, i] - M * (1 - varrho[k, r, i]))
            model.addConstr(h[k, r, i] >= xi[i] - a_tilde[k, i] - M * (1 - x_tilde[k, r, i]))
            model.addConstr(h[k, r, i] <= xi[i] - a_tilde[k, i] + M * (1 - x_tilde[k, r, i]))

# 48
for k in K:
    for r in R:
        model.addConstr(gp.quicksum(y_tilde[k, r, i, j] * t_tilde[i][j] for i in C for j in C) +
                        gp.quicksum(lambda_var[k, r, i] * tau_l for i in C) +
                        gp.quicksum(varrho[k, r, j] * tau_r for j in C) +
                        gp.quicksum(x_tilde[k, r, i] * s[i] for i in C) +
                        gp.quicksum(h[k, r, i] for i in C) <= T_max)

model.update()
print(f"Model created with {model.NumVars} variables and {model.NumConstrs} constraints")

print("\nSolving the model...")
model.setParam('TimeLimit', 600)  
model.setParam('MIPGap', 0.05)  
model.setParam('OutputFlag', 1)  

model.optimize()

if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print("\n" + "="*60)
    print("SOLUTION FOUND!")
    print("="*60)
    print(f"Objective value: {model.ObjVal:.2f}")
    print(f"Total cost: {cost.getValue():.2f}")
    print(f"Spanning time: {spanning_time.X:.2f}")
    print(f"Solution status: {model.status}")
    
    if model.status == GRB.TIME_LIMIT:
        print(f"MIP Gap: {model.MIPGap*100:.2f}%")
    
    print("\n" + "-"*60)
    print("TRUCK ROUTES:")
    print("-"*60)
    for k in K:
        route = []
        current = 0
        visited = set([0])
        route.append(0)
        
        max_iter = len(N)
        iter_count = 0
        while len(visited) < len([i for i in N if x[k, i].X > 0.5]) and iter_count < max_iter:
            found = False
            for j in N:
                if j not in visited and y[k, current, j].X > 0.5:
                    route.append(j)
                    visited.add(j)
                    current = j
                    found = True
                    break
            if not found:
                break
            iter_count += 1
        
        if len(route) > 1:
            print(f"Vehicle {k}: {' -> '.join(map(str, route))}")
            print(f"  Truck serves: {[i for i in C if x[k, i].X > 0.5]}")
    
    print("\n" + "-"*60)
    print("DRONE TRIPS:")
    print("-"*60)
    for k in K:
        for r in R:
            served = [i for i in C if x_tilde[k, r, i].X > 0.5]
            launch = [i for i in N if lambda_var[k, r, i].X > 0.5]
            land = [i for i in N if rho[k, r, i].X > 0.5]
            
            if served:
                print(f"Drone {k}, Trip {r}: Launch at {launch[0] if launch else 'N/A'}, "
                      f"Serve {served}, Land at {land[0] if land else 'N/A'}")

elif model.status == GRB.INFEASIBLE:
    print("\n" + "="*60)
    print("MODEL IS INFEASIBLE!")
    print("="*60)
    print("Computing IIS")
    model.computeIIS()
    model.write("model_iis.ilp")
    print("\nIIS written to 'model_iis.ilp'")
    print("\nConflicting constraints:")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"  {c.constrName}")
            
else:
    print("\nNo solution found!")
    print(f"Status: {model.status}")