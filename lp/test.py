import pulp as pl
import numpy as np
import json
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    type=str,
    help="Relative JSON filename, e.g. S042_N5_C_3G_R50.json",
)
parser.add_argument(
    "--subfolder",
    type=str,
    help="Relative JSON filename, e.g. S042_N5_C_3G_R50.json",
)

args = parser.parse_args()

DATA_ROOT = "../data/generated/data"
RESULT_ROOT = "./result/"

filename = args.filename  # e.g., "S42_N5_C_U_R50.json"
subfolder = args.subfolder

files = []
if filename and subfolder:
    print("Either one: filename OR subfolder")
    sys.exit(1)

if filename:
    files.append(filename)
else:
    subfolder_path = os.path.join(DATA_ROOT, subfolder)
    files = [
        file
        for file in os.listdir(subfolder_path)
        if os.path.isfile((os.path.join(subfolder_path, file)))
    ]


def run(filename):
    parts = filename.split("_")
    subfolder = parts[1]
    data_path = os.path.join(DATA_ROOT, subfolder, filename)

    # Prepare output path
    output_dir = os.path.join(RESULT_ROOT, subfolder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)

    with open(data_path, "r") as f:
        data = json.load(f)

    config = data["Config"]
    nodes = data["Nodes"]

    num_customers = config["General"]["NUM_CUSTOMERS"]
    # num_nodes = config["General"]["NUM_NODES"]
    # MAX_COORD = config["General"]["MAX_COORD_KM"]
    T_max = config["General"]["T_MAX_SYSTEM_H"]
    T_tilde_max = config["Vehicles"]["DRONE_DURATION_H"]
    V_TRUCK = config["Vehicles"]["V_TRUCK_KM_H"]
    V_DRONE = config["Vehicles"]["V_DRONE_KM_H"]
    Q = config["Vehicles"]["CAPACITY_TRUCK"]
    Q_tilde = config["Vehicles"]["CAPACITY_DRONE"]
    num_vehicles = config["Vehicles"]["NUM_TRUCKS"]
    # num_drones = config["Vehicles"]["NUM_DRONES"]

    # test configuration
    # T_tilde_max = 10.0
    # V_DRONE = 100.0

    tau_l = config["Vehicles"]["DRONE_TAKEOFF_MIN"] / 60.0
    tau_r = config["Vehicles"]["DRONE_LANDING_MIN"] / 60.0
    service_time = config["Vehicles"]["SERVICE_TIME_MIN"] / 60.0

    depot_info = config["Depot"]
    depot_idx = depot_info["id"]
    depot_coord = np.array(depot_info["coord"])
    depot_tw = depot_info["time_window_h"]

    coords = [depot_coord]
    demands = {depot_idx: 0}
    time_windows = {depot_idx: depot_tw}
    service_times = {depot_idx: 0}

    linehaul_indices = []
    backhaul_indices = []

    for node in nodes:
        node_id = node["id"]
        coords.append(np.array(node["coord"]))
        demands[node_id] = node["demand"]
        time_windows[node_id] = node["tw_h"]
        service_times[node_id] = service_time

        if node["type"] == "LINEHAUL":
            linehaul_indices.append(node_id)
        else:
            backhaul_indices.append(node_id)

    coords = np.array(coords)

    n_nodes = len(coords)
    end_depot_idx = n_nodes
    coords = np.vstack([coords, depot_coord])
    demands[end_depot_idx] = 0
    time_windows[end_depot_idx] = depot_tw
    service_times[end_depot_idx] = 0

    n_nodes = len(coords)

    L = linehaul_indices
    B = backhaul_indices
    C = L + B
    N = [depot_idx] + C
    N_end = C + [end_depot_idx]
    N_all = [depot_idx] + C + [end_depot_idx]

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

    t_start[end_depot_idx] = time_windows[end_depot_idx][0]
    t_end[end_depot_idx] = time_windows[end_depot_idx][1]
    s[end_depot_idx] = 0

    K = list(range(1, num_vehicles + 1))
    num_drone_routes = num_customers
    R = list(range(1, num_drone_routes + 1))

    c = 1.0
    c_tilde = 0.2
    M_edge = n_nodes * n_nodes
    M_node = end_depot_idx
    M_Q_tilde = Q_tilde + 1
    M_Q = Q + 1
    M_T = T_max + 1

    w1 = 0.0
    w2 = 1.0

    model = pl.LpProblem("VRPBD", pl.LpMinimize)

    x = pl.LpVariable.dicts("x", ((k, i) for k in K for i in N_all), cat="Binary")
    y = pl.LpVariable.dicts(
        "y",
        ((k, i, j) for k in K for i in N_all for j in N_all if i != j),
        cat="Binary",
    )
    z = pl.LpVariable.dicts(
        "z", ((k, i) for k in K for i in N_all), lowBound=0, cat="Integer"
    )
    p = pl.LpVariable.dicts(
        "p", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    a = pl.LpVariable.dicts(
        "a", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    b = pl.LpVariable.dicts(
        "b", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )

    x_tilde = pl.LpVariable.dicts(
        "x_tilde", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    y_tilde = pl.LpVariable.dicts(
        "y_tilde",
        ((k, r, i, j) for k in K for r in R for i in N_all for j in N_all if i != j),
        cat="Binary",
    )
    z_tilde = pl.LpVariable.dicts(
        "z_tilde",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Integer",
    )
    lambda_var = pl.LpVariable.dicts(
        "lambda", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    varrho = pl.LpVariable.dicts(
        "varrho", ((k, r, i) for k in K for r in R for i in N_all), cat="Binary"
    )
    p_tilde = pl.LpVariable.dicts(
        "p_tilde",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Continuous",
    )
    # a_tilde represent arrival time before landing step
    a_tilde = pl.LpVariable.dicts(
        "a_tilde", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    # b_tilde represent departure time before launching step
    b_tilde = pl.LpVariable.dicts(
        "b_tilde", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    h = pl.LpVariable.dicts(
        "h",
        ((k, r, i) for k in K for r in R for i in N_all),
        lowBound=0,
        cat="Continuous",
    )

    Z_lambda = pl.LpVariable.dicts(
        "Z_lambda",
        ((k, r) for k in K for r in R),
        lowBound=0,
        cat="Continuous",
    )
    Z_varrho = pl.LpVariable.dicts(
        "Z_varrho",
        ((k, r) for k in K for r in R),
        lowBound=0,
        cat="Continuous",
    )

    xi = pl.LpVariable.dicts("xi", (i for i in N_all), lowBound=0, cat="Continuous")

    m_lambda = pl.LpVariable.dicts(
        "m_lambda", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )
    m_varrho = pl.LpVariable.dicts(
        "m_varrho", ((k, i) for k in K for i in N_all), lowBound=0, cat="Continuous"
    )

    tardiness = pl.LpVariable("tardiness", lowBound=0, cat="Continuous")

    cost = pl.lpSum(
        [y[k, i, j] * c * d[i][j] for k in K for i in N for j in N_end if i != j]
    ) + pl.lpSum(
        [
            y_tilde[k, r, i, j] * c_tilde * d_tilde[i][j]
            for k in K
            for r in R
            for i in N
            for j in N_end
            if i != j
        ]
    )

    model += w1 * cost + w2 * tardiness

    for i in C:
        model += tardiness >= xi[i] + s[i] - t_end[i], f"tardiness_{i}"

    # each customer is served once
    for i in C:
        model += (
            pl.lpSum([x[k, i] for k in K])
            + pl.lpSum([x_tilde[k, r, i] for k in K for r in R])
        ) == 1

    # flow preservation for truck node
    for k in K:
        for i in N_end:
            model += pl.lpSum([y[k, j, i] for j in N if j != i]) == x[k, i]
        for i in N:
            model += pl.lpSum([y[k, i, j] for j in N_end if j != i]) == x[k, i]

    # flow preservation for drone node
    for k in K:
        for r in R:
            for i in C:
                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) >= 1 - M_node * (1 - x_tilde[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) <= 1 + M_node * (1 - x_tilde[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) >= 1 - M_node * (1 - x_tilde[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) <= 1 + M_node * (1 - x_tilde[k, r, i])

            for i in N:
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) >= 1 - M_node * (1 - lambda_var[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) <= 1 + M_node * (1 - lambda_var[k, r, i])

                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) >= -M_node * (
                    x_tilde[k, r, i] + lambda_var[k, r, i] + varrho[k, r, i]
                )
                model += pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if i != j]
                ) <= M_node * (x_tilde[k, r, i] + lambda_var[k, r, i] + varrho[k, r, i])

            for i in N_end:
                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) >= 1 - M_node * (1 - varrho[k, r, i])
                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) <= 1 + M_node * (1 - varrho[k, r, i])

                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) >= -M_node * (
                    x_tilde[k, r, i] + varrho[k, r, i] + lambda_var[k, r, i]
                )
                model += pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if i != j]
                ) <= M_node * (x_tilde[k, r, i] + varrho[k, r, i] + lambda_var[k, r, i])

    # if trip (k, r) is unspecified, then its edge set is unmarked
    for k in K:
        for r in R:
            model += pl.lpSum(
                [y_tilde[k, r, i, j] for i in N for j in N_end if i != j]
            ) <= M_edge * pl.lpSum([lambda_var[k, r, i] for i in N])

    # if fleet k uses only drone, y[k, 0, end_depot_idx] set as 1
    for k in K:
        for r in R:
            model += pl.lpSum(
                [y[k, i, j] for i in N for j in N_end if i != j]
            ) <= M_edge * (1 - y[k, 0, end_depot_idx])

    # launching node and landing node must co-exist
    for k in K:
        for r in R:
            # model += pl.lpSum([lambda_var[k, r, i] for i in N]) <= 1
            # model += pl.lpSum([varrho[k, r, j] for j in N_end]) <= 1
            model += pl.lpSum([lambda_var[k, r, i] for i in N]) == pl.lpSum(
                [varrho[k, r, j] for j in N_end]
            )

    # launching node must be the truck node and outgoing node by drone
    for k in K:
        for r in R:
            for i in N:
                model += lambda_var[k, r, i] <= x[k, i]
                model += lambda_var[k, r, i] <= pl.lpSum(
                    [y_tilde[k, r, i, j] for j in N_end if j != i]
                )
                model += (
                    lambda_var[k, r, i]
                    >= x[k, i]
                    + pl.lpSum([y_tilde[k, r, i, j] for j in N_end if j != i])
                    - 1
                )

    # landing node must be the truck node and ingoing node by drone
    for k in K:
        for r in R:
            for i in N_end:
                model += varrho[k, r, i] <= x[k, i]
                model += varrho[k, r, i] <= pl.lpSum(
                    [y_tilde[k, r, j, i] for j in N if j != i]
                )
                model += (
                    varrho[k, r, i]
                    >= x[k, i]
                    + pl.lpSum([y_tilde[k, r, j, i] for j in N if j != i])
                    - 1
                )

    # drone launching node can not be 'end_depot_idx'
    # drone landing node can not be '0'
    for k in K:
        for r in R:
            model += lambda_var[k, r, end_depot_idx] == 0
            model += varrho[k, r, 0] == 0

    # MTZ of truck
    for k in K:
        for i in N:
            for j in N_end:
                if i != j:
                    model += z[k, i] - z[k, j] + 1 <= M_node * (1 - y[k, i, j])
                    model += z[k, i] - z[k, j] + 1 >= -M_node * (1 - y[k, i, j])

    # MTZ of drone
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += z_tilde[k, r, i] - z_tilde[k, r, j] + 1 <= M_node * (
                            1 - y_tilde[k, r, i, j]
                        )
                        model += z_tilde[k, r, i] - z_tilde[k, r, j] + 1 >= -M_node * (
                            1 - y_tilde[k, r, i, j]
                        )

    # the volume drone trip (k, r) carries from launching node
    # the volume drone trip (k, r) carries to landing node
    for k in K:
        for r in R:
            model += Z_lambda[k, r] == pl.lpSum([q[i] * x_tilde[k, r, i] for i in L])
            model += Z_varrho[k, r] == -pl.lpSum([q[i] * x_tilde[k, r, i] for i in B])

    # the volume drone k carries to landing node i
    for k in K:
        for r in R:
            for i in N_end:
                model += m_varrho[k, i] - Z_varrho[k, r] <= M_Q_tilde * (
                    1 - varrho[k, r, i]
                )
                model += m_varrho[k, i] - Z_varrho[k, r] >= -M_Q_tilde * (
                    1 - varrho[k, r, i]
                )

            for i in N:
                model += m_lambda[k, i] - Z_lambda[k, r] <= M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )
                model += m_lambda[k, i] - Z_lambda[k, r] >= -M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )

    for k in K:
        for i in N:
            model += m_lambda[k, i] <= M_Q_tilde * pl.lpSum(
                [lambda_var[k, r, i] for r in R]
            )
        for i in N_end:
            model += m_varrho[k, i] <= M_Q_tilde * pl.lpSum(
                [varrho[k, r, i] for r in R]
            )

    for k in K:
        model += (
            p[k, 0]
            == pl.lpSum([q[i] * x[k, i] for i in L])
            + pl.lpSum([q[i] * x_tilde[k, r, i] for r in R for i in L])
            - m_lambda[k, 0]
        )
    # relationship between payloads at two consecutive truck nodes
    for k in K:
        for i in N:
            for j in N_end:
                if i != j:
                    # if edge (i, j) sets and j is neither landing node nor launching node
                    model += p[k, j] <= p[k, i] - q[j] - m_lambda[k, j] + m_varrho[
                        k, j
                    ] + M_Q * (1 - y[k, i, j])
                    model += p[k, j] >= p[k, i] - q[j] - m_lambda[k, j] + m_varrho[
                        k, j
                    ] - M_Q * (1 - y[k, i, j])

    # upper bound and lower bound of payload in each truck
    for k in K:
        for i in N:
            model += p[k, i] <= Q
            model += p[k, i] >= 0

    # payload of drone carries from launching node
    for k in K:
        for r in R:
            for i in N:
                model += p_tilde[k, r, i] - m_lambda[k, i] >= -M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )
                model += p_tilde[k, r, i] - m_lambda[k, i] <= M_Q_tilde * (
                    1 - lambda_var[k, r, i]
                )

    # drone payload at landing node equals 0 because it passes all for its truck
    for k in K:
        for r in R:
            for i in N_all:
                model += p_tilde[k, r, i] >= -M_Q_tilde * (
                    1 - varrho[k, r, i] + lambda_var[k, r, i]
                )
                model += p_tilde[k, r, i] <= M_Q_tilde * (
                    1 - varrho[k, r, i] + lambda_var[k, r, i]
                )
    for k in K:
        for r in R:
            for i in N_all:
                model += p_tilde[k, r, i] >= -M_Q_tilde * (
                    1 - x[k, i] + varrho[k, r, i] + lambda_var[k, r, i]
                )
                model += p_tilde[k, r, i] <= M_Q_tilde * (
                    1 - x[k, i] + varrho[k, r, i] + lambda_var[k, r, i]
                )
    # drone payload between two consecutive nodes from [launching node ... last served node]
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += p_tilde[k, r, j] <= p_tilde[k, r, i] - q[
                            j
                        ] + M_Q_tilde * (1 - y_tilde[k, r, i, j] + varrho[k, r, j])
                        model += p_tilde[k, r, j] >= p_tilde[k, r, i] - q[
                            j
                        ] - M_Q_tilde * (1 - y_tilde[k, r, i, j] + varrho[k, r, j])

    # upper bound and lower bound of drone payload
    for k in K:
        for r in R:
            for i in N_all:
                model += p_tilde[k, r, i] <= Q_tilde
                model += p_tilde[k, r, i] >= 0

    # served time >= lower bound of provided time window
    for i in C:
        model += xi[i] >= t_start[i]

    # truck: served time >= arrival time
    # drone: served time >= arrival time + landing time
    for k in K:
        for i in C:
            model += xi[i] >= a[k, i] - M_T * (1 - x[k, i])
            model += xi[i] >= a_tilde[k, i] + tau_r - M_T * (
                1 - pl.lpSum([x_tilde[k, r, i] for r in R])
            )
    # depature time >= served time + service time
    for k in K:
        for i in C:
            model += b[k, i] >= xi[i] + s[i] - M_T * (1 - x[k, i])
            model += b_tilde[k, i] >= xi[i] + s[i] - M_T * (
                1 - pl.lpSum([x_tilde[k, r, i] for r in R])
            )

    # arrival time and departure time of 2 consecutive nodes in truck route
    for k in K:
        for i in N:
            for j in N_end:
                if i != j:
                    model += a[k, j] >= b[k, i] + t[i][j] - M_T * (1 - y[k, i, j])
                    model += a[k, j] <= b[k, i] + t[i][j] + M_T * (1 - y[k, i, j])

    # arrival time and departure time of 2 consecutive nodes in drone route
    for k in K:
        for r in R:
            for i in N:
                for j in N_end:
                    if i != j:
                        model += a_tilde[k, j] >= b_tilde[k, i] + tau_l + t_tilde[i][
                            j
                        ] - M_T * (1 - y_tilde[k, r, i, j])
                        model += a_tilde[k, j] <= b_tilde[k, i] + tau_l + t_tilde[i][
                            j
                        ] + M_T * (1 - y_tilde[k, r, i, j])

    # at launching node, drone must launch between arrival time and departure time of related truck
    for k in K:
        for r in R:
            for i in N:
                model += b_tilde[k, i] <= b[k, i] + M_T * (1 - lambda_var[k, r, i])
                model += b_tilde[k, i] >= a[k, i] - M_T * (1 - lambda_var[k, r, i])
    # landing time at i must belong to operation time window at i of truck
    for k in K:
        for r in R:
            for j in N_end:
                model += a_tilde[k, j] + h[k, r, j] + tau_r >= a[k, j] - M_T * (
                    1 - varrho[k, r, j]
                )
            for j in N:
                model += a_tilde[k, j] + h[k, r, j] + tau_r <= b[k, j] + M_T * (
                    1 - varrho[k, r, j]
                )

    # at drone-served node i:
    # waiting time = served time - (arrival time + landing time)
    # waiting time >= 0
    for k in K:
        for r in R:
            for i in C:
                model += h[k, r, i] >= xi[i] - a_tilde[k, i] - tau_r - M_T * (
                    1 - x_tilde[k, r, i]
                )
                model += h[k, r, i] <= xi[i] - a_tilde[k, i] - tau_r + M_T * (
                    1 - x_tilde[k, r, i]
                )
                model += h[k, r, i] >= 0

    # drone usage limitation
    for k in K:
        for r in R:
            flight_time = pl.lpSum(
                [
                    y_tilde[k, r, i, j] * t_tilde[i][j]
                    for i in N
                    for j in N_end
                    if i != j
                ]
            )
            # launch_time = pl.lpSum([lambda_var[k, r, i] * tau_l for i in N])
            # land_time = pl.lpSum([varrho[k, r, j] * tau_r for j in N_end])
            launch_time = tau_l * (
                pl.lpSum([lambda_var[k, r, i] for i in N])
                + pl.lpSum([x_tilde[k, r, i] for i in C])
            )
            land_time = tau_r * (
                pl.lpSum([varrho[k, r, j] for j in N_end])
                + pl.lpSum([x_tilde[k, r, i] for i in C])
            )
            service_time = pl.lpSum([x_tilde[k, r, i] * s[i] for i in C])
            wait_time = pl.lpSum([h[k, r, i] for i in N])

            total_time = (
                flight_time + launch_time + land_time + service_time + wait_time
            )
            model += total_time <= T_tilde_max

    # if i is landing node of trip (k, r), j is launching node of trip (k, r+1)
    # so:
    # 1. landing node of (k, r) <= launching node of (k, r+1) in truck route's sequence
    # 2. drone must landing before continueing launching in the next trips
    for k in K:
        for r in R[:-1]:
            for i in N_end:
                for j in N:
                    model += z[k, j] >= z[k, i] - M_node * (
                        2 - varrho[k, r, i] - lambda_var[k, r + 1, j]
                    )

            for i in N_end:
                for j in N:
                    model += b_tilde[k, j] >= a_tilde[k, i] + tau_r - M_node * (
                        2 - varrho[k, r, i] - lambda_var[k, r + 1, j]
                    )

    # in fleet k, the drone trip r must be specified before drone trip r+1
    for k in K:
        for r in R[:-1]:
            model += pl.lpSum(lambda_var[k, r + 1, i] for i in N) <= pl.lpSum(
                lambda_var[k, r, i] for i in N
            )

    # drone k must be incoorperated with truck k
    for k in K:
        for r in R:
            model += pl.lpSum(lambda_var[k, r, i] for i in N) <= pl.lpSum(
                y[k, 0, j] for j in N_end
            )

    # truck k must be used before truck k+1
    for k in K[:-1]:
        model += pl.lpSum(y[k, 0, j] for j in C) >= pl.lpSum(y[k + 1, 0, j] for j in C)

    print(
        f"Model created with {len(model.variables())} variables and {len(model.constraints)} constraints"
    )
    print("\nSolving the model")

    model.writeLP("model.lp")

    # solver = pl.PULP_CBC_CMD(timeLimit=36000, gapRel=0.05, msg=True)
    solver = pl.PULP_CBC_CMD(timeLimit=1800, gapRel=0.01, msg=True)
    model.solve(solver)

    if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
        result_data = {
            "weights": {
                "cost": w1,
                "tardiness": w2,
                "wait": 0.0,  # Adjust if you add a weight for wait time
            },
            "objective": round(pl.value(model.objective), 2),
            "routes": [],
        }

        print("[TEST]: x")
        for k, v in x.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: x_tilde")
        for k, v in x_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: y")
        for k, v in y.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: y_tilde")
        for k, v in y_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: Z_lambda")
        for k, v in Z_lambda.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: Z_varrho")
        for k, v in Z_varrho.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: z")
        for k, v in z.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: z_tilde")
        for k, v in z_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: a")
        for k, v in a.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: xi")
        for k, v in xi.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: b")
        for k, v in b.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: h")
        for k, v in h.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: m_lambda")
        for k, v in m_lambda.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: m_varrho")
        for k, v in m_varrho.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: lambda")
        for k, v in lambda_var.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: varrho")
        for k, v in varrho.items():
            print(f"\t {k, pl.value(v)}")
        print("[TEST]: p", Q)
        for k, v in p.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: p_tilde", Q_tilde)
        for k, v in p_tilde.items():
            print(f"\t {k, pl.value(v)}")

        print("[TEST]: tardiness.value")
        print(pl.value(tardiness))

        print("[TEST]: tardiness correctness")
        temp = 0.0
        for i in C:
            temp = max(temp, xi[i].varValue + s[i] - t_end[i])

        if tardiness == temp:
            print(True)
        else:
            print(False)

        # parse from lp model variables into solution and store it into .json format
        for k in K:
            route = [0]
            current = 0
            visited = {0}

            # 1. Reconstruct Truck Route
            max_iter = len(N_all) * 2
            for _ in range(max_iter):
                found = False
                for j in N_end:  # Use N_all to ensure end_depot_idx is reachable
                    if (
                        j not in visited
                        and y[k, current, j].varValue
                        and pl.value(y[k, current, j]) > 0.5
                    ):
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found or current == end_depot_idx:
                    break

            # 2. Process Route Data (Only if truck left the depot)
            if len(route) > 1:
                route_info = {
                    "id": int(k),
                    "route": route,
                    "arrival": [
                        round(pl.value(a[k, i]), 2) if idx > 0 else None
                        for idx, i in enumerate(route)
                    ],
                    "departure": [
                        round(pl.value(b[k, i]), 2) if idx < len(route) - 1 else None
                        for idx, i in enumerate(route)
                    ],
                    "trips": [],
                }

                # 3. Process Drone Trips (Sorties)
                for r in R:
                    # Check if this specific trip r for truck k was used
                    # We identify the launch node first
                    launch_nodes = [
                        i
                        for i in N
                        if pl.value(lambda_var[k, r, i])
                        and pl.value(lambda_var[k, r, i]) > 0.5
                    ]

                    if launch_nodes:
                        launch_node = launch_nodes[0]
                        drone_route = [launch_node]
                        drone_current = launch_node
                        drone_visited = {launch_node}

                        # Reconstruct the drone path for this specific sortie
                        for _ in range(len(N_all)):
                            found_next = False
                            for j in N_all:
                                if (
                                    j not in drone_visited
                                    and y_tilde[k, r, drone_current, j].varValue
                                    and pl.value(y_tilde[k, r, drone_current, j]) > 0.5
                                ):
                                    drone_route.append(j)
                                    drone_visited.add(j)
                                    drone_current = j
                                    found_next = True
                                    break
                            if not found_next:
                                break

                        if len(drone_route) < 3:
                            break
                        # Retrieve arrival/departure for drone trip
                        trip_info = {
                            "id": int(r),
                            "route": drone_route,
                            "arrival": [
                                round(pl.value(a_tilde[k, i]), 2) if idx > 0 else None
                                for idx, i in enumerate(drone_route)
                            ],
                            "departure": [
                                round(pl.value(b_tilde[k, i]), 2)
                                if idx < len(drone_route) - 1
                                else None
                                for idx, i in enumerate(drone_route)
                            ],
                        }
                        route_info["trips"].append(trip_info)

                result_data["routes"].append(route_info)

        os.remove(output_path)
        # Write to file
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)

        print(f"✔ Results saved to {output_path}")

    else:
        print("\n")
        print("NO SOLUTION FOUND!")
        print(f"Status: {pl.LpStatus[model.status]}")


for file in files:
    run(file)
