import os
import numpy as np
import random
import math
import pandas as pd


def generate_random_data(num_customers, T_max=10.0, speed_factor=1.0):
    """
    Generates a VRPBTW instance using Solomon's structure for time windows.

    Args:
        num_customers (int): The number of customer nodes (N).
        T_max (float): The total time horizon (Due Time of the Depot).
        speed_factor (float): Multiplier for distance to get travel time.
                               (Higher factor means faster travel/tighter time constraints)

    Returns:
        pd.DataFrame: DataFrame containing the node data.
    """
    # Set seed for reproducibility
    np.random.seed(42 + num_customers)
    random.seed(42 + num_customers)

    # 1. Initialize Depot Node (ID 0)
    depot_data = {
        "ID": 0,
        "X_COORD": 0.0,
        "Y_COORD": 0.0,
        "DEMAND": 0,
        "SERVICE_TIME": 0.0,
        "READY_TIME": 0.0,
        "DUE_TIME": T_max,
    }

    data = [depot_data]

    # 2. Determine Linehaul and Backhaul counts (1:1 ratio)
    num_linehaul = num_customers // 2

    # Sample coordinates (ID 1 to N) uniformly from [0, 1] x [0, 1]
    coords = np.random.uniform(0, 1, size=(num_customers, 2))

    # Randomly assign customer type (Linehaul vs. Backhaul)
    customer_indices = list(range(1, num_customers + 1))
    random.shuffle(customer_indices)
    linehaul_indices = set(customer_indices[:num_linehaul])

    for i in range(1, num_customers + 1):
        x, y = coords[i - 1]

        # --- Demand Generation (Signed) ---
        if i in linehaul_indices:
            # Linehaul: Positive Demand [1, 10]
            demand = random.randint(1, 10)
            cust_type = "Linehaul"
        else:
            # Backhaul: Negative Demand [-10, -1]
            demand = random.randint(-10, -1)
            cust_type = "Backhaul"

        # Random service time (small and positive)
        service_time = round(random.uniform(0.01, 0.1), 2)

        # --- Solomon Time Window Generation ---

        # Calculate Euclidean Distance from Depot (0,0)
        distance_to_depot = math.sqrt(x**2 + y**2)

        # EAT (Earliest Arrival Time) = Travel Time from Depot
        # EAT is the minimum time needed to travel from the depot
        EAT = distance_to_depot * speed_factor

        # Sampling coefficients for time window tightness (Tau_a and Tau_b in [0, 1])
        # To mimic clustering/randomness, we use different uniform ranges.
        # For simplicity and to introduce variance, we sample widely:
        tau_a = random.uniform(0.1, 0.9)  # Controls how far 'a_i' is from EAT
        tau_b = random.uniform(0.1, 0.9)  # Controls how far 'b_i' is from EAT

        # Solomon Formula for Ready Time (a_i) and Due Time (b_i):
        # The terms (T_max - EAT) represent the remaining slack time.

        # Ready Time (a_i): Time window starts *before* EAT
        ready_time = max(0.0, EAT - tau_a * (EAT))

        # Due Time (b_i): Time window ends *after* EAT
        due_time = EAT + service_time + tau_b * (T_max - EAT)

        # Final validation: Ensure ready_time <= due_time and due_time <= T_max
        if ready_time >= due_time:
            # If the window is too tight or inverted, ensure a small, valid window
            ready_time = max(0.0, EAT - 0.5)
            due_time = min(T_max, ready_time + 1.0)  # minimum width of 1.0

        data.append(
            {
                "ID": i,
                "X_COORD": x,
                "Y_COORD": y,
                "DEMAND": demand,
                "SERVICE_TIME": service_time,
                "READY_TIME": round(ready_time, 2),
                "DUE_TIME": round(due_time, 2),
                "CUSTOMER_TYPE": cust_type,
            }
        )

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":
    CUSTOMER_SIZES = [20, 50, 100, 150]
    OUTPUT_FOLDER = "data"
    print("Generating VRPBTW Datasets")

    dataset_df = None

    for N in CUSTOMER_SIZES:
        # We will use T_max = 10.0 and speed_factor = 1.0 for consistency
        dataset_df = generate_random_data(N, T_max=10.0, speed_factor=1.0)
        output_filename = f"VRPBTW_N{N}.csv"

        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        dataset_df.to_csv(output_path, index=False)

        print(
            f"- Successfully generated dataset: {output_filename} ({len(dataset_df)} nodes)."
        )

    print("\n---")
    print("- Dataset Preview")
    if dataset_df is not None:
        print(dataset_df.head())
    else:
        print("Error: No dataset is generated")
