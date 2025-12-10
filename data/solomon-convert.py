import os
import glob
import math
import json
import shutil


# --- Configuration ---
SOLOMON_ROOT = "benchmark/vrptw/vrptw-solomon-100"
VRPBTW_ROOT = "benchmark/vrpbtw/vrpbtw-solomon-100-derived"

TARGET_SIZES = [10, 15, 20, 50, 100]

# (Linehaul %, Backhaul %) – backhaul % is not used directly but kept for readability
RATIOS = [
    (50, 50),
    (60, 40),
    (70, 30),
]

FIELDS = [
    "CUST NO.",
    "XCOORD.",
    "YCOORD.",
    "DEMAND",
    "READY TIME",
    "DUE DATE",
    "SERVICE TIME",
]


# -------------------------------------------------------------
# Parse Solomon format
# -------------------------------------------------------------
def parse_solomon_file(filepath):
    """
    Reads a Solomon VRPTW .txt file and extracts:
    - max vehicles
    - capacity
    - depot record
    - customer records
    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) < 10:
        raise ValueError("Invalid Solomon instance: too few lines")

    # Vehicle info (line 5 in 1-based)
    vehicle_line = lines[4].split()
    max_vehicles = int(vehicle_line[0])
    capacity = int(vehicle_line[1])

    # Customer table starts at line 10
    customers = []
    for line in lines[9:]:
        parts = line.split()
        if not parts:
            continue

        customers.append(
            {
                "CUST NO.": int(parts[0]),
                "XCOORD.": int(parts[1]),
                "YCOORD.": int(parts[2]),
                "DEMAND": int(parts[3]),
                "READY TIME": int(parts[4]),
                "DUE DATE": int(parts[5]),
                "SERVICE TIME": int(parts[6]),
            }
        )

    depot = customers[0]
    customer_nodes = customers[1:]

    return max_vehicles, capacity, depot, customer_nodes


# -------------------------------------------------------------
# Create VRPBTW JSON instance
# -------------------------------------------------------------
def create_vrpbtw_instance_json(
    num_vehicle,
    capacity,
    depot,
    original_customers,
    target_size,
    linehaul_percent,
    filename_base,
):
    """
    Creates a JSON VRPBTW instance:
    - Subsets customers
    - Splits into linehaul (+demand) and backhaul (-demand)
    """

    # Select subset of customers
    subset = original_customers[:target_size]

    # Split L/B
    linehaul_fraction = linehaul_percent / 100
    num_linehaul = math.ceil(target_size * linehaul_fraction)
    num_backhaul = target_size - num_linehaul

    linehaul_customers = subset[:num_linehaul]
    backhaul_customers = subset[num_linehaul:]

    # Construct all nodes (depot first)
    depot_copy = depot.copy()
    depot_copy["DEMAND"] = 0

    all_customers = []

    # Linehaul (positive demand)
    for c in linehaul_customers:
        all_customers.append(c.copy())

    # Backhaul (negative demand)
    for c in backhaul_customers:
        new_c = c.copy()
        new_c["DEMAND"] = -new_c["DEMAND"]
        all_customers.append(new_c)

    # File + directory
    instance_name = f"{filename_base.lower()}-n{target_size}-b{num_backhaul}"
    output_subfolder = f"n{target_size}"
    output_dir = os.path.join(VRPBTW_ROOT, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, instance_name + ".json")

    # Build JSON structure (clean, minimal)
    json_data = {
        "instance": instance_name,
        "base": filename_base.lower(),
        "num_vehicle": num_vehicle,
        "capacity": capacity,
        "n_customers": target_size,
        "n_backhaul": num_backhaul,
        "depot": {
            "x": depot_copy["XCOORD."],
            "y": depot_copy["YCOORD."],
            "ready_time": depot_copy["READY TIME"],
            "due_date": depot_copy["DUE DATE"],
            "service_time": depot_copy["SERVICE TIME"],
        },
        "customers": [
            {
                "id": c["CUST NO."],
                "x": c["XCOORD."],
                "y": c["YCOORD."],
                "demand": c["DEMAND"],
                "ready_time": c["READY TIME"],
                "due_date": c["DUE DATE"],
                "service_time": c["SERVICE TIME"],
            }
            for c in all_customers
        ],
    }

    # Save JSON
    with open(output_path, "w") as jf:
        json.dump(json_data, jf, indent=4)

    print(f"Generated JSON: {output_path}")


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
def generate_vrpbtw_datasets():
    print("--- Generating VRPBTW JSON datasets ---")
    print(f"Source: {SOLOMON_ROOT}")
    print(f"Destination: {VRPBTW_ROOT}")

    # Remove old output
    if os.path.exists(VRPBTW_ROOT):
        print(f"Removing existing directory: {VRPBTW_ROOT}")
        shutil.rmtree(VRPBTW_ROOT)

    # Load all Solomon files
    all_files = glob.glob(os.path.join(SOLOMON_ROOT, "**", "*.txt"), recursive=True)

    if not all_files:
        print("No Solomon instances found!")
        return

    for filepath in all_files:
        try:
            num_vehicle, capacity, depot, customers = parse_solomon_file(filepath)
            filename_base = os.path.basename(filepath).split(".")[0]
            max_size = len(customers)

            # Only 100- or 200- customer Solomon files are allowed
            if max_size == 100:
                allowed_sizes = [s for s in TARGET_SIZES if s <= 100]
            elif max_size == 200:
                allowed_sizes = [150, 200]
            else:
                print(f"Skipping {filename_base}: unexpected {max_size} customers.")
                continue

            # Generate datasets
            for size in allowed_sizes:
                for linehaul_pct, _ in RATIOS:
                    create_vrpbtw_instance_json(
                        num_vehicle,
                        capacity,
                        depot,
                        customers,
                        size,
                        linehaul_pct,
                        filename_base,
                    )

        except Exception as e:
            print(f"Error processing {filepath}: {e}")


# -------------------------------------------------------------
if __name__ == "__main__":
    generate_vrpbtw_datasets()
