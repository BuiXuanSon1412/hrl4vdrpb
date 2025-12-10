import os
import glob
import math
import json
import shutil

SOLOMON_ROOT = "benchmark/vrptw/vrptw-homberger-200"
VRPBTW_ROOT = "benchmark/vrpbtw/vrpbtw-homberger-200-derived"

TARGET_SIZES = [150, 200]
RATIOS = [(50, 50), (60, 40), (70, 30)]


def parse_solomon_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    vehicle_line = lines[4].split()
    max_vehicles = int(vehicle_line[0])
    capacity = int(vehicle_line[1])

    customers = []
    for i in range(9, len(lines)):
        parts = lines[i].split()
        if not parts:
            continue

        customers.append(
            {
                "id": int(parts[0]),
                "x": int(parts[1]),
                "y": int(parts[2]),
                "demand": int(parts[3]),
                "ready_time": int(parts[4]),
                "due_date": int(parts[5]),
                "service_time": int(parts[6]),
            }
        )

    depot = customers[0]
    customer_nodes = customers[1:]

    return max_vehicles, capacity, depot, customer_nodes


def write_instance_json(
    num_vehicle,
    capacity,
    depot,
    customers,
    target_size,
    linehaul_pct,
    filename_base,
):
    subset = customers[:target_size]

    num_linehaul = math.ceil(target_size * (linehaul_pct / 100))
    num_backhaul = target_size - num_linehaul

    linehaul = subset[:num_linehaul]
    backhaul = subset[num_linehaul:]

    # Create JSON fields
    depot_json = {
        "x": depot["x"],
        "y": depot["y"],
        "ready_time": depot["ready_time"],
        "due_date": depot["due_date"],
        "service_time": depot["service_time"],
    }

    customer_list = []

    # Linehaul customers (positive demand)
    for c in linehaul:
        customer_list.append(
            {
                "id": c["id"],
                "x": c["x"],
                "y": c["y"],
                "demand": c["demand"],
                "ready_time": c["ready_time"],
                "due_date": c["due_date"],
                "service_time": c["service_time"],
            }
        )

    # Backhaul customers (negative demand)
    for c in backhaul:
        customer_list.append(
            {
                "id": c["id"],
                "x": c["x"],
                "y": c["y"],
                "demand": -c["demand"],  # NEGATIVE
                "ready_time": c["ready_time"],
                "due_date": c["due_date"],
                "service_time": c["service_time"],
            }
        )

    # Filename format: {base}-n{customers}-b{backhaul}.json
    base = filename_base.lower()
    instance_name = f"{base}-n{target_size}-b{num_backhaul}"
    filename = instance_name + ".json"

    output_subfolder = f"n{target_size}"
    output_dir = os.path.join(VRPBTW_ROOT, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    json_data = {
        "instance": instance_name,
        "base": base,
        "num_vehicle": num_vehicle,
        "capacity": capacity,
        "n_customers": target_size,
        "n_backhaul": num_backhaul,
        "depot": depot_json,
        "customers": customer_list,
    }

    with open(output_path, "w") as jf:
        json.dump(json_data, jf, indent=4)

    print(f"Generated {output_path}")
    return output_path


def generate_vrpbtw_datasets():
    if os.path.exists(VRPBTW_ROOT):
        shutil.rmtree(VRPBTW_ROOT)

    all_files = glob.glob(os.path.join(SOLOMON_ROOT, "**", "*.TXT"), recursive=True)

    if not all_files:
        print("No source files found.")
        return

    for filepath in all_files:
        try:
            num_vehicle, capacity, depot, customers = parse_solomon_file(filepath)

            filename_base = os.path.basename(filepath).split(".")[0]
            max_size = len(customers)

            if max_size == 100:
                target_sizes = [s for s in TARGET_SIZES if s <= 100]
            elif max_size == 200:
                target_sizes = [150, 200]
            else:
                continue

            for size in target_sizes:
                for linehaul_pct, _ in RATIOS:
                    write_instance_json(
                        num_vehicle,
                        capacity,
                        depot,
                        customers,
                        size,
                        linehaul_pct,
                        filename_base,
                    )

        except Exception as e:
            print(f"Error reading {filepath}: {e}")


if __name__ == "__main__":
    generate_vrpbtw_datasets()
