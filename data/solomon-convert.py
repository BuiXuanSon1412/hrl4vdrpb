import os
import glob
import math
import csv
import shutil  # Needed for folder removal

# --- Configuration ---
# Root directory containing the Solomon files
SOLOMON_ROOT = "benchmark/vrptw-solomon-100"
# Directory to save the new VRPBTW datasets (before subfolders)
VRPBTW_ROOT = "benchmark/vrpbtw-solomon-100-derived"

# Target customer counts to generate
TARGET_SIZES = [10, 15, 20, 50, 100]

# Linehaul percentage ratios (Linehaul / Backhaul)
RATIOS = [
    (50, 50),
    (60, 40),
    (70, 30),
]

OUTPUT_FIELDS = [
    "CUST NO.",
    "XCOORD.",
    "YCOORD.",
    "DEMAND",
    "READY TIME",
    "DUE DATE",
    "SERVICE TIME",
]

# --- File Parsing and Conversion Logic ---


def parse_solomon_file(filepath):
    """
    Reads a standard Solomon VRPTW file and extracts problem data.
    Returns: (max_vehicles, vehicle_capacity, depot_data, customer_data)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) < 10:
        raise ValueError("File is too short to be a standard Solomon instance.")

    # Line 4 contains vehicle data
    vehicle_line = lines[4].split()
    max_vehicles = int(vehicle_line[0])
    capacity = int(vehicle_line[1])

    # Data starts at line 10
    customer_data = []
    for i in range(9, len(lines)):
        line = lines[i].split()
        if not line:
            continue

        data = {
            "CUST NO.": int(line[0]),
            "XCOORD.": int(line[1]),
            "YCOORD.": int(line[2]),
            "DEMAND": int(line[3]),
            "READY TIME": int(line[4]),
            "DUE DATE": int(line[5]),
            "SERVICE TIME": int(line[6]),
        }
        customer_data.append(data)

    depot_data = customer_data[0]
    customer_nodes = customer_data[1:]

    return max_vehicles, capacity, depot_data, customer_nodes


def create_vrpbtw_instance_csv(
    max_vehicles,
    depot_data,
    original_customers,
    target_size,
    linehaul_percent,
    filename_base,
):
    """
    Creates a new VRPBTW instance (CSV format) by subsetting customers and
    splitting demand using positive (linehaul) / negative (backhaul) convention.
    Uses the full filename_base for uniqueness.
    """

    # 1. Select the subset of customers (nodes 1 to target_size)
    subset_customers = original_customers[:target_size]

    # 2. Split into Linehaul (positive DEMAND) and Backhaul (negative DEMAND)
    linehaul_percent_value = linehaul_percent / 100
    num_linehaul = math.ceil(target_size * linehaul_percent_value)
    num_backhaul = target_size - num_linehaul

    linehaul_customers = subset_customers[:num_linehaul]
    backhaul_customers = subset_customers[num_linehaul:]

    # 3. Assign Demands and Format
    all_nodes = []
    depot = depot_data.copy()
    depot["DEMAND"] = 0
    all_nodes.append(depot)

    for cust in linehaul_customers:
        new_cust = cust.copy()
        all_nodes.append(new_cust)

    for cust in backhaul_customers:
        new_cust = cust.copy()
        new_cust["DEMAND"] = -new_cust["DEMAND"]
        all_nodes.append(new_cust)

    # 4. Format the new Filename and Folder
    distribution_identifier = filename_base.lower()

    # Filename: {OriginalFileBase}-n{customers}-b{backhaul}-v{vehicles}.csv
    new_filename = (
        f"{distribution_identifier}-n{target_size}-b{num_backhaul}-k{max_vehicles}.csv"
    )

    # New Subfolder: n<number of customers> (e.g., n100)
    output_subfolder = f"n{target_size}"
    output_dir = os.path.join(VRPBTW_ROOT, output_subfolder)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, new_filename)

    # 5. Write to CSV
    with open(output_filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_nodes)

    print(f"Generated: {output_filepath}")


# --- Main Execution ---


def generate_vrpbtw_datasets():
    """
    Main function to orchestrate the dataset generation, including folder cleanup.
    """
    print("--- Starting VRPBTW Dataset Generation (CSV Format, Organized) ---")
    print(f"Source: {SOLOMON_ROOT}")
    print(f"Target Base: {VRPBTW_ROOT}")

    # --- FOLDER CLEANUP IMPLEMENTATION ---
    if os.path.exists(VRPBTW_ROOT):
        print(f"Removing existing output directory: {VRPBTW_ROOT}")
        try:
            shutil.rmtree(VRPBTW_ROOT)
        except OSError as e:
            print(f"Error removing directory {VRPBTW_ROOT}: {e}. Aborting.")
            return
    # --- END FOLDER CLEANUP ---

    all_files = glob.glob(os.path.join(SOLOMON_ROOT, "**", "*.txt"), recursive=True)

    if not all_files:
        print(f"No Solomon VRPTW files found in {SOLOMON_ROOT}. Check your path!")
        return

    # Process each found file
    for filepath in all_files:
        try:
            max_vehicles, capacity, depot_data, original_customers = parse_solomon_file(
                filepath
            )

            filename_base = os.path.basename(filepath).split(".")[0]
            max_size = len(original_customers)

            # --- IMPLEMENTING THE CUSTOM FILTERING LOGIC (100 or 200 customers) ---
            target_sizes_for_file = []

            if max_size == 100:
                target_sizes_for_file = [s for s in TARGET_SIZES if s <= 100]
            elif max_size == 200:
                target_sizes_for_file = [150, 200]
            else:
                print(
                    f"Skipping file {filename_base}. Unexpected actual customer count: {max_size}. Must be 100 or 200."
                )
                continue

            # 2. Iterate over filtered target sizes and ratios
            for size in target_sizes_for_file:
                for linehaul_pct, _ in RATIOS:
                    # 3. Create and save the VRPBTW instance
                    create_vrpbtw_instance_csv(
                        max_vehicles,
                        depot_data,
                        original_customers,
                        size,
                        linehaul_pct,
                        filename_base,
                    )

        except ValueError as ve:
            print(f"Error processing file {filepath} (Structure Error): {ve}")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")


if __name__ == "__main__":
    generate_vrpbtw_datasets()
