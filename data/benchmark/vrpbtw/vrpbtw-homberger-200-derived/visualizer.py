import json
import matplotlib

# Use the 'Agg' backend to avoid needing a display/plt.show()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os


def visualize_vrp_data(file_path):
    # 1. Validation and Loading
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
        return

    # 2. Data Extraction
    depot_x = data["depot"]["x"]
    depot_y = data["depot"]["y"]

    linehaul_x, linehaul_y = [], []
    backhaul_x, backhaul_y = [], []

    for customer in data["customers"]:
        if customer["demand"] >= 0:
            linehaul_x.append(customer["x"])
            linehaul_y.append(customer["y"])
        else:
            backhaul_x.append(customer["x"])
            backhaul_y.append(customer["y"])

    # 3. Plotting
    plt.figure(figsize=(12, 10))
    plt.scatter(
        linehaul_x,
        linehaul_y,
        c="blue",
        marker="o",
        s=40,
        alpha=0.6,
        label="Linehaul (Delivery)",
    )
    plt.scatter(
        backhaul_x,
        backhaul_y,
        c="red",
        marker="^",
        s=40,
        alpha=0.6,
        label="Backhaul (Pickup)",
    )
    plt.scatter(
        depot_x,
        depot_y,
        c="green",
        marker="s",
        s=200,
        label="Depot",
        edgecolors="black",
    )

    plt.title(
        f"Instance: {data['instance']}\n({data['n_customers']} Customers)", fontsize=14
    )
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.annotate(
        "DEPOT",
        (depot_x, depot_y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
    )

    # 4. Save to img/ folder
    output_dir = "img"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Generate filename based on the instance name
    output_filename = f"{data['instance']}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path)
    plt.close()  # Clean up memory
    print(f"Visualization saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize VRPTW JSON and save as image."
    )
    parser.add_argument("path", help="Path to the JSON data file")
    args = parser.parse_args()

    visualize_vrp_data(args.path)
