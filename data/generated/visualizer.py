import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def visualize_vrp_instance(file_path):
    # 1. Load the instance data
    with open(file_path, "r") as f:
        data = json.load(f)

    # 2. Determine Output Path
    # Example: data/N10/S046_N10_RC_R50.json -> img/N10/S046_N10_RC_R50.png
    path_parts = file_path.split(os.sep)
    # Replace 'data' with 'img' and change extension to .png
    if "data" in path_parts:
        path_parts[path_parts.index("data")] = "img"

    output_image = os.path.join(*path_parts).replace(".json", ".png")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_image), exist_ok=True)

    # 3. Extract Depot and Node information
    depot_coord = data["Config"]["Depot"]["coord"]
    nodes = data["Nodes"]

    lh_coords = np.array([n["coord"] for n in nodes if n["type"] == "LINEHAUL"])
    bh_coords = np.array([n["coord"] for n in nodes if n["type"] == "BACKHAUL"])

    # 4. Plotting
    plt.figure(figsize=(12, 10))

    # Depot (Red Square)
    plt.scatter(
        depot_coord[0],
        depot_coord[1],
        c="black",
        marker="s",
        s=120,
        label="Depot",
        zorder=5,
    )

    # Linehaul (Blue Circles)
    if lh_coords.size > 0:
        plt.scatter(
            lh_coords[:, 0],
            lh_coords[:, 1],
            c="blue",
            marker="o",
            s=40,
            alpha=0.6,
            label="Linehaul",
            zorder=4,
        )

    # Backhaul (Green Triangles)
    if bh_coords.size > 0:
        plt.scatter(
            bh_coords[:, 0],
            bh_coords[:, 1],
            c="red",
            marker="^",
            s=40,
            alpha=0.6,
            label="Backhaul",
            zorder=4,
        )

    plt.title(f"Instance: {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save to the new img/ directory
    plt.savefig(output_image)
    plt.close()
    print(f"Visualization saved to: {output_image}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_vrp_instance(sys.argv[1])
    else:
        print("Usage: python script_name.py data/N10/S046_N10_RC_R50.json")
