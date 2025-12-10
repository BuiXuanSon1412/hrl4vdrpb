import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def find_file_recursive(target_name):
    """Recursively search for <target_name> inside <root_dir>."""
    root = Path(__file__).resolve().parent.parent / "data" / "benchmark"
    for path in root.rglob("*"):
        if path.is_file() and path.name == target_name:
            return path
    return None


def visualize(filename):
    # Find file inside data/benchmark recursively
    file_path = find_file_recursive(filename)

    if file_path is None:
        print(f"File '{filename}' not found inside data/benchmark/")
        return

    # Load CSV
    df = pd.read_csv(file_path)

    # Depot = first row
    depot = df.iloc[0]
    customers = df.iloc[1:]

    depot_x = depot["XCOORD."]
    depot_y = depot["YCOORD."]

    cx = customers["XCOORD."]
    cy = customers["YCOORD."]

    # Standard plot with grid background
    plt.figure(figsize=(7, 7))

    plt.scatter(cx, cy, s=30, color="red", label="Customers")
    plt.scatter(depot_x, depot_y, s=150, color="black", marker="s", label="Depot")

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(filename)
    plt.legend()

    # Enable grid
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Build parallel saver path: data/... → visualizer/...

    root = Path(__file__).resolve().parent.parent  # project root
    relative = file_path.relative_to(root / "data")
    output_path = root / "visualizer" / relative
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <filename>")
        sys.exit(1)

    visualize(sys.argv[1])
