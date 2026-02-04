import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration ---
RUNS = range(1, 6)
BASE_RESULT_DIR = "./result/raw/drone/"
IMAGE_DIR = "./img/hv_boxplots"
ALGORITHM = "CIAGEA"
NORMALIZED_REF_POINT = np.array([1.0, 1.0])

NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEEDS = [42, 43, 44, 45, 46]

# Define colors for the 4 problem sizes
COLORS = ["#3D5E95", "#458855", "#c44145", "#e66c15"]


def get_algorithm_history(json_path):
    """Load history from a JSON result file."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("history", {})
    except Exception as e:
        print(e)
        return None


def find_global_nadir(base_path, num_nodes, distribution):
    """Find global nadir for a specific size and distribution."""
    all_points = []
    size_dir = f"N{num_nodes}"

    for seed in SEEDS:
        instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

        for run in RUNS:
            json_path = base_path / str(run) / ALGORITHM / size_dir / instance_file
            history = get_algorithm_history(json_path)

            if history:
                for gen_data in history.values():
                    all_points.append(np.array(gen_data))

    if not all_points:
        return None

    combined_points = np.vstack(all_points)
    global_nadir = np.max(combined_points, axis=0)
    global_nadir[global_nadir == 0] = 1e-9

    return global_nadir


def collect_final_hv_values(base_path, num_nodes, distribution, global_nadir):
    """
    Collect final generation HV values for all runs and seeds.

    Returns:
        List of final HV values (one per run×seed combination)
    """
    hv_values = []
    size_dir = f"N{num_nodes}"

    for seed in SEEDS:
        instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"

        for run in RUNS:
            json_path = base_path / str(run) / ALGORITHM / size_dir / instance_file
            history = get_algorithm_history(json_path)

            if history and len(history) > 0:
                # Get final generation
                final_gen = str(max([int(g) for g in history.keys()]))
                final_front = np.array(history[final_gen])

                # Normalize and calculate HV
                normalized_front = final_front / global_nadir
                hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)
                hv_values.append(hv)

    return hv_values


def plot_combined_distribution_boxplot():
    """
    Creates a single plot comparing distributions on the X-axis.
    Each distribution group contains 4 boxes for the problem sizes.
    """
    base_path = Path(BASE_RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    all_positions = []
    all_data = []
    color_mapping = []

    # Calculate spacing
    group_width = 0.8
    offset_step = group_width / len(NUM_NODES)

    for dist_idx, dist in enumerate(DISTRIBUTIONS):
        print(f"Processing distribution: {dist}")

        # Calculate base position for this distribution group (1, 2, 3...)
        base_pos = dist_idx + 1

        for size_idx, num_nodes in enumerate(NUM_NODES):
            # Calculate specific box position within the group
            pos = (
                base_pos
                - (group_width / 2)
                + (size_idx * offset_step)
                + (offset_step / 2)
            )

            try:
                global_nadir = find_global_nadir(base_path, num_nodes, dist)
                if global_nadir is None:
                    continue

                hv_values = collect_final_hv_values(
                    base_path, num_nodes, dist, global_nadir
                )

                if hv_values:
                    all_data.append(hv_values)
                    all_positions.append(pos)
                    color_mapping.append(COLORS[size_idx])
            except Exception as e:
                print(f"Error at {dist} N{num_nodes}: {e}")

    # Create the boxplot
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=offset_step * 0.8,
        patch_artist=True,
        showmeans=False,  # Mean removed as requested
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Apply colors to boxes
    for patch, color in zip(bp["boxes"], color_mapping):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Formatting X-axis
    ax.set_xticks(range(1, len(DISTRIBUTIONS) + 1))
    ax.set_xticklabels(DISTRIBUTIONS, fontsize=16)

    # ax.set_title(
    #    f"HV Performance by Distribution and Problem Size ({ALGORITHM})",
    #    fontsize=14,
    #    pad=20,
    # )
    ax.set_xlabel("Coordinate distribution", fontsize=16)
    ax.set_ylabel("Hypervolume", fontsize=16)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Create manual legend for the problem sizes
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            marker="s",
            markerfacecolor=COLORS[i],
            markersize=16,
            label=f"{NUM_NODES[i]} nodes",
        )
        for i in range(len(NUM_NODES))
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=4,
        frameon=True,
        columnspacing=3.0,
        fontsize=16,
    )
    plt.tight_layout()
    save_path = output_dir / f"{ALGORITHM}_HV_grouped_boxplot.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"Saved grouped plot to: {save_path}")


# Update main to call the new function
if __name__ == "__main__":
    plot_combined_distribution_boxplot()
