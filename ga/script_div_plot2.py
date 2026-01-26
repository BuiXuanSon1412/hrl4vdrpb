import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline  # Added for smoothing
from scipy.signal import savgol_filter

# --- Configuration ---
BASE_RESULT_DIR = "./result/raw/div/"
IMAGE_DIR = "./img/div_evolution"
INIT_DIVS = [10, 20, 40]
NUM_NODES = [100, 200, 400, 1000]
DISTRIBUTIONS = ["C", "R", "RC"]
SEED = 42

# Colors for different init_div values
COLORS = {
    10: "#3D5E95",  # Blue
    20: "#458855",  # Green
    40: "#A94145",  # Red
}


def load_div_history(base_path, init_div, num_nodes, distribution, seed):
    """Load div values across generations from a single JSON file."""
    size_dir = f"N{num_nodes}"
    instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"
    json_path = (
        base_path / f"INIT{init_div}" / "1" / "CIAGEA" / size_dir / instance_file
    )

    if not json_path.exists():
        print(f"Warning: File not found: {json_path}")
        return None, None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("history", {})
        if not history:
            return None, None

        generations, div_values = [], []
        for gen_key in sorted(history.keys(), key=int):
            gen_data = history[gen_key]
            if isinstance(gen_data, list) and len(gen_data) == 2:
                generations.append(int(gen_key))
                div_values.append(gen_data[1])
        return generations, div_values
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None, None


def collect_div_data_for_size(base_path, num_nodes):
    """Collect and average div evolution data for a given problem size."""
    init_div_data = {}
    for init_div in INIT_DIVS:
        all_div_sequences = []
        for dist in DISTRIBUTIONS:
            gens, divs = load_div_history(base_path, init_div, num_nodes, dist, SEED)
            if gens is not None and divs is not None:
                all_div_sequences.append((gens, divs))

        if all_div_sequences:
            min_len = min(len(divs) for _, divs in all_div_sequences)
            aligned_gens = all_div_sequences[0][0][:min_len]
            aligned_divs = np.array([divs[:min_len] for _, divs in all_div_sequences])
            mean_divs = np.mean(aligned_divs, axis=0)
            init_div_data[init_div] = (aligned_gens, mean_divs)
    return init_div_data


def plot_smooth_line(ax, x, y, label, color):
    """Enhanced Savitzky-Golay smoothing for cleaner plots."""
    data_len = len(y)

    if data_len > 15:
        # 1. Increase window_size for much heavier smoothing.
        # It must be ODD and less than the data length.
        window_size = 41

        # Ensure window_size isn't larger than our data
        if window_size >= data_len:
            window_size = data_len if data_len % 2 != 0 else data_len - 1

        # 2. Use polyorder=2 (Lower orders = smoother curves)
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=2)

        # Plot the clean line
        ax.plot(x, y_smooth, color=color, linewidth=3, label=label, alpha=1.0, zorder=3)

        # 3. Reduce scatter visibility or remove it to make the line pop
        # ax.scatter(x, y, color=color, s=5, alpha=0.15, zorder=2)
    else:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)


def plot_size_evolution(num_nodes):
    """Plot smoothed div evolution for a single problem size."""
    base_path = Path(BASE_RESULT_DIR)
    init_div_data = collect_div_data_for_size(base_path, num_nodes)
    fig, ax = plt.subplots(figsize=(8, 5))

    for init_div in INIT_DIVS:
        if init_div in init_div_data:
            gens, mean_divs = init_div_data[init_div]
            plot_smooth_line(
                ax, gens, mean_divs, f"init_div={init_div}", COLORS[init_div]
            )

    ax.set_title(f"N{num_nodes}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("The number of divisions (div)", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig, ax


def plot_all_sizes_comparison():
    """Create a 2x2 subplot comparing all 4 problem sizes with smoothing."""
    base_path = Path(BASE_RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, num_nodes in enumerate(NUM_NODES):
        ax = axes[idx]
        init_div_data = collect_div_data_for_size(base_path, num_nodes)
        for init_div in INIT_DIVS:
            if init_div in init_div_data:
                gens, mean_divs = init_div_data[init_div]
                plot_smooth_line(
                    ax, gens, mean_divs, f"init_div={init_div}", COLORS[init_div]
                )

        ax.set_title(f"N{num_nodes}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Division Value (div)", fontsize=12)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    save_path = output_dir / "div_evolution_comparison_smooth.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Generating smoothed plots...")
    plot_all_sizes_comparison()

    # Create individual smoothed plots
    output_dir = Path(IMAGE_DIR)
    for num_nodes in NUM_NODES:
        fig, ax = plot_size_evolution(num_nodes)
        save_path = output_dir / f"div_evolution_N{num_nodes}_smooth.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"All smoothed plots saved to {IMAGE_DIR}/")


if __name__ == "__main__":
    main()
