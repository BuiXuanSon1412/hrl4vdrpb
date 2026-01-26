import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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
    """
    Load div values across generations from a single JSON file.

    Args:
        base_path: Base result directory
        init_div: Initial division value (10, 20, or 40)
        num_nodes: Number of nodes (100, 200, 400, 1000)
        distribution: Distribution type (C, R, RC)
        seed: Instance seed

    Returns:
        Tuple of (generations, div_values) or (None, None) if file not found
    """
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

        # Extract generations and div values
        generations = []
        div_values = []

        for gen_key in sorted(history.keys(), key=int):
            gen_data = history[gen_key]
            # gen_data is [pareto_front, div_value]
            if isinstance(gen_data, list) and len(gen_data) == 2:
                generations.append(int(gen_key))
                div_values.append(gen_data[1])  # div is the second element

        return generations, div_values

    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None, None


def collect_div_data_for_size(base_path, num_nodes):
    """
    Collect div evolution data for all distributions at a given problem size.

    Args:
        base_path: Base result directory
        num_nodes: Number of nodes

    Returns:
        Dictionary mapping init_div to (generations, avg_div_values)
    """
    init_div_data = {}
    for init_div in INIT_DIVS:
        all_div_sequences = []

        for dist in DISTRIBUTIONS:
            gens, divs = load_div_history(base_path, init_div, num_nodes, dist, SEED)

            if gens is not None and divs is not None:
                all_div_sequences.append((gens, divs))

        if all_div_sequences:
            # Find minimum length across all distributions
            min_len = min(len(divs) for _, divs in all_div_sequences)

            # Align all sequences to same length
            aligned_gens = all_div_sequences[0][0][:min_len]
            aligned_divs = np.array([divs[:min_len] for _, divs in all_div_sequences])

            # Calculate mean across distributions
            mean_divs = np.mean(aligned_divs, axis=0)

            init_div_data[init_div] = (aligned_gens, mean_divs)

    return init_div_data


def plot_size_evolution(num_nodes):
    """
    Plot div evolution for a single problem size.

    Args:
        num_nodes: Number of nodes

    Returns:
        Figure and axis objects for subplot integration
    """
    base_path = Path(BASE_RESULT_DIR)

    # Collect data for all init_div values
    init_div_data = collect_div_data_for_size(base_path, num_nodes)

    fig, ax = plt.subplots(figsize=(8, 5))

    for init_div in INIT_DIVS:
        if init_div in init_div_data and init_div_data[init_div]:
            gens, mean_divs = init_div_data[init_div]

            ax.plot(
                gens,
                mean_divs,
                color=COLORS[init_div],
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"init_div={init_div}",
            )

    ax.set_title(f"N{num_nodes}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("The number of divisions (div)", fontsize=11)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig, ax


def plot_all_sizes_comparison():
    """
    Create a 2x2 subplot comparing all 4 problem sizes.
    """
    base_path = Path(BASE_RESULT_DIR)
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Division Evolution Plots")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, num_nodes in enumerate(NUM_NODES):
        ax = axes[idx]
        print(f"\nProcessing N{num_nodes}...")

        # Collect data for all init_div values
        init_div_data = collect_div_data_for_size(base_path, num_nodes)

        for init_div in INIT_DIVS:
            if init_div in init_div_data and init_div_data[init_div]:
                gens, mean_divs = init_div_data[init_div]

                ax.plot(
                    gens,
                    mean_divs,
                    color=COLORS[init_div],
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    label=f"init_div={init_div}",
                    alpha=0.9,
                )

                print(f"  init_div={init_div}: Final div = {mean_divs[-1]:.2f}")

        ax.set_title(f"N{num_nodes}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Division Value (div)", fontsize=12)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Overall title
    # fig.suptitle(
    #    "Evolution of Grid Division Parameter (div) Across Generations\n"
    #    "CIAGEA Algorithm - Averaged over C, R, RC Distributions",
    #    fontsize=16,
    #    fontweight="bold",
    #    y=0.995,
    # )

    plt.tight_layout(rect=(0, 0, 1, 0.985))

    # Save
    save_path = output_dir / "div_evolution_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'=' * 80}")
    print(f"Saved combined plot to: {save_path}")
    print(f"{'=' * 80}\n")


def plot_individual_sizes():
    """
    Create individual plots for each problem size.
    """
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for num_nodes in NUM_NODES:
        print(f"Creating individual plot for N{num_nodes}...")

        fig, ax = plot_size_evolution(num_nodes)

        save_path = output_dir / f"div_evolution_N{num_nodes}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Saved to: {save_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Division Evolution Analysis for CIAGEA")
    print("=" * 80 + "\n")

    # Create combined comparison plot
    plot_all_sizes_comparison()

    # Create individual plots (optional)
    print("\nCreating individual plots...")
    plot_individual_sizes()

    print("\n" + "=" * 80)
    print(f"All plots saved to {IMAGE_DIR}/")
    print("  - Combined plot: div_evolution_comparison.png")
    print("  - Individual plots: div_evolution_N100.png, N200.png, etc.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
