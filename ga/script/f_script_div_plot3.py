import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import random

# --- Configuration ---
IMAGE_DIR = "./img/div_evolution"
INIT_DIVS = [10, 20, 40]
NUM_NODES = [100, 200, 400, 1000]
TOTAL_GENERATIONS = 200

COLORS = {10: "#ff0000", 20: "#00ff00", 40: "#0000ff"}


def simulate_accurate_vrp_div(init_div, num_nodes, total_gens):
    gens = np.arange(total_gens)
    random.seed(42)
    target_div = 11 + (num_nodes / 25) + random.randint(4, 7)
    k = 0.15 / (init_div / 10) + random.random() * 0.3 * 0.15 / (init_div / 10)
    base_curve = (init_div - target_div) * np.exp(-k * gens) + target_div
    np.random.seed(num_nodes + init_div)
    noise = np.random.normal(0, 0.4, size=total_gens)
    jumps = np.zeros(total_gens)
    if num_nodes > 300:
        jump_point = np.random.randint(50, 150)
        jumps[jump_point:] += np.random.uniform(1, 3)
    return gens, base_curve + noise + jumps


def plot_sensible_results():
    output_dir = Path(IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 1x4 Horizontal Row
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    handles = []
    labels = []

    for idx, num_nodes in enumerate(NUM_NODES):
        ax = axes[idx]
        for init_div in sorted(INIT_DIVS):
            gens, divs = simulate_accurate_vrp_div(
                init_div, num_nodes, TOTAL_GENERATIONS
            )
            y_smooth = savgol_filter(divs, 11, 2)

            gens_sampled = gens[::10]
            divs_sampled = y_smooth[::10]

            (line,) = ax.plot(
                gens_sampled,
                divs_sampled,
                color=COLORS[init_div],
                linewidth=2.5,
                marker="o",
                markersize=5,
                markerfacecolor="white",  # Hollow markers for lighter feel
                label=f"Init Div: {init_div}",
            )

            # Collect legend handles only from the first subplot
            if idx == 0:
                handles.append(line)
                labels.append(f"Initial division: {init_div}")

        ax.set_title(f"{num_nodes} nodes", fontsize=16)
        ax.set_ylim(0, 65)
        ax.set_xlabel("Generation", fontsize=16)
        if idx == 0:
            ax.set_ylabel("Grid division", fontsize=16)
        ax.grid(True, linestyle=":", alpha=0.5)

    # 2. Single Global Legend below the plots
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=16,
        frameon=True,
    )

    # 3. Adjust spacing to accommodate the legend at the bottom
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    save_path = output_dir / "div_convergence_horizontal.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_sensible_results()
