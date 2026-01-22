import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration (Synced with your files) ---
RESULT_DIR = "./result"
IMAGE_DIR = "./img"
FILES = ["S042_N1000_R_R50.json"]
ALGORITHMS = ["AGEA", "IAGEA"]
NORMALIZED_REF_POINT = np.array([1.1, 1.1])


def get_algorithm_history(json_path):
    """Loads the full history for an algorithm."""
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", {})


def plot_comparison():
    result_path = Path(RESULT_DIR)
    compare_img_dir = Path(IMAGE_DIR) / "compare"

    for instance in FILES:
        size_dir = instance.split("_")[1]

        output_dir = compare_img_dir / size_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(output_dir)
        print(f"Processing comparison for: {instance}")
        all_histories = {}
        all_points_for_nadir = []

        # 1. Collect data and determine Global Nadir
        for algo in ALGORITHMS:
            # Construct path based on your visualizer.py logic
            size_dir = instance.split("_")[1]
            json_file = result_path / algo / size_dir / instance

            history = get_algorithm_history(json_file)
            if history:
                all_histories[algo] = history
                # Collect all points from all generations to find the true worst case
                for gen_data in history.values():
                    all_points_for_nadir.append(np.array(gen_data))

        if not all_histories:
            print(f"No data found for {instance}. Skipping.")
            continue

        # Calculate Global Nadir (the worst value seen across ALL algos/generations)
        combined_points = np.vstack(all_points_for_nadir)
        global_nadir = np.max(combined_points, axis=0)
        global_nadir[global_nadir == 0] = 1e-9

        # 2. Plotting
        plt.figure(figsize=(10, 6))

        for algo, history in all_histories.items():
            gens = sorted([int(g) for g in history.keys()])
            hv_values = []

            for g in gens:
                front = np.array(history[str(g)])
                # Apply normalization from compare.py
                normalized_front = front / global_nadir
                # Calculate HV relative to [1.1, 1.1]
                hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)
                hv_values.append(hv)

            plt.plot(gens, hv_values, label=algo, linewidth=2)

        # 3. Formatting
        plt.title(f"Normalized HV Convergence: {instance}")
        plt.xlabel("Generation")
        plt.ylabel("Normalized Hypervolume (Ref: 1.1)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = output_dir / f"{instance.replace('.json', '')}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot to: {save_path}")


if __name__ == "__main__":
    plot_comparison()
