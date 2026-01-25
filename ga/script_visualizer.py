import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from moo_algorithm.metric import cal_hv

# --- Configuration ---
RUNS = range(1, 6)  # Runs 1 to 5
BASE_RESULT_DIR = "./result/drone/"
IMAGE_DIR = "./img"
FILES = [
    "S042_N200_C_R50.json",
    "S042_N200_R_R50.json",
    "S042_N200_RC_R50.json",
    "S043_N200_C_R50.json",
    "S043_N200_R_R50.json",
    "S043_N200_RC_R50.json",
    "S044_N200_C_R50.json",
    "S044_N200_R_R50.json",
    "S044_N200_RC_R50.json",
    "S045_N200_C_R50.json",
    "S045_N200_R_R50.json",
    "S045_N200_RC_R50.json",
    "S046_N200_C_R50.json",
    "S046_N200_R_R50.json",
    "S046_N200_RC_R50.json",
]
ALGORITHMS = ["NSGA_III", "NSGA_II", "MOEAD", "PFG_MOEA", "AGEA", "CIAGEA"]
NORMALIZED_REF_POINT = np.array([1.1, 1.1])


def get_algorithm_history(json_path):
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", {})


def plot_average_comparison():
    base_path = Path(BASE_RESULT_DIR)
    compare_img_dir = Path(IMAGE_DIR) / "compare_avg"

    for instance in FILES:
        size_dir = instance.split("_")[1]
        output_dir = compare_img_dir / size_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing Average Comparison for: {instance}")

        plt.figure(figsize=(10, 6))

        # 1. First Pass: Determine Global Nadir across all runs and all algorithms
        all_points_for_nadir = []
        for run in RUNS:
            for algo in ALGORITHMS:
                json_file = base_path / str(run) / algo / size_dir / instance
                history = get_algorithm_history(json_file)
                if history:
                    for gen_data in history.values():
                        all_points_for_nadir.append(np.array(gen_data))

        if not all_points_for_nadir:
            print(f"No data found for {instance}. Skipping.")
            continue

        combined_points = np.vstack(all_points_for_nadir)
        global_nadir = np.max(combined_points, axis=0)
        global_nadir[global_nadir == 0] = 1e-9

        # 2. Second Pass: Calculate HV per run and then average
        for algo in ALGORITHMS:
            run_hvs = []
            # We want to reset this for the logic, but we need to ensure
            # we have a valid X-axis for plotting
            current_algo_gens = None

            for run in RUNS:
                json_file = base_path / str(run) / algo / size_dir / instance
                history = get_algorithm_history(json_file)

                if history and len(history) > 0:
                    # Extract and sort generations
                    gens = sorted([int(g) for g in history.keys()])

                    current_run_hv = []
                    for g in gens:
                        front = np.array(history[str(g)])
                        normalized_front = front / global_nadir
                        hv = cal_hv(normalized_front, NORMALIZED_REF_POINT)
                        current_run_hv.append(hv)

                    run_hvs.append(current_run_hv)
                    # Sync the X-axis to the generations found in the data
                    if current_algo_gens is None:
                        current_algo_gens = gens

            # Only plot if we found data and the dimensions match
            if run_hvs and current_algo_gens:
                run_hvs_np = np.array(run_hvs)

                # Critical Check: Ensure all runs have the same number of generations
                # If they don't, we truncate to the shortest run to avoid shape errors
                min_len = min(len(r) for r in run_hvs)
                run_hvs_filtered = np.array([r[:min_len] for r in run_hvs])
                plot_x = current_algo_gens[:min_len]

                mean_hv = np.mean(run_hvs_filtered, axis=0)
                std_hv = np.std(run_hvs_filtered, axis=0)

                # Plotting with explicit matching dimensions
                (line,) = plt.plot(plot_x, mean_hv, label=algo, linewidth=2)
                plt.fill_between(
                    plot_x,
                    mean_hv - std_hv,
                    mean_hv + std_hv,
                    color=line.get_color(),
                    alpha=0.15,
                )

        # 3. Formatting
        plt.title(f"Average HV Convergence (5 Runs): {instance}")
        plt.xlabel("Generation")
        plt.ylabel("Mean Normalized Hypervolume")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = output_dir / f"AVG_{instance.replace('.json', '')}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved average plot to: {save_path}")


if __name__ == "__main__":
    plot_average_comparison()
