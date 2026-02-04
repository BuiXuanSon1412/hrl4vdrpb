import numpy as np
from pathlib import Path
import json
import pandas as pd
from typing import List, Tuple, Optional


def get_final_objectives(json_path: Path) -> np.ndarray:
    """
    Extract final generation objectives from a result JSON file.

    Args:
        json_path: Path to the JSON result file

    Returns:
        numpy array of shape (n_solutions, 2) containing [tardiness, cost] pairs
    """
    if not json_path.exists():
        return np.array([]).reshape(0, 2)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = data.get("history", {})
        if not history:
            return np.array([]).reshape(0, 2)

        # Get the last generation
        last_gen_key = str(max([int(k) for k in history.keys()]))

        # Get the objectives list directly
        gen_data = history[last_gen_key]
        objectives = np.array(gen_data)

        return objectives

    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return np.array([]).reshape(0, 2)


def calculate_batch_statistics(
    result_dir: str,
    num_nodes: int,
    distribution: str,
    seeds: List[int],
    runs: List[int],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate average and standard deviation of tardiness and cost for a batch.

    Args:
        result_dir: Base directory for results
        num_nodes: Number of customer nodes (10, 20, etc.)
        distribution: Distribution type (C, R, RC)
        seeds: List of instance seeds
        runs: List of run numbers

    Returns:
        Tuple of (avg_tardiness, std_tardiness, avg_cost, std_cost)
    """
    all_tardiness = []
    all_cost = []

    result_path = Path(result_dir)
    size_dir = f"N{num_nodes}"

    for run in runs:
        for seed in seeds:
            instance_file = f"S{seed:03d}_N{num_nodes}_{distribution}_R50.json"
            json_path = result_path / str(run) / "CIAGEA" / size_dir / instance_file

            objectives = get_final_objectives(json_path)

            if objectives.shape[0] > 0:
                # Extract tardiness (column 0) and cost (column 1)
                all_tardiness.extend(objectives[:, 0].tolist())
                all_cost.extend(objectives[:, 1].tolist())

    if not all_tardiness or not all_cost:
        return 0.0, 0.0, 0.0, 0.0

    avg_tardiness = np.mean(all_tardiness)
    std_tardiness = np.std(all_tardiness)
    avg_cost = np.mean(all_cost)
    std_cost = np.std(all_cost)

    return avg_tardiness.item(), std_tardiness.item(), avg_cost.item(), std_cost.item()


def main():
    # Configuration
    RESULT_DIR = "./result/drone/"
    OUTPUT_CSV = "./result/metric/drone/obj_metric.csv"

    num_nodes_list = [10, 20]
    distributions = ["C", "R", "RC"]
    seeds = [42, 43, 44, 45, 46]
    runs = [1, 2, 3, 4, 5]

    # Prepare output directory
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect statistics for each batch
    data = []

    for num_nodes in num_nodes_list:
        for dist in distributions:
            print(f"Processing batch: N{num_nodes}_{dist}")

            avg_tard, std_tard, avg_cost, std_cost = calculate_batch_statistics(
                result_dir=RESULT_DIR,
                num_nodes=num_nodes,
                distribution=dist,
                seeds=seeds,
                runs=runs,
            )

            batch_name = f"{num_nodes}{dist}"

            data.append(
                {
                    "batch": batch_name,
                    "num_nodes": num_nodes,
                    "distribution": dist,
                    "avg_tardiness": avg_tard,
                    "std_tardiness": std_tard,
                    "avg_cost": avg_cost,
                    "std_cost": std_cost,
                    "num_instances": len(seeds),
                    "num_runs": len(runs),
                }
            )

            print(f"  Avg Tardiness: {avg_tard:.4f} ± {std_tard:.4f}")
            print(f"  Avg Cost: {avg_cost:.2f} ± {std_cost:.2f}")

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCompleted! Objective statistics saved to '{OUTPUT_CSV}'")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
