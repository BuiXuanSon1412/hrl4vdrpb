import numpy as np
from pathlib import Path
import json
from moo_algorithm.metric import cal_igd
import pandas as pd
from typing import List
import multiprocessing as mp


def selection_nondominate(solutions):
    """
    Select non-dominated solutions from a set of solutions.

    Args:
        solutions: numpy array of shape (n_solutions, n_objectives)

    Returns:
        numpy array of non-dominated solutions
    """
    n_solutions = solutions.shape[0]
    is_dominated = np.zeros(n_solutions, dtype=bool)

    for i in range(n_solutions):
        for j in range(n_solutions):
            if i == j:
                continue

            # Check if solution j dominates solution i
            # j dominates i if: j is better or equal in all objectives AND strictly better in at least one
            better_or_equal = np.all(solutions[j] <= solutions[i])
            strictly_better = np.any(solutions[j] < solutions[i])

            if better_or_equal and strictly_better:
                is_dominated[i] = True
                break

    return solutions[~is_dominated]


def find_nadir_point_igd(result_dir, size_dir, instance_file, algorithms):
    """
    Find the global nadir point and collect all algorithm data.
    Same as HV version but returns all history data for IGD calculation.
    """
    result_path = Path(result_dir)
    all_data = {}
    all_points = []

    for algo in algorithms:
        json_path = result_path / algo / size_dir / instance_file

        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history", {})
            if history:
                all_data[algo] = history
                for gen_data in history.values():
                    all_points.append(np.array(gen_data))
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

    if not all_points:
        raise ValueError("No data found for any algorithm!")

    combined_points = np.vstack(all_points)
    nadir_point = np.max(combined_points, axis=0)
    nadir_point[nadir_point == 0] = 1e-9

    return nadir_point, all_data


def cal_igd_one_dataset(result_dir, size_dir, instance_file, run_number=None):
    """
    Calculate IGD for all algorithms on one dataset/instance.

    Args:
        result_dir: Base result directory (e.g., "./result/raw/drone")
        size_dir: Size directory (e.g., "N100")
        instance_file: Instance filename (e.g., "S042_N100_RC_R50.json")
        run_number: Optional run number (1-5)

    Returns:
        Tuple of IGD values for all algorithms
    """
    if run_number is not None:
        result_dir = str(Path(result_dir) / str(run_number))

    algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]

    try:
        # 1. Load all data and find global nadir point
        nadir_point, all_data = find_nadir_point_igd(
            result_dir, size_dir, instance_file, algorithms
        )

        # 2. Collect final generation Pareto fronts for each algorithm
        all_pareto_data = {}
        for algo in algorithms:
            if algo in all_data:
                history = all_data[algo]
                final_gen_key = str(max([int(k) for k in history.keys()]))
                pareto_points = np.array(history[final_gen_key])
                all_pareto_data[algo] = pareto_points
            else:
                # If algorithm data not found, use empty array
                all_pareto_data[algo] = np.array([]).reshape(0, 2)

        # 3. Normalize all Pareto fronts by the global nadir point
        normalized_pareto_data = {}
        for algo, pareto_set in all_pareto_data.items():
            if pareto_set.shape[0] > 0:
                normalized_pareto_data[algo] = pareto_set / nadir_point
            else:
                normalized_pareto_data[algo] = pareto_set

        # 4. Combine all normalized solutions and find approximate Pareto front
        valid_solutions = [
            data for data in normalized_pareto_data.values() if data.shape[0] > 0
        ]

        if not valid_solutions:
            print(f"Warning: No valid solutions found for {instance_file}")
            return tuple([0.0] * len(algorithms))

        all_normalized_solutions = np.concatenate(valid_solutions, axis=0)
        approximate_front = selection_nondominate(all_normalized_solutions)

        # 5. Calculate IGD for each algorithm
        results_igd = {}
        for algo in algorithms:
            if normalized_pareto_data[algo].shape[0] > 0:
                igd_value = cal_igd(normalized_pareto_data[algo], approximate_front)
                results_igd[algo] = igd_value
            else:
                # No solutions for this algorithm
                results_igd[algo] = float("inf")

        return tuple(results_igd[algo] for algo in algorithms)

    except Exception as e:
        print(f"Error calculating IGD for {instance_file}: {e}")
        import traceback

        traceback.print_exc()
        return tuple([float("inf")] * len(algorithms))


def process_single_instance(args):
    """Worker function for parallel processing."""
    run_num, size_dir, dist, seed, ratio, result_dir = args
    num_customers = int(size_dir[1:])
    instance_file = f"S{seed:03d}_{size_dir}_{dist}_R{ratio}.json"

    print(f"Processing: Run {run_num} - {instance_file}")

    igd_values = cal_igd_one_dataset(
        result_dir=result_dir,
        size_dir=size_dir,
        instance_file=instance_file,
        run_number=run_num,
    )

    row_data = [run_num, num_customers, dist, seed, ratio]
    row_data.extend(igd_values)
    return row_data


def main():
    RESULT_DIR = "./result/drone/"
    OUTPUT_CSV = "./result/metric/drone/igd_metric.csv"
    NUM_PROCESSES = 12  # Adjust based on your CPU cores

    size_dirs = ["N100", "N200", "N400", "N1000"]
    distributions = ["R", "C", "RC"]
    instance_seeds = [42, 43, 44, 45, 46]
    run_numbers = [1, 2, 3, 4, 5]
    ratio = 50
    algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare all tasks
    tasks = [
        (run_num, size_dir, dist, seed, ratio, RESULT_DIR)
        for run_num in run_numbers
        for size_dir in size_dirs
        for dist in distributions
        for seed in instance_seeds
    ]

    # Process in parallel
    print(f"Processing {len(tasks)} instances using {NUM_PROCESSES} processes...")

    with mp.Pool(processes=NUM_PROCESSES) as pool:
        data = pool.map(process_single_instance, tasks)

    # Create DataFrame and save
    column_list: List[str] = ["run", "num_customers", "distribution", "seed", "ratio"]
    column_list.extend([f"{algo}_igd" for algo in algorithms])
    columns = pd.Index(column_list)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Completed! IGD data saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
