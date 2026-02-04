import numpy as np
from pathlib import Path
import json
from moo_algorithm.metric import cal_hv
import pandas as pd
from typing import List


def find_nadir_point(result_dir, size_dir, instance_file, algorithms):
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


def cal_hv_one_dataset(result_dir, size_dir, instance_file, run_number=None):
    if run_number is not None:
        result_dir = str(Path(result_dir) / str(run_number))

    # FIXED: Added "IAGEA" to match the main loop's expectations
    algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]

    try:
        nadir_point, all_data = find_nadir_point(
            result_dir, size_dir, instance_file, algorithms
        )

        reference_point = np.array([1.0, 1.0])
        results_hv = {}

        for algo in algorithms:
            if algo in all_data:
                history = all_data[algo]
                final_gen_key = str(max([int(k) for k in history.keys()]))
                objectives_final_gen = np.array(history[final_gen_key])
                normalized_objectives = objectives_final_gen / nadir_point
                hv_value = cal_hv(normalized_objectives, reference_point)
                results_hv[algo] = hv_value
            else:
                results_hv[algo] = 0.0

        # FIXED: Returning a tuple of 7 values to match the 7 algorithms
        return tuple(results_hv[algo] for algo in algorithms)

    except Exception as e:
        print(f"Error calculating HV: {e}")
        return tuple([0.0] * len(algorithms))


def main():
    RESULT_DIR = "./result/raw/drone/"
    OUTPUT_CSV = "./result/processed/drone/hv_metric.csv"
    size_dirs = ["N100", "N200", "N400", "N1000"]
    distributions = ["R", "C", "RC"]
    instance_seeds = [42, 43, 44, 45, 46]
    run_numbers = [1, 2, 3, 4, 5]
    ratio = 50

    # Ensure these 7 match exactly with the return of cal_hv_one_dataset
    algorithms = ["MOEAD", "NSGA_II", "NSGA_III", "PFG_MOEA", "AGEA", "CIAGEA"]
    data = []

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for run_num in run_numbers:
        for size_dir in size_dirs:
            num_customers = int(size_dir[1:])
            for dist in distributions:
                for seed in instance_seeds:
                    instance_file = f"S{seed:03d}_{size_dir}_{dist}_R{ratio}.json"
                    hv_values = cal_hv_one_dataset(
                        result_dir=RESULT_DIR,
                        size_dir=size_dir,
                        instance_file=instance_file,
                        run_number=run_num,
                    )
                    row_data = [run_num, num_customers, dist, seed, ratio]
                    row_data.extend(hv_values)
                    data.append(row_data)

    column_list: List[str] = ["run", "num_customers", "distribution", "seed", "ratio"]
    column_list.extend([f"{algo}_hv" for algo in algorithms])

    columns = pd.Index(column_list)

    # Now the DataFrame constructor will be happy
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Completed! HV data saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()
