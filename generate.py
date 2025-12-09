from rl.problem import generate_solomon_like_vrpbtw
import os

CUSTOMER_SIZES = [20, 50, 100, 150]
OUTPUT_FOLDER = "data"
print("Generating VRPBTW Datasets")

dataset_df = None

for N in CUSTOMER_SIZES:
    # We will use T_max = 10.0 and speed_factor = 1.0 for consistency
    dataset_df = generate_solomon_like_vrpbtw(N, T_max=10.0, speed_factor=1.0)
    output_filename = f"VRPBTW_N{N}.csv"

    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    dataset_df.to_csv(output_path, index=False)

    print(
        f"- Successfully generated dataset: {output_filename} ({len(dataset_df)} nodes)."
    )

print("\n---")
print("- Dataset Preview")
if dataset_df is not None:
    print(dataset_df.head())
else:
    print("Error: No dataset is generated")
