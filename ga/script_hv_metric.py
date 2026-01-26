import pandas as pd

# Load the dataset
df = pd.read_csv("./result/processed/drone/hv_metric.csv")

# Define the algorithm columns to calculate metrics for
algo_cols = [
    "MOEAD_hv",
    "NSGA_II_hv",
    "NSGA_III_hv",
    "PFG_MOEA_hv",
    "AGEA_hv",
    "CIAGEA_hv",
]

# Group by 'num_customers' and 'distribution', then calculate mean and std
hv_stats = df.groupby(["num_customers", "distribution"])[algo_cols].agg(["mean", "std"])

# Flatten the multi-index columns for easier viewing (e.g., MOEAD_hv_mean)
hv_stats.columns = [f"{col}_{stat}" for col, stat in hv_stats.columns]

# Reset index to turn groups back into columns
hv_stats = hv_stats.reset_index()

# Save the results to a CSV file
hv_stats.to_csv("./result/processed/drone/hv_mean_std.csv", index=False)

# Display the final summary table
print(hv_stats)
