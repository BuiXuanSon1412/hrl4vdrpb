import pandas as pd

# Load the IGD dataset
df_igd = pd.read_csv("./result/processed/drone/igd_metric.csv")

# Identify algorithm columns for IGD
algo_cols_igd = [col for col in df_igd.columns if col.endswith("_igd")]

# Group by 'num_customers' and 'distribution', then calculate mean and std
igd_stats = df_igd.groupby(["num_customers", "distribution"])[algo_cols_igd].agg(
    ["mean", "std"]
)

# Flatten the multi-index columns
igd_stats.columns = [f"{col}_{stat}" for col, stat in igd_stats.columns]

# Reset index
igd_stats = igd_stats.reset_index()

# Save the results to a CSV file
igd_stats.to_csv("./result/processed/drone/igd_mean_dev.csv", index=False)

# Display the summary
print(igd_stats)
