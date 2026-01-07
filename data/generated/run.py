import os
import json
import generate  # Importing your provided generation script
import visualizer  # Importing your provided visualization script


def run():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Define all possible batches based on your config and script choices
    node_sizes = [10, 20, 50, 100, 200, 400]
    distributions = ["R", "C", "RC"]
    instances_per_batch = 5
    base_seed = 42
    backhaul_ratio = 0.5

    print(
        f"Starting generation for {len(node_sizes) * len(distributions)} total batches..."
    )

    for n in node_sizes:
        # Create directory for this node size
        output_dir = os.path.join("data", f"N{n}")
        os.makedirs(output_dir, exist_ok=True)

        for dist in distributions:
            print(f"--- Processing Batch: N={n}, Dist={dist} ---")

            for i in range(instances_per_batch):
                current_seed = base_seed + i

                # 1. Generate Data using imported create_instance function
                instance_data = generate.create_instance(
                    config, n, dist, backhaul_ratio, current_seed
                )

                # Construct filename
                filename = (
                    f"S{current_seed:03d}_N{n}_{dist}_R{int(backhaul_ratio * 100)}.json"
                )
                file_path = os.path.join(output_dir, filename)

                # Save the JSON file
                with open(file_path, "w") as f:
                    json.dump(instance_data, f, indent=4)

                # 2. Generate Visualization using imported visualize function
                # This function automatically mirrors the path into the 'img/' folder
                visualizer.visualize_vrp_instance(file_path)

                print(f"Successfully processed: {filename}")


if __name__ == "__main__":
    run()
