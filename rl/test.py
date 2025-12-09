import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Dict
import os
from sklearn.cluster import KMeans

from problem import Customer, CustomerType
from environment import VehicleDroneRoutingEnv
from agent import HierarchicalAgent


def generate_clustered_customers(
    num_customers: int,
    num_clusters: int,
    map_size: float,
    cluster_std: float = 10.0,
    depot: Tuple[float, float] | None = None,
) -> List[Customer]:
    """
    Generate customers following a cluster distribution

    Args:
        num_customers: Total number of customers
        num_clusters: Number of clusters
        map_size: Size of the map
        cluster_std: Standard deviation of customer positions within clusters
        depot: Depot location (used to avoid placing clusters too close)

    Returns:
        List of Customer objects
    """
    if depot is None:
        depot = (map_size / 2, map_size / 2)

    # Generate cluster centers avoiding depot area
    cluster_centers = []
    min_distance_from_depot = map_size * 0.2  # Clusters at least 20% away from depot

    for _ in range(num_clusters):
        attempts = 0
        while attempts < 100:
            center_x = np.random.uniform(map_size * 0.1, map_size * 0.9)
            center_y = np.random.uniform(map_size * 0.1, map_size * 0.9)

            # Check distance from depot
            dist_from_depot = np.sqrt(
                (center_x - depot[0]) ** 2 + (center_y - depot[1]) ** 2
            )

            if dist_from_depot >= min_distance_from_depot:
                cluster_centers.append((center_x, center_y))
                break
            attempts += 1

        if attempts >= 100:
            # Fallback: place randomly
            cluster_centers.append(
                (np.random.uniform(0, map_size), np.random.uniform(0, map_size))
            )

    # Assign customers to clusters
    customers_per_cluster = num_customers // num_clusters
    remainder = num_customers % num_clusters

    customers = []
    num_linehaul = num_customers // 2

    for cluster_idx, (cx, cy) in enumerate(cluster_centers):
        # Determine number of customers in this cluster
        n_customers = customers_per_cluster
        if cluster_idx < remainder:
            n_customers += 1

        # Generate customers around cluster center
        for i in range(n_customers):
            customer_id = len(customers)

            # Generate position with normal distribution around cluster center
            x = np.clip(np.random.normal(cx, cluster_std), 0, map_size)
            y = np.clip(np.random.normal(cy, cluster_std), 0, map_size)

            # Assign customer type
            customer_type = (
                CustomerType.LINEHAUL
                if customer_id < num_linehaul
                else CustomerType.BACKHAUL
            )

            customers.append(
                Customer(
                    id=customer_id,
                    x=x,
                    y=y,
                    demand=np.random.uniform(5, 15),
                    time_window_start=np.random.uniform(0, 100),
                    time_window_end=np.random.uniform(100, 200),
                    customer_type=customer_type,
                )
            )

    return customers


def create_clustered_environment(
    num_customers: int = 100,
    num_clusters: int = 5,
    num_vehicles: int = 7,
    num_drones: int = 7,
    cluster_std: float = 10.0,
    map_size: float = 100.0,
) -> VehicleDroneRoutingEnv:
    """
    Create an environment with clustered customer distribution
    """
    env = VehicleDroneRoutingEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        num_drones=num_drones,
        map_size=map_size,
        normalize_rewards=True,
        cost_weight=0.5,
        satisfaction_weight=0.5,
    )

    # Override reset to use clustered customers
    original_reset = env.reset

    def clustered_reset(*, seed=None, options=None):
        state, info = original_reset(seed=seed, options=options)

        # Generate clustered customers
        env.customers = generate_clustered_customers(
            num_customers=num_customers,
            num_clusters=num_clusters,
            map_size=map_size,
            cluster_std=cluster_std,
            depot=env.depot,
        )

        return env._get_state(), info

    env.reset = clustered_reset
    return env


def test_on_clustered_data(
    checkpoint_path: str,
    num_test_episodes: int = 50,
    num_clusters_list: List[int] = [3, 5, 7, 10],
    cluster_std_list: List[float] = [5.0, 10.0, 15.0, 20.0],
    num_customers: int = 100,
    num_vehicles: int = 7,
    num_drones: int = 7,
) -> Dict:
    """
    Test trained model on various cluster configurations

    Args:
        checkpoint_path: Path to trained model checkpoint
        num_test_episodes: Number of test episodes per configuration
        num_clusters_list: List of cluster counts to test
        cluster_std_list: List of cluster standard deviations to test
        num_customers: Number of customers
        num_vehicles: Number of vehicles
        num_drones: Number of drones

    Returns:
        Dictionary containing test results
    """
    print("=" * 80)
    print("TESTING ON CLUSTERED CUSTOMER DISTRIBUTIONS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Episodes per Config: {num_test_episodes}")
    print(f"Customers: {num_customers}, Vehicles: {num_vehicles}, Drones: {num_drones}")
    print("=" * 80)

    # Create base environment
    base_env = VehicleDroneRoutingEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        num_drones=num_drones,
        normalize_rewards=True,
        cost_weight=0.5,
        satisfaction_weight=0.5,
    )

    # Load agent
    agent = HierarchicalAgent(base_env)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        agent.load(checkpoint_path)
        print("✓ Checkpoint loaded successfully!\n")
    else:
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return {}

    all_results = {}

    # Test 1: Varying number of clusters (fixed std)
    print("\n" + "=" * 80)
    print("TEST 1: Varying Number of Clusters (std=10.0)")
    print("=" * 80)

    cluster_results = []
    for num_clusters in num_clusters_list:
        print(f"\nTesting with {num_clusters} clusters...")

        env = create_clustered_environment(
            num_customers=num_customers,
            num_clusters=num_clusters,
            num_vehicles=num_vehicles,
            num_drones=num_drones,
            cluster_std=10.0,
        )
        agent.env = env

        results = run_test_episodes(agent, num_test_episodes)
        cluster_results.append({"num_clusters": num_clusters, "stats": results})

        print_results(results, f"{num_clusters} Clusters")

    all_results["varying_clusters"] = cluster_results

    # Test 2: Varying cluster spread (fixed num_clusters)
    print("\n" + "=" * 80)
    print("TEST 2: Varying Cluster Spread (5 clusters)")
    print("=" * 80)

    spread_results = []
    for cluster_std in cluster_std_list:
        print(f"\nTesting with cluster std={cluster_std}...")

        env = create_clustered_environment(
            num_customers=num_customers,
            num_clusters=5,
            num_vehicles=num_vehicles,
            num_drones=num_drones,
            cluster_std=cluster_std,
        )
        agent.env = env

        results = run_test_episodes(agent, num_test_episodes)
        spread_results.append({"cluster_std": cluster_std, "stats": results})

        print_results(results, f"Std={cluster_std}")

    all_results["varying_spread"] = spread_results

    # Test 3: Baseline (random distribution for comparison)
    print("\n" + "=" * 80)
    print("TEST 3: Baseline (Random Distribution)")
    print("=" * 80)

    base_env = VehicleDroneRoutingEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        num_drones=num_drones,
        normalize_rewards=True,
        cost_weight=0.5,
        satisfaction_weight=0.5,
    )
    agent.env = base_env

    baseline_results = run_test_episodes(agent, num_test_episodes)
    all_results["baseline"] = baseline_results

    print_results(baseline_results, "Random Distribution")

    return all_results


def run_test_episodes(agent: HierarchicalAgent, num_episodes: int) -> Dict:
    """
    Run multiple test episodes and collect statistics
    """
    results = {
        "rewards": [],
        "costs": [],
        "satisfactions": [],
        "service_rates": [],
        "customers_served": [],
    }

    for _ in range(num_episodes):
        result = agent.generate_solution(training=False)

        results["rewards"].append(result["total_reward"])
        results["costs"].append(result["total_cost"])
        results["satisfactions"].append(result["total_satisfaction"])
        results["service_rates"].append(result["service_rate"])
        results["customers_served"].append(result["customers_served"])

    # Compute statistics
    stats = {
        "mean_reward": np.mean(results["rewards"]),
        "std_reward": np.std(results["rewards"]),
        "mean_cost": np.mean(results["costs"]),
        "std_cost": np.std(results["costs"]),
        "mean_satisfaction": np.mean(results["satisfactions"]),
        "std_satisfaction": np.std(results["satisfactions"]),
        "mean_service_rate": np.mean(results["service_rates"]),
        "std_service_rate": np.std(results["service_rates"]),
        "mean_customers_served": np.mean(results["customers_served"]),
        "min_service_rate": np.min(results["service_rates"]),
        "max_service_rate": np.max(results["service_rates"]),
    }

    return stats


def print_results(stats: Dict, config_name: str):
    """
    Pretty print test results
    """
    print(f"\n📊 Results for {config_name}:")
    print(
        f"  Service Rate: {stats['mean_service_rate'] * 100:.1f}% ± {stats['std_service_rate'] * 100:.1f}%"
    )
    print(
        f"    (min: {stats['min_service_rate'] * 100:.1f}%, max: {stats['max_service_rate'] * 100:.1f}%)"
    )
    print(f"  Customers Served: {stats['mean_customers_served']:.1f}")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Cost: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
    print(
        f"  Mean Satisfaction: {stats['mean_satisfaction']:.2f} ± {stats['std_satisfaction']:.2f}"
    )


def visualize_clustered_solution(
    checkpoint_path: str,
    num_clusters: int = 5,
    cluster_std: float = 10.0,
    num_customers: int = 100,
    num_vehicles: int = 7,
    num_drones: int = 7,
):
    """
    Visualize a solution on clustered data
    """
    # Create environment
    env = create_clustered_environment(
        num_customers=num_customers,
        num_clusters=num_clusters,
        num_vehicles=num_vehicles,
        num_drones=num_drones,
        cluster_std=cluster_std,
    )

    # Load agent
    agent = HierarchicalAgent(env)
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return

    # Generate solution
    agent.env.reset()
    result = agent.generate_solution(training=False)

    # Identify clusters using K-means
    customer_positions = np.array([[c.x, c.y] for c in agent.env.customers])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(customer_positions)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Customer Clusters
    ax1 = axes[0]
    ax1.set_title(
        f"Customer Distribution ({num_clusters} Clusters)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")

    # Plot depot
    ax1.scatter(
        agent.env.depot[0],
        agent.env.depot[1],
        c="red",
        s=300,
        marker="s",
        label="Depot",
        zorder=5,
        edgecolors="black",
    )

    # Plot customers colored by cluster
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, num_clusters))
    for cluster_id in range(num_clusters):
        cluster_customers = [
            c
            for i, c in enumerate(agent.env.customers)
            if cluster_labels[i] == cluster_id
        ]

        xs = [c.x for c in cluster_customers]
        ys = [c.y for c in cluster_customers]

        ax1.scatter(
            xs,
            ys,
            c=[colors[cluster_id]],
            s=100,
            alpha=0.6,
            label=f"Cluster {cluster_id}",
            zorder=3,
        )

    # Plot cluster centers
    ax1.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="black",
        s=200,
        marker="X",
        edgecolors="white",
        linewidths=2,
        zorder=4,
        label="Cluster Centers",
    )

    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Solution Routes
    ax2 = axes[1]
    ax2.set_title("Vehicle Routes on Clustered Data", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")

    # Plot depot
    ax2.scatter(
        agent.env.depot[0],
        agent.env.depot[1],
        c="red",
        s=300,
        marker="s",
        label="Depot",
        zorder=5,
        edgecolors="black",
    )

    # Plot customers (served vs unserved)
    for customer in agent.env.customers:
        if customer.id in agent.env.served_customers:
            color = (
                "blue" if customer.customer_type == CustomerType.LINEHAUL else "green"
            )
            marker = "o" if customer.customer_type == CustomerType.LINEHAUL else "^"
            alpha = 0.7
        else:
            color = "red"
            marker = "x"
            alpha = 1.0

        ax2.scatter(
            customer.x,
            customer.y,
            c=color,
            s=100,
            marker=marker,
            alpha=alpha,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )

    # Plot vehicle routes
    colors_vehicles = ["purple", "orange", "brown", "pink", "cyan", "magenta", "yellow"]
    for i, vehicle in enumerate(agent.env.vehicles):
        if len(vehicle.route) > 0:
            route_x = [agent.env.depot[0]]
            route_y = [agent.env.depot[1]]

            for customer_id in vehicle.route:
                customer = agent.env.customers[customer_id]
                route_x.append(customer.x)
                route_y.append(customer.y)

            route_x.append(agent.env.depot[0])
            route_y.append(agent.env.depot[1])

            ax2.plot(
                route_x,
                route_y,
                c=colors_vehicles[i % len(colors_vehicles)],
                linewidth=2.5,
                alpha=0.7,
                label=f"Vehicle {i}",
                marker="o",
                markersize=4,
            )

    # Create legend
    linehaul_patch = mpatches.Patch(color="blue", label="Linehaul (Served)")
    backhaul_patch = mpatches.Patch(color="green", label="Backhaul (Served)")
    unserved_patch = mpatches.Patch(color="red", label="Unserved")

    handles, labels = ax2.get_legend_handles_labels()
    handles.extend([linehaul_patch, backhaul_patch, unserved_patch])
    ax2.legend(handles=handles, loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = (
        f"Clusters: {num_clusters} | Std: {cluster_std}\n"
        f"Service Rate: {result['service_rate'] * 100:.1f}% "
        f"({result['customers_served']}/{agent.env.num_customers})\n"
        f"Cost: {result['total_cost']:.1f} | "
        f"Satisfaction: {result['total_satisfaction']:.2f}"
    )

    fig.text(
        0.5,
        0.02,
        metrics_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    # Save figure
    img_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "img",
        f"clustered_solution_c{num_clusters}_s{int(cluster_std)}.png",
    )
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=150, bbox_inches="tight")

    print(f"✓ Visualization saved to '{img_path}'")
    plt.show()


def generate_comparison_plots(results: Dict):
    """
    Generate comparison plots for different cluster configurations
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Service Rate vs Number of Clusters
    ax1 = axes[0, 0]
    cluster_data = results["varying_clusters"]
    num_clusters = [d["num_clusters"] for d in cluster_data]
    service_rates = [d["stats"]["mean_service_rate"] * 100 for d in cluster_data]
    service_stds = [d["stats"]["std_service_rate"] * 100 for d in cluster_data]

    ax1.errorbar(
        num_clusters,
        service_rates,
        yerr=service_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
    )
    ax1.axhline(
        y=results["baseline"]["mean_service_rate"] * 100,
        color="r",
        linestyle="--",
        label="Baseline (Random)",
    )
    ax1.set_xlabel("Number of Clusters", fontsize=12)
    ax1.set_ylabel("Service Rate (%)", fontsize=12)
    ax1.set_title("Service Rate vs Number of Clusters", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Cost vs Number of Clusters
    ax2 = axes[0, 1]
    costs = [d["stats"]["mean_cost"] for d in cluster_data]
    cost_stds = [d["stats"]["std_cost"] for d in cluster_data]

    ax2.errorbar(
        num_clusters,
        costs,
        yerr=cost_stds,
        marker="s",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="orange",
    )
    ax2.axhline(
        y=results["baseline"]["mean_cost"],
        color="r",
        linestyle="--",
        label="Baseline (Random)",
    )
    ax2.set_xlabel("Number of Clusters", fontsize=12)
    ax2.set_ylabel("Mean Cost", fontsize=12)
    ax2.set_title("Cost vs Number of Clusters", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Service Rate vs Cluster Spread
    ax3 = axes[1, 0]
    spread_data = results["varying_spread"]
    cluster_stds = [d["cluster_std"] for d in spread_data]
    service_rates_spread = [d["stats"]["mean_service_rate"] * 100 for d in spread_data]
    service_stds_spread = [d["stats"]["std_service_rate"] * 100 for d in spread_data]

    ax3.errorbar(
        cluster_stds,
        service_rates_spread,
        yerr=service_stds_spread,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="green",
    )
    ax3.axhline(
        y=results["baseline"]["mean_service_rate"] * 100,
        color="r",
        linestyle="--",
        label="Baseline (Random)",
    )
    ax3.set_xlabel("Cluster Standard Deviation", fontsize=12)
    ax3.set_ylabel("Service Rate (%)", fontsize=12)
    ax3.set_title("Service Rate vs Cluster Spread", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Satisfaction vs Cluster Spread
    ax4 = axes[1, 1]
    satisfactions = [d["stats"]["mean_satisfaction"] for d in spread_data]
    satisfaction_stds = [d["stats"]["std_satisfaction"] for d in spread_data]

    ax4.errorbar(
        cluster_stds,
        satisfactions,
        yerr=satisfaction_stds,
        marker="d",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="purple",
    )
    ax4.axhline(
        y=results["baseline"]["mean_satisfaction"],
        color="r",
        linestyle="--",
        label="Baseline (Random)",
    )
    ax4.set_xlabel("Cluster Standard Deviation", fontsize=12)
    ax4.set_ylabel("Mean Satisfaction", fontsize=12)
    ax4.set_title("Satisfaction vs Cluster Spread", fontsize=13, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    # Save figure
    img_path = os.path.join(
        os.path.dirname(__file__), "..", "img", "cluster_comparison_plots.png"
    )
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=150, bbox_inches="tight")

    print(f"✓ Comparison plots saved to '{img_path}'")
    plt.show()


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    checkpoint_path = "hrl_vdrpb_checkpoint.pt"

    print("\n" + "=" * 80)
    print("CLUSTER DISTRIBUTION TESTING")
    print("=" * 80 + "\n")

    # Run comprehensive tests
    results = test_on_clustered_data(
        checkpoint_path=checkpoint_path,
        num_test_episodes=50,
        num_clusters_list=[3, 5, 7, 10],
        cluster_std_list=[5.0, 10.0, 15.0, 20.0],
        num_customers=100,
        num_vehicles=7,
        num_drones=7,
    )

    if results:
        # Generate comparison plots
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 80)
        generate_comparison_plots(results)

        # Visualize specific configurations
        print("\n" + "=" * 80)
        print("GENERATING SOLUTION VISUALIZATIONS")
        print("=" * 80)

        # Visualize tight clusters
        print("\nVisualizing tight clusters (std=5.0)...")
        visualize_clustered_solution(
            checkpoint_path=checkpoint_path,
            num_clusters=5,
            cluster_std=5.0,
        )

        # Visualize spread clusters
        print("\nVisualizing spread clusters (std=20.0)...")
        visualize_clustered_solution(
            checkpoint_path=checkpoint_path,
            num_clusters=5,
            cluster_std=20.0,
        )

        # Visualize many clusters
        print("\nVisualizing many clusters (10 clusters)...")
        visualize_clustered_solution(
            checkpoint_path=checkpoint_path,
            num_clusters=10,
            cluster_std=10.0,
        )

    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
