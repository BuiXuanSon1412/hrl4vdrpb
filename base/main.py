import torch
import numpy as np
from typing import Dict, Tuple
import os


from problem import CustomerType
from environment import VehicleDroneRoutingEnv
from agent import HierarchicalAgent


def train_hrl_agent(
    num_episodes: int = 1000,
    batch_size: int = 32,
    eval_interval: int = 50,
    save_path: str = "hrl_checkpoint.pt",
):
    """
    Main training loop with improved monitoring
    """
    # Create environment and agent
    env = VehicleDroneRoutingEnv(num_customers=100, num_vehicles=7, num_drones=7)

    agent = HierarchicalAgent(env)

    # Training metrics
    episode_rewards = []
    episode_costs = []
    episode_satisfactions = []
    service_rates = []

    print("Starting HRL Training...")
    print(
        f"Environment: {env.num_customers} customers, {env.num_vehicles} vehicles, {env.num_drones} drones"
    )
    print("=" * 80)

    for episode in range(num_episodes):
        # Generate solution (collect experience)
        result = agent.generate_solution(training=True)

        episode_rewards.append(result["total_reward"])
        episode_costs.append(result["total_cost"])
        episode_satisfactions.append(result["total_satisfaction"])
        service_rates.append(result["service_rate"])

        # Update exploration parameters
        agent.update_exploration(episode, num_episodes)

        # Train policies
        if episode > 0 and episode % 5 == 0:
            losses = agent.train_step(batch_size)

            if losses and episode % 10 == 0:
                print(
                    f"Ep {episode}: "
                    f"Served={result['customers_served']}/{env.num_customers} "
                    f"({result['service_rate'] * 100:.1f}%), "
                    f"Reward={result['total_reward']:.2f}, "
                    f"Cost={result['total_cost']:.2f}, "
                    f"Sat={result['total_satisfaction']:.2f}, "
                    f"ε={agent.epsilon:.3f}, "
                    f"T={agent.temperature:.2f}"
                )

        # Periodic evaluation
        if episode > 0 and episode % eval_interval == 0:
            print("\n" + "=" * 80)
            print(f"EVALUATION at Episode {episode}")
            print("=" * 80)

            # Run evaluation episodes
            eval_rewards = []
            eval_costs = []
            eval_satisfactions = []
            eval_service_rates = []

            for _ in range(10):
                eval_result = agent.generate_solution(training=False)
                eval_rewards.append(eval_result["total_reward"])
                eval_costs.append(eval_result["total_cost"])
                eval_satisfactions.append(eval_result["total_satisfaction"])
                eval_service_rates.append(eval_result["service_rate"])

            print(
                f"Avg Service Rate: {np.mean(eval_service_rates) * 100:.1f}% ± {np.std(eval_service_rates) * 100:.1f}%"
            )
            print(
                f"Avg Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}"
            )
            print(f"Avg Cost: {np.mean(eval_costs):.2f} ± {np.std(eval_costs):.2f}")
            print(
                f"Avg Satisfaction: {np.mean(eval_satisfactions):.2f} ± {np.std(eval_satisfactions):.2f}"
            )

            # Check if service rate is improving
            if np.mean(eval_service_rates) < 0.9:
                print(
                    f"⚠️  WARNING: Low service rate ({np.mean(eval_service_rates) * 100:.1f}%)"
                )
            else:
                print(f"✓ Good service rate!")

            print("=" * 80 + "\n")

            # Save checkpoint
            agent.save(save_path)
            print(f"Model saved to {save_path}\n")

        # Clear buffers periodically
        if episode > 0 and episode % 100 == 0:
            agent.clear_buffers()

            # Print training progress summary
            recent_service_rates = service_rates[-100:]
            print(
                f"\n📊 Last 100 episodes avg service rate: {np.mean(recent_service_rates) * 100:.1f}%"
            )

    print("\nTraining completed!")

    # Final statistics
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(
        f"Final avg service rate (last 100 eps): {np.mean(service_rates[-100:]) * 100:.1f}%"
    )
    print(f"Best service rate achieved: {max(service_rates) * 100:.1f}%")
    print(f"Episodes with 100% service: {sum(1 for sr in service_rates if sr >= 1.0)}")

    return agent, {
        "rewards": episode_rewards,
        "costs": episode_costs,
        "satisfactions": episode_satisfactions,
        "service_rates": service_rates,
    }


def evaluate_agent(
    agent: HierarchicalAgent, num_episodes: int = 100
) -> Tuple[Dict, Dict]:
    """
    Comprehensive evaluation of trained agent
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
        "avg_served": np.mean(results["customers_served"]),
    }

    return stats, results


def visualize_solution(agent: HierarchicalAgent):
    """
    Visualize a single solution
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available for visualization")
        return

    # Generate solution
    agent.env.reset()
    result = agent.generate_solution(training=False)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Routes
    ax1.set_title("Vehicle and Drone Routes", fontsize=14, fontweight="bold")
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
    )

    # Plot customers
    for customer in agent.env.customers:
        if customer.id in agent.env.served_customers:
            # Served customers
            color = (
                "blue" if customer.customer_type == CustomerType.LINEHAUL else "green"
            )
            marker = "o" if customer.customer_type == CustomerType.LINEHAUL else "^"
        else:
            # Unserved customers (in red)
            color = "red"
            marker = "x"

        ax1.scatter(customer.x, customer.y, c=color, s=100, marker=marker, zorder=3)
        ax1.text(customer.x + 2, customer.y + 2, f"{customer.id}", fontsize=8)

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

            ax1.plot(
                route_x,
                route_y,
                c=colors_vehicles[i % len(colors_vehicles)],
                linewidth=2,
                alpha=0.6,
                label=f"Vehicle {i}",
            )

    # Create legend
    linehaul_patch = mpatches.Patch(color="blue", label="Linehaul (Served)")
    backhaul_patch = mpatches.Patch(color="green", label="Backhaul (Served)")
    unserved_patch = mpatches.Patch(color="red", label="Unserved")
    ax1.legend(
        handles=[linehaul_patch, backhaul_patch, unserved_patch], loc="upper right"
    )
    ax1.grid(True, alpha=0.3)

    # Plot 2: Metrics
    ax2.set_title("Solution Metrics", fontsize=14, fontweight="bold")
    ax2.axis("off")

    service_rate = result["service_rate"] * 100

    metrics_text = f"""
    SOLUTION QUALITY
    {"=" * 40}
    
    Service Rate: {service_rate:.1f}%
    Customers Served: {result["customers_served"]} / {agent.env.num_customers}
    
    Total Reward: {result["total_reward"]:.2f}
    Total Cost: {result["total_cost"]:.2f}
    Total Satisfaction: {result["total_satisfaction"]:.2f}
    Avg Satisfaction: {result["total_satisfaction"] / max(1, result["customers_served"]):.3f}
    
    VEHICLE UTILIZATION
    {"=" * 40}
    """

    for i, vehicle in enumerate(agent.env.vehicles):
        metrics_text += f"\nVehicle {i}: {len(vehicle.route)} customers"
        metrics_text += f" (Load: {vehicle.current_load:.1f}/{vehicle.capacity})"

    ax2.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()
    img_path = os.path.join(
        os.path.dirname(__file__), "..", "img", "hrl_solution_visualization.png"
    )
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(os.path.abspath(img_path), dpi=150, bbox_inches="tight")

    print("Visualization saved to 'hrl_solution_visualization.png'")
    plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 80)
    print("HIERARCHICAL RL FOR VEHICLE-DRONE ROUTING WITH BACKHAULS")
    print("=" * 80)
    print()

    # Train agent
    agent, training_history = train_hrl_agent(
        num_episodes=500,
        batch_size=32,
        eval_interval=50,
        save_path="hrl_vdrpb_checkpoint.pt",
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Final evaluation
    stats, results = evaluate_agent(agent, num_episodes=100)

    print("\nFinal Performance Statistics:")
    print(
        f"  Service Rate: {stats['mean_service_rate'] * 100:.1f}% ± {stats['std_service_rate'] * 100:.1f}%"
    )
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Cost: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
    print(
        f"  Mean Satisfaction: {stats['mean_satisfaction']:.2f} ± {stats['std_satisfaction']:.2f}"
    )
    print(
        f"  Avg Customers Served: {stats['avg_served']:.1f} / {agent.env.num_customers}"
    )

    # Visualize a solution
    print("\nGenerating visualization...")
    visualize_solution(agent)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
