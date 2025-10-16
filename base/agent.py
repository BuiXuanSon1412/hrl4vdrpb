import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from environment import VehicleDroneRoutingEnv
from module import HighLevelPolicy, LowLevelPolicy


class HierarchicalAgent:
    """
    Hierarchical RL agent combining high-level and low-level policies
    """

    def __init__(
        self,
        env: VehicleDroneRoutingEnv,
        high_level_lr: float = 3e-4,
        low_level_lr: float = 3e-4,
        gamma: float = 0.99,
    ):
        self.env = env
        self.gamma = gamma

        # High-level policy
        assert env.observation_space.shape is not None, (
            "Observation space shape cannot be None"
        )
        state_dim = env.observation_space.shape[0]
        self.high_level_policy = HighLevelPolicy(
            state_dim=state_dim, num_vehicles=env.num_vehicles
        )
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(), lr=high_level_lr
        )

        # Low-level policy
        self.low_level_policy = LowLevelPolicy(
            customer_dim=7, vehicle_dim=6, num_customers=env.num_customers
        )
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(), lr=low_level_lr
        )

        # Experience buffers
        self.high_level_buffer = []
        self.low_level_buffer = []

    def select_vehicle(self, state: np.ndarray, training: bool = True) -> int:
        """High-level decision: select which vehicle to plan"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            vehicle_logits, _ = self.high_level_policy(state_tensor)

            if training:
                dist = torch.distributions.Categorical(logits=vehicle_logits)
                vehicle_id = int(dist.sample().item())
            else:
                vehicle_id = int(vehicle_logits.argmax(dim=1).item())

        return vehicle_id

    def select_customer(
        self,
        customer_features: np.ndarray,
        vehicle_context: np.ndarray,
        valid_actions: List[int],
        training: bool = True,
    ) -> Tuple[int, bool]:
        """Low-level decision: select customer and drone usage"""

        # Create mask for invalid actions
        mask = torch.ones(self.env.num_customers, dtype=torch.bool)
        mask[valid_actions] = False

        with torch.no_grad():
            customer_tensor = torch.FloatTensor(customer_features).unsqueeze(0)
            context_tensor = torch.FloatTensor(vehicle_context).unsqueeze(0)

            customer_logits, drone_logits, _ = self.low_level_policy(
                customer_tensor, context_tensor, mask.unsqueeze(0)
            )

            if training:
                # Sample from distribution
                customer_dist = torch.distributions.Categorical(logits=customer_logits)
                customer_id = int(customer_dist.sample().item())

                drone_dist = torch.distributions.Categorical(logits=drone_logits)
                use_drone = bool(drone_dist.sample().item())
            else:
                # Greedy selection
                customer_id = int(customer_logits.argmax(dim=1).item())
                use_drone = bool(drone_logits.argmax(dim=1).item())

        return customer_id, use_drone

    def generate_solution(self, training: bool = True) -> Dict:
        """
        Auto-regressive solution generation
        """
        state, _ = self.env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < self.env.num_customers * 2:
            # High-level: select vehicle
            vehicle_id = self.select_vehicle(state, training)

            # Get valid actions for this vehicle
            valid_actions = self.env.get_valid_actions(vehicle_id)

            # Initialize these variables before the if-else block
            customer_features: Optional[np.ndarray] = None
            vehicle_context: Optional[np.ndarray] = None

            if len(valid_actions) == 1 and valid_actions[0] == -1:
                # Only depot return available
                customer_id = -1
                use_drone = False
            else:
                # Extract features for low-level policy
                customer_features = self._extract_customer_features(state)
                vehicle_context = self._extract_vehicle_context(state, vehicle_id)

                # Low-level: select customer and drone usage
                customer_id, use_drone = self.select_customer(
                    customer_features, vehicle_context, valid_actions, training
                )

            # Execute action
            action = {
                "vehicle_id": vehicle_id,
                "customer_id": customer_id,
                "use_drone": use_drone,
                "drone_id": 0 if use_drone else None,
            }

            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Store experience
            if training:
                self.high_level_buffer.append(
                    {
                        "state": state,
                        "vehicle_id": vehicle_id,
                        "reward": reward,
                        "next_state": next_state,
                        "done": terminated or truncated,
                    }
                )

                self.low_level_buffer.append(
                    {
                        "customer_features": customer_features
                        if customer_id != -1
                        else None,
                        "vehicle_context": vehicle_context
                        if customer_id != -1
                        else None,
                        "customer_id": customer_id,
                        "use_drone": use_drone,
                        "reward": reward,
                        "done": terminated or truncated,
                    }
                )

            episode_reward += reward
            state = next_state
            steps += 1

        return {
            "total_reward": episode_reward,
            "total_cost": self.env.total_cost,
            "total_satisfaction": self.env.total_satisfaction,
            "steps": steps,
        }

    def _extract_customer_features(self, state: np.ndarray) -> np.ndarray:
        """Extract customer features from state"""
        # State structure: [depot(2), customers(num_customers*7), vehicles, drones]
        start_idx = 2
        end_idx = 2 + self.env.num_customers * 7
        customer_data = state[start_idx:end_idx].reshape(self.env.num_customers, 7)
        return customer_data

    def _extract_vehicle_context(
        self, state: np.ndarray, vehicle_id: int
    ) -> np.ndarray:
        """Extract vehicle context from state"""
        start_idx = 2 + self.env.num_customers * 7 + vehicle_id * 6
        end_idx = start_idx + 6
        return state[start_idx:end_idx]

    def train_step(self, batch_size: int = 32):
        """
        Perform one training step for both policies
        """
        if (
            len(self.high_level_buffer) < batch_size
            or len(self.low_level_buffer) < batch_size
        ):
            return {}

        # Train high-level policy
        high_level_loss = self._train_high_level(batch_size)

        # Train low-level policy
        low_level_loss = self._train_low_level(batch_size)

        return {"high_level_loss": high_level_loss, "low_level_loss": low_level_loss}

    def _train_high_level(self, batch_size: int) -> float:
        """Train high-level policy using policy gradient"""
        # Sample batch
        indices = np.random.choice(
            len(self.high_level_buffer), batch_size, replace=False
        )
        batch = [self.high_level_buffer[i] for i in indices]

        states = torch.FloatTensor([exp["state"] for exp in batch])
        vehicle_ids = torch.LongTensor([exp["vehicle_id"] for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])

        # Forward pass
        vehicle_logits, values = self.high_level_policy(states)

        # Calculate advantages
        advantages = rewards - values.squeeze()

        # Policy loss
        log_probs = F.log_softmax(vehicle_logits, dim=1)
        selected_log_probs = log_probs.gather(1, vehicle_ids.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        # Backprop
        self.high_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), 1.0)
        self.high_level_optimizer.step()

        return loss.item()

    def _train_low_level(self, batch_size: int) -> float:
        """Train low-level policy using policy gradient"""
        # Sample batch (filter out depot returns)
        valid_experiences = [
            exp for exp in self.low_level_buffer if exp["customer_features"] is not None
        ]

        if len(valid_experiences) < batch_size:
            return 0.0

        indices = np.random.choice(len(valid_experiences), batch_size, replace=False)
        batch = [valid_experiences[i] for i in indices]

        customer_features = torch.FloatTensor(
            [exp["customer_features"] for exp in batch]
        )
        vehicle_contexts = torch.FloatTensor([exp["vehicle_context"] for exp in batch])
        customer_ids = torch.LongTensor([exp["customer_id"] for exp in batch])
        use_drones = torch.LongTensor([int(exp["use_drone"]) for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])

        # Forward pass
        customer_logits, drone_logits, values = self.low_level_policy(
            customer_features, vehicle_contexts
        )

        # Calculate advantages
        advantages = rewards - values.squeeze()

        # Customer selection loss
        customer_log_probs = F.log_softmax(customer_logits, dim=1)
        selected_customer_log_probs = customer_log_probs.gather(
            1, customer_ids.unsqueeze(1)
        ).squeeze()
        customer_policy_loss = -(
            selected_customer_log_probs * advantages.detach()
        ).mean()

        # Drone usage loss
        drone_log_probs = F.log_softmax(drone_logits, dim=1)
        selected_drone_log_probs = drone_log_probs.gather(
            1, use_drones.unsqueeze(1)
        ).squeeze()
        drone_policy_loss = -(selected_drone_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Total loss
        loss = customer_policy_loss + 0.5 * drone_policy_loss + 0.5 * value_loss

        # Backprop
        self.low_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_level_policy.parameters(), 1.0)
        self.low_level_optimizer.step()

        return loss.item()

    def clear_buffers(self):
        """Clear experience buffers"""
        self.high_level_buffer = []
        self.low_level_buffer = []

    def save(self, path: str):
        """Save model checkpoints"""
        torch.save(
            {
                "high_level_state_dict": self.high_level_policy.state_dict(),
                "low_level_state_dict": self.low_level_policy.state_dict(),
                "high_level_optimizer": self.high_level_optimizer.state_dict(),
                "low_level_optimizer": self.low_level_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.high_level_policy.load_state_dict(checkpoint["high_level_state_dict"])
        self.low_level_policy.load_state_dict(checkpoint["low_level_state_dict"])
        self.high_level_optimizer.load_state_dict(checkpoint["high_level_optimizer"])
        self.low_level_optimizer.load_state_dict(checkpoint["low_level_optimizer"])
