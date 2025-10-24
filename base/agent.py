import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from module import (
    create_low_level_policy,
    create_high_level_policy,
)


class HierarchicalAgent:
    """
    Hierarchical RL agent combining high-level and low-level policies
    """

    def __init__(
        self,
        env,
        high_level_lr: float = 3e-4,
        low_level_lr: float = 3e-4,
        gamma: float = 0.99,
        embedding_dim: int = 128,
        num_layers: int = 3,
        head_num: int = 8,
        qkv_dim: int = 16,
        ff_hidden_dim: int = 512,
        entropy_coef: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        temperature_start: float = 2.0,
        temperature_end: float = 0.5,
        temperature_decay: float = 0.995,
    ):
        self.env = env
        self.gamma = gamma
        self.embedding_dim = embedding_dim
        self.entropy_coef = entropy_coef

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.temperature = temperature_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay

        # Enhanced high-level policy
        state_dim = env.observation_space.shape[0]
        self.high_level_policy = create_high_level_policy(
            state_dim=state_dim,
            num_vehicles=env.num_vehicles,
            embedding_dim=embedding_dim,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(), lr=high_level_lr
        )

        # Enhanced low-level policy
        self.low_level_policy = create_low_level_policy(
            customer_dim=7,
            vehicle_dim=6,
            num_customers=env.num_customers,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
            logit_clipping=10.0,
        )
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(), lr=low_level_lr
        )

        # Experience buffers
        self.high_level_buffer = []
        self.low_level_buffer = []

        # Training statistics
        self.train_stats = {
            "high_level_losses": [],
            "low_level_losses": [],
            "entropy": [],
        }

    def update_exploration(self, episode: int, total_episodes: int):
        """Update exploration parameters (epsilon and temperature)"""
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start * (self.epsilon_decay**episode)
        )

        # Decay temperature
        self.temperature = max(
            self.temperature_end,
            self.temperature_start * (self.temperature_decay**episode),
        )

    def select_vehicle(self, state: np.ndarray, training: bool = True) -> int:
        """High-level decision: select which vehicle to plan"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            vehicle_logits, _ = self.high_level_policy(state_tensor)

            if training:
                # Epsilon-greedy exploration
                if np.random.random() < self.epsilon:
                    vehicle_id = int(np.random.randint(0, self.env.num_vehicles))
                else:
                    # Sample with temperature
                    dist = torch.distributions.Categorical(
                        logits=vehicle_logits / self.temperature
                    )
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
        """
        Low-level decision: select customer and drone usage
        Enhanced with attention-based encoding
        """
        # Set model to eval mode to avoid InstanceNorm issues with batch_size=1
        was_training = self.low_level_policy.training
        self.low_level_policy.eval()

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
                # Epsilon-greedy + temperature sampling
                if np.random.random() < self.epsilon or len(valid_actions) == 0:
                    # Random valid customer
                    customer_id = (
                        int(np.random.choice(valid_actions))
                        if len(valid_actions) > 0
                        else 0
                    )
                    use_drone = bool(np.random.randint(0, 2))
                else:
                    # Extract logits for valid actions
                    valid_logits = customer_logits[0][valid_actions]

                    # Check for invalid values (inf, -inf, nan)
                    if torch.any(torch.isnan(valid_logits)) or torch.any(
                        torch.isinf(valid_logits)
                    ):
                        # Fallback to random selection if logits are invalid
                        customer_id = int(np.random.choice(valid_actions))
                    else:
                        # Apply temperature and softmax
                        valid_logits_temp = valid_logits / self.temperature

                        # Clamp logits to prevent overflow in softmax
                        valid_logits_temp = torch.clamp(
                            valid_logits_temp, min=-50, max=50
                        )

                        valid_probs = F.softmax(valid_logits_temp, dim=0)

                        # Final check: ensure probs are valid
                        if torch.any(torch.isnan(valid_probs)) or not torch.isfinite(
                            valid_probs.sum()
                        ):
                            customer_id = int(np.random.choice(valid_actions))
                        else:
                            # Sample from valid actions
                            valid_dist = torch.distributions.Categorical(valid_probs)
                            sampled_idx = int(valid_dist.sample().item())
                            customer_id = valid_actions[sampled_idx]

                    # Drone decision
                    drone_logits_temp = torch.clamp(
                        drone_logits / self.temperature, min=-50, max=50
                    )
                    drone_probs = F.softmax(drone_logits_temp, dim=1)

                    if torch.any(torch.isnan(drone_probs)):
                        use_drone = bool(np.random.randint(0, 2))
                    else:
                        drone_dist = torch.distributions.Categorical(drone_probs)
                        use_drone = bool(drone_dist.sample().item())
            else:
                # Greedy selection - choose best from valid actions
                if len(valid_actions) > 0:
                    valid_logits = customer_logits[0][valid_actions]

                    # Handle invalid logits
                    if torch.any(torch.isnan(valid_logits)) or torch.all(
                        torch.isinf(valid_logits)
                    ):
                        customer_id = valid_actions[0]  # Pick first valid action
                    else:
                        best_valid_idx = int(valid_logits.argmax().item())
                        customer_id = valid_actions[best_valid_idx]
                else:
                    customer_id = 0  # Fallback

                use_drone = bool(drone_logits.argmax(dim=1).item())

        # Restore training mode if it was on
        if was_training:
            self.low_level_policy.train()

        return customer_id, use_drone

    def generate_solution(self, training: bool = True) -> Dict:
        """
        Auto-regressive solution generation
        """
        # Set models to eval mode during solution generation to avoid batch norm issues
        self.high_level_policy.eval()
        self.low_level_policy.eval()

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

        # Calculate service rate
        customers_served = len(self.env.served_customers)
        service_rate = customers_served / self.env.num_customers

        return {
            "total_reward": episode_reward,
            "total_cost": self.env.total_cost,
            "total_satisfaction": self.env.total_satisfaction,
            "steps": steps,
            "customers_served": customers_served,
            "service_rate": service_rate,
        }

    def _extract_customer_features(self, state: np.ndarray) -> np.ndarray:
        """Extract customer features from state"""
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

    def train_step(self, batch_size: int = 32) -> Dict:
        """
        Enhanced training step with entropy regularization
        """
        if (
            len(self.high_level_buffer) < batch_size
            or len(self.low_level_buffer) < batch_size
        ):
            return {}

        # Set models to training mode
        self.high_level_policy.train()
        self.low_level_policy.train()

        # Train high-level policy
        high_level_loss, high_level_entropy = self._train_high_level(batch_size)

        # Train low-level policy
        low_level_loss, low_level_entropy = self._train_low_level(batch_size)

        # Update statistics
        self.train_stats["high_level_losses"].append(high_level_loss)
        self.train_stats["low_level_losses"].append(low_level_loss)
        self.train_stats["entropy"].append(low_level_entropy)

        return {
            "high_level_loss": high_level_loss,
            "high_level_entropy": high_level_entropy,
            "low_level_loss": low_level_loss,
            "low_level_entropy": low_level_entropy,
        }

    def _train_high_level(self, batch_size: int) -> Tuple[float, float]:
        """Train high-level policy using policy gradient with entropy regularization"""
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

        # Entropy bonus for exploration
        probs = F.softmax(vehicle_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Total loss
        loss = policy_loss + entropy_loss + 0.5 * value_loss

        # Backprop
        self.high_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), 1.0)
        self.high_level_optimizer.step()

        return loss.item(), entropy.item()

    def _train_low_level(self, batch_size: int) -> Tuple[float, float]:
        """Train low-level policy using policy gradient with entropy regularization"""
        # Sample batch (filter out depot returns)
        valid_experiences = [
            exp for exp in self.low_level_buffer if exp["customer_features"] is not None
        ]

        if len(valid_experiences) < batch_size:
            return 0.0, 0.0

        indices = np.random.choice(len(valid_experiences), batch_size, replace=False)
        batch = [valid_experiences[i] for i in indices]

        customer_features = torch.FloatTensor(
            [exp["customer_features"] for exp in batch]
        )
        vehicle_contexts = torch.FloatTensor([exp["vehicle_context"] for exp in batch])
        customer_ids = torch.LongTensor([exp["customer_id"] for exp in batch])
        use_drones = torch.LongTensor([int(exp["use_drone"]) for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])

        # Create dummy masks for training (all actions valid during training)
        mask = torch.zeros(batch_size, self.env.num_customers, dtype=torch.bool)

        # Forward pass
        customer_logits, drone_logits, values = self.low_level_policy(
            customer_features, vehicle_contexts, mask
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

        # Customer entropy for exploration
        customer_probs = F.softmax(customer_logits, dim=1)
        customer_entropy = -(customer_probs * customer_log_probs).sum(dim=1).mean()

        # Drone usage loss
        drone_log_probs = F.log_softmax(drone_logits, dim=1)
        selected_drone_log_probs = drone_log_probs.gather(
            1, use_drones.unsqueeze(1)
        ).squeeze()
        drone_policy_loss = -(selected_drone_log_probs * advantages.detach()).mean()

        # Drone entropy for exploration
        drone_probs = F.softmax(drone_logits, dim=1)
        drone_entropy = -(drone_probs * drone_log_probs).sum(dim=1).mean()

        # Combined entropy
        total_entropy = customer_entropy + drone_entropy
        entropy_loss = -self.entropy_coef * total_entropy

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Total loss
        loss = (
            customer_policy_loss
            + 0.5 * drone_policy_loss
            + entropy_loss
            + 0.5 * value_loss
        )

        # Backprop
        self.low_level_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_level_policy.parameters(), 1.0)
        self.low_level_optimizer.step()

        return loss.item(), total_entropy.item()

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
                "train_stats": self.train_stats,
                "epsilon": self.epsilon,
                "temperature": self.temperature,
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
        if "train_stats" in checkpoint:
            self.train_stats = checkpoint["train_stats"]
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "temperature" in checkpoint:
            self.temperature = checkpoint["temperature"]
