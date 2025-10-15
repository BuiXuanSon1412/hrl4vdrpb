import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import gym
from gym import spaces

# ============================================================================
# Data Structures
# ============================================================================


class CustomerType(Enum):
    LINEHAUL = 0
    BACKHAUL = 1


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    time_window_start: float
    time_window_end: float
    customer_type: CustomerType


@dataclass
class Vehicle:
    id: int
    capacity: float
    current_load: float
    x: float
    y: float
    current_time: float
    route: List[int]
    visited_linehaul: bool
    visited_backhaul: bool


@dataclass
class Drone:
    id: int
    capacity: float
    battery_capacity: float
    current_battery: float
    speed: float
    is_available: bool
    x: float
    y: float


# ============================================================================
# Environment
# ============================================================================


class VehicleDroneRoutingEnv(gym.Env):
    """
    Environment for Vehicle Routing Problem with Drones and Backhauls
    """

    def __init__(
        self,
        num_customers: int = 20,
        num_vehicles: int = 3,
        num_drones: int = 2,
        vehicle_capacity: float = 100.0,
        drone_capacity: float = 10.0,
        drone_battery: float = 100.0,
        map_size: float = 100.0,
    ):
        super().__init__()

        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.num_drones = num_drones
        self.vehicle_capacity = vehicle_capacity
        self.drone_capacity = drone_capacity
        self.drone_battery = drone_battery
        self.map_size = map_size

        # Depot location
        self.depot = (map_size / 2, map_size / 2)

        # State components
        self.customers: List[Customer] = []
        self.vehicles: List[Vehicle] = []
        self.drones: List[Drone] = []
        self.served_customers = set()

        # For tracking solution quality
        self.total_cost = 0.0
        self.total_satisfaction = 0.0

        # Action and observation spaces (will be defined in reset)
        self.action_space = spaces.Discrete(num_customers + 1)  # +1 for return to depot
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._get_state_dim(),), dtype=np.float32
        )

    def _get_state_dim(self) -> int:
        """Calculate state dimension"""
        customer_features = 7  # x, y, demand, tw_start, tw_end, type, served
        vehicle_features = 6  # x, y, load, time, visited_linehaul, visited_backhaul
        drone_features = 5  # x, y, battery, available, capacity

        state_dim = (
            self.num_customers * customer_features
            + self.num_vehicles * vehicle_features
            + self.num_drones * drone_features
            + 2
        )  # depot coordinates
        return state_dim

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.customers = []
        self.vehicles = []
        self.drones = []
        self.served_customers = set()
        self.total_cost = 0.0
        self.total_satisfaction = 0.0

        # Generate random customers
        num_linehaul = self.num_customers // 2
        num_backhaul = self.num_customers - num_linehaul

        for i in range(self.num_customers):
            customer_type = (
                CustomerType.LINEHAUL if i < num_linehaul else CustomerType.BACKHAUL
            )
            self.customers.append(
                Customer(
                    id=i,
                    x=np.random.uniform(0, self.map_size),
                    y=np.random.uniform(0, self.map_size),
                    demand=np.random.uniform(5, 20),
                    time_window_start=np.random.uniform(0, 100),
                    time_window_end=np.random.uniform(100, 200),
                    customer_type=customer_type,
                )
            )

        # Initialize vehicles at depot
        for i in range(self.num_vehicles):
            self.vehicles.append(
                Vehicle(
                    id=i,
                    capacity=self.vehicle_capacity,
                    current_load=0.0,
                    x=self.depot[0],
                    y=self.depot[1],
                    current_time=0.0,
                    route=[],
                    visited_linehaul=False,
                    visited_backhaul=False,
                )
            )

        # Initialize drones at depot
        for i in range(self.num_drones):
            self.drones.append(
                Drone(
                    id=i,
                    capacity=self.drone_capacity,
                    battery_capacity=self.drone_battery,
                    current_battery=self.drone_battery,
                    speed=2.0,  # Drones are faster
                    is_available=True,
                    x=self.depot[0],
                    y=self.depot[1],
                )
            )

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []

        # Depot coordinates (normalized)
        state.extend([self.depot[0] / self.map_size, self.depot[1] / self.map_size])

        # Customer features
        for customer in self.customers:
            state.extend(
                [
                    customer.x / self.map_size,
                    customer.y / self.map_size,
                    customer.demand / 20.0,  # Normalize by max demand
                    customer.time_window_start / 200.0,
                    customer.time_window_end / 200.0,
                    float(customer.customer_type.value),
                    float(customer.id in self.served_customers),
                ]
            )

        # Vehicle features
        for vehicle in self.vehicles:
            state.extend(
                [
                    vehicle.x / self.map_size,
                    vehicle.y / self.map_size,
                    vehicle.current_load / vehicle.capacity,
                    vehicle.current_time / 200.0,
                    float(vehicle.visited_linehaul),
                    float(vehicle.visited_backhaul),
                ]
            )

        # Drone features
        for drone in self.drones:
            state.extend(
                [
                    drone.x / self.map_size,
                    drone.y / self.map_size,
                    drone.current_battery / drone.battery_capacity,
                    float(drone.is_available),
                    drone.capacity / self.drone_capacity,
                ]
            )

        return np.array(state, dtype=np.float32)

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_satisfaction(
        self, arrival_time: float, tw_start: float, tw_end: float
    ) -> float:
        """
        Calculate satisfaction score based on time window adherence
        Gaussian function centered on time window
        """
        tw_center = (tw_start + tw_end) / 2
        tw_width = (tw_end - tw_start) / 2

        if tw_start <= arrival_time <= tw_end:
            # Inside time window: high satisfaction
            return 1.0
        else:
            # Outside time window: exponential decay
            deviation = min(abs(arrival_time - tw_start), abs(arrival_time - tw_end))
            return np.exp(-deviation / (tw_width + 1e-6))

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        Action format: {
            'vehicle_id': int,
            'customer_id': int,
            'use_drone': bool,
            'drone_id': Optional[int]
        }
        """
        vehicle_id = action["vehicle_id"]
        customer_id = action["customer_id"]
        use_drone = action.get("use_drone", False)
        drone_id = action.get("drone_id", None)

        vehicle = self.vehicles[vehicle_id]

        # Check if returning to depot
        if customer_id == -1:
            distance = self.calculate_distance(
                vehicle.x, vehicle.y, self.depot[0], self.depot[1]
            )
            travel_time = distance / 1.0  # Vehicle speed = 1.0
            cost = distance

            vehicle.x = self.depot[0]
            vehicle.y = self.depot[1]
            vehicle.current_time += travel_time
            vehicle.current_load = 0.0

            self.total_cost += cost
            reward = -cost

            # Check if all customers served
            done = len(self.served_customers) == self.num_customers

            return self._get_state(), reward, done, {"cost": cost, "satisfaction": 0.0}

        customer = self.customers[customer_id]

        # Check feasibility
        if customer.id in self.served_customers:
            return (
                self._get_state(),
                -100.0,
                False,
                {"error": "customer_already_served"},
            )

        # Check precedence constraint
        if (
            customer.customer_type == CustomerType.BACKHAUL
            and not vehicle.visited_linehaul
        ):
            return self._get_state(), -100.0, False, {"error": "precedence_violation"}

        # Check capacity
        if vehicle.current_load + customer.demand > vehicle.capacity:
            return self._get_state(), -50.0, False, {"error": "capacity_exceeded"}

        # Calculate travel distance and time
        if use_drone and drone_id is not None:
            drone = self.drones[drone_id]
            if not drone.is_available or customer.demand > drone.capacity:
                return self._get_state(), -50.0, False, {"error": "drone_not_feasible"}

            # Drone deployment
            distance = self.calculate_distance(
                vehicle.x, vehicle.y, customer.x, customer.y
            )
            return_distance = self.calculate_distance(
                customer.x, customer.y, vehicle.x, vehicle.y
            )
            total_distance = distance + return_distance

            battery_cost = total_distance * 0.5  # Battery consumption rate
            if battery_cost > drone.current_battery:
                return (
                    self._get_state(),
                    -50.0,
                    False,
                    {"error": "insufficient_battery"},
                )

            travel_time = total_distance / drone.speed
            cost = total_distance * 1.5  # Drone cost factor

            drone.current_battery -= battery_cost
            drone.is_available = False

        else:
            # Vehicle serves customer
            distance = self.calculate_distance(
                vehicle.x, vehicle.y, customer.x, customer.y
            )
            travel_time = distance / 1.0  # Vehicle speed
            cost = distance

            vehicle.x = customer.x
            vehicle.y = customer.y

        # Update time and calculate satisfaction
        arrival_time = vehicle.current_time + travel_time
        satisfaction = self.calculate_satisfaction(
            arrival_time, customer.time_window_start, customer.time_window_end
        )

        # Update vehicle state
        vehicle.current_time = arrival_time
        vehicle.current_load += customer.demand
        vehicle.route.append(customer_id)

        if customer.customer_type == CustomerType.LINEHAUL:
            vehicle.visited_linehaul = True
        else:
            vehicle.visited_backhaul = True

        # Mark customer as served
        self.served_customers.add(customer.id)

        # Update totals
        self.total_cost += cost
        self.total_satisfaction += satisfaction

        # Calculate reward (multi-objective)
        reward = -cost + 10.0 * satisfaction  # Weight satisfaction higher

        # Check if done
        done = len(self.served_customers) == self.num_customers

        info = {
            "cost": cost,
            "satisfaction": satisfaction,
            "arrival_time": arrival_time,
        }

        return self._get_state(), reward, done, info

    def get_valid_actions(self, vehicle_id: int) -> List[int]:
        """Get list of valid customer IDs for current vehicle"""
        vehicle = self.vehicles[vehicle_id]
        valid = []

        for customer in self.customers:
            if customer.id in self.served_customers:
                continue

            # Check precedence
            if (
                customer.customer_type == CustomerType.BACKHAUL
                and not vehicle.visited_linehaul
            ):
                continue

            # Check capacity
            if vehicle.current_load + customer.demand > vehicle.capacity:
                continue

            valid.append(customer.id)

        # Always allow return to depot
        valid.append(-1)

        return valid


# ============================================================================
# Neural Network Modules
# ============================================================================


class CustomerEncoder(nn.Module):
    """Encode customer features using attention mechanism"""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (batch, num_customers, input_dim)
        x = self.input_proj(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class HighLevelPolicy(nn.Module):
    """
    High-level policy: decides which vehicle to plan next and strategic decisions
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_vehicles: int = 3):
        super().__init__()

        self.num_vehicles = num_vehicles

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Vehicle selection head
        self.vehicle_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_vehicles),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        features = self.encoder(state)

        # Vehicle selection logits
        vehicle_logits = self.vehicle_selector(features)

        # State value
        value = self.value_head(features)

        return vehicle_logits, value


class LowLevelPolicy(nn.Module):
    """
    Low-level policy: decides which customer to visit next and whether to use drone
    Uses pointer network architecture for auto-regressive generation
    """

    def __init__(
        self,
        customer_dim: int = 7,
        vehicle_dim: int = 6,
        hidden_dim: int = 128,
        num_customers: int = 20,
    ):
        super().__init__()

        self.num_customers = num_customers
        self.hidden_dim = hidden_dim

        # Customer encoder
        self.customer_encoder = CustomerEncoder(customer_dim, hidden_dim)

        # Vehicle/context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pointer mechanism for customer selection
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.pointing_scale = np.sqrt(hidden_dim)

        # Drone usage decision
        self.drone_decision = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Use drone or not
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, customer_features, vehicle_context, mask=None):
        """
        Args:
            customer_features: (batch, num_customers, customer_dim)
            vehicle_context: (batch, vehicle_dim)
            mask: (batch, num_customers) - mask for served customers
        """
        # Encode customers
        customer_encoded = self.customer_encoder(
            customer_features
        )  # (batch, num_customers, hidden)

        # Encode context
        context_encoded = self.context_encoder(vehicle_context)  # (batch, hidden)

        # Pointer mechanism for customer selection
        query = self.pointer_query(context_encoded).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.pointer_key(customer_encoded)  # (batch, num_customers, hidden)

        # Calculate attention scores
        logits = (
            torch.matmul(query, keys.transpose(1, 2)) / self.pointing_scale
        )  # (batch, 1, num_customers)
        logits = logits.squeeze(1)  # (batch, num_customers)

        # Apply mask
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        # Drone usage decision
        # Aggregate customer info with context
        customer_agg = customer_encoded.mean(dim=1)  # (batch, hidden)
        drone_input = torch.cat([customer_agg, context_encoded], dim=1)
        drone_logits = self.drone_decision(drone_input)  # (batch, 2)

        # Value estimation
        value = self.value_head(context_encoded)

        return logits, drone_logits, value


# ============================================================================
# HRL Agent
# ============================================================================


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
                vehicle_id = dist.sample().item()
            else:
                vehicle_id = vehicle_logits.argmax(dim=1).item()

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
                customer_id = customer_dist.sample().item()

                drone_dist = torch.distributions.Categorical(logits=drone_logits)
                use_drone = bool(drone_dist.sample().item())
            else:
                # Greedy selection
                customer_id = customer_logits.argmax(dim=1).item()
                use_drone = bool(drone_logits.argmax(dim=1).item())

        return customer_id, use_drone

    def generate_solution(self, training: bool = True) -> Dict:
        """
        Auto-regressive solution generation
        """
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < self.env.num_customers * 2:
            # High-level: select vehicle
            vehicle_id = self.select_vehicle(state, training)

            # Get valid actions for this vehicle
            valid_actions = self.env.get_valid_actions(vehicle_id)

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

            next_state, reward, done, info = self.env.step(action)

            # Store experience
            if training:
                self.high_level_buffer.append(
                    {
                        "state": state,
                        "vehicle_id": vehicle_id,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
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
                        "done": done,
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


# ============================================================================
# Training Loop
# ============================================================================


def train_hrl_agent(
    num_episodes: int = 1000,
    batch_size: int = 32,
    eval_interval: int = 50,
    save_path: str = "hrl_checkpoint.pt",
):
    """
    Main training loop for HRL agent
    """
    # Create environment and agent
    env = VehicleDroneRoutingEnv(num_customers=20, num_vehicles=3, num_drones=2)

    agent = HierarchicalAgent(env)

    # Training metrics
    episode_rewards = []
    episode_costs = []
    episode_satisfactions = []

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

        # Train policies
        if episode > 0 and episode % 5 == 0:  # Train every 5 episodes
            losses = agent.train_step(batch_size)

            if losses:
                print(
                    f"Episode {episode}: "
                    f"Reward={result['total_reward']:.2f}, "
                    f"Cost={result['total_cost']:.2f}, "
                    f"Satisfaction={result['total_satisfaction']:.2f}, "
                    f"HL_Loss={losses.get('high_level_loss', 0):.4f}, "
                    f"LL_Loss={losses.get('low_level_loss', 0):.4f}"
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

            for _ in range(10):
                eval_result = agent.generate_solution(training=False)
                eval_rewards.append(eval_result["total_reward"])
                eval_costs.append(eval_result["total_cost"])
                eval_satisfactions.append(eval_result["total_satisfaction"])

            print(
                f"Avg Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}"
            )
            print(f"Avg Cost: {np.mean(eval_costs):.2f} ± {np.std(eval_costs):.2f}")
            print(
                f"Avg Satisfaction: {np.mean(eval_satisfactions):.2f} ± {np.std(eval_satisfactions):.2f}"
            )
            print("=" * 80 + "\n")

            # Save checkpoint
            agent.save(save_path)
            print(f"Model saved to {save_path}\n")

        # Clear buffers periodically to prevent memory overflow
        if episode > 0 and episode % 100 == 0:
            agent.clear_buffers()

    print("\nTraining completed!")

    return agent, {
        "rewards": episode_rewards,
        "costs": episode_costs,
        "satisfactions": episode_satisfactions,
    }


# ============================================================================
# Evaluation and Visualization
# ============================================================================


def evaluate_agent(agent: HierarchicalAgent, num_episodes: int = 100) -> Dict:
    """
    Comprehensive evaluation of trained agent
    """
    results = {
        "rewards": [],
        "costs": [],
        "satisfactions": [],
        "avg_time_deviation": [],
        "served_customers": [],
    }

    for _ in range(num_episodes):
        result = agent.generate_solution(training=False)

        results["rewards"].append(result["total_reward"])
        results["costs"].append(result["total_cost"])
        results["satisfactions"].append(result["total_satisfaction"])
        results["served_customers"].append(len(agent.env.served_customers))

    # Compute statistics
    stats = {
        "mean_reward": np.mean(results["rewards"]),
        "std_reward": np.std(results["rewards"]),
        "mean_cost": np.mean(results["costs"]),
        "std_cost": np.std(results["costs"]),
        "mean_satisfaction": np.mean(results["satisfactions"]),
        "std_satisfaction": np.std(results["satisfactions"]),
        "avg_served": np.mean(results["served_customers"]),
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
        color = "blue" if customer.customer_type == CustomerType.LINEHAUL else "green"
        marker = "o" if customer.customer_type == CustomerType.LINEHAUL else "^"
        ax1.scatter(customer.x, customer.y, c=color, s=100, marker=marker, zorder=3)
        ax1.text(customer.x + 2, customer.y + 2, f"{customer.id}", fontsize=8)

    # Plot vehicle routes
    colors_vehicles = ["purple", "orange", "brown"]
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
    linehaul_patch = mpatches.Patch(color="blue", label="Linehaul")
    backhaul_patch = mpatches.Patch(color="green", label="Backhaul")
    ax1.legend(handles=[linehaul_patch, backhaul_patch], loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Metrics
    ax2.set_title("Solution Metrics", fontsize=14, fontweight="bold")
    ax2.axis("off")

    metrics_text = f"""
    SOLUTION QUALITY
    {"=" * 40}
    
    Total Reward: {result["total_reward"]:.2f}
    Total Cost: {result["total_cost"]:.2f}
    Total Satisfaction: {result["total_satisfaction"]:.2f}
    Avg Satisfaction: {result["total_satisfaction"] / agent.env.num_customers:.3f}
    
    Customers Served: {len(agent.env.served_customers)} / {agent.env.num_customers}
    
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
    plt.savefig("hrl_solution_visualization.png", dpi=150, bbox_inches="tight")
    print("Visualization saved to 'hrl_solution_visualization.png'")
    plt.show()


# ============================================================================
# Main Execution
# ============================================================================

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
