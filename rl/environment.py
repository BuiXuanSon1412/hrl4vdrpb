import gymnasium as gym
from gymnasium import spaces

import numpy as np
from typing import List, Dict, Tuple, Optional

from problem import Customer, Vehicle, Drone, CustomerType


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
        normalize_rewards: bool = True,
        cost_weight: float = 0.5,
        satisfaction_weight: float = 0.5,
    ):
        super().__init__()

        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.num_drones = num_drones
        self.vehicle_capacity = vehicle_capacity
        self.drone_capacity = drone_capacity
        self.drone_battery = drone_battery
        self.map_size = map_size

        # Reward configuration
        self.normalize_rewards = normalize_rewards
        self.cost_weight = cost_weight
        self.satisfaction_weight = satisfaction_weight

        # Calculate normalization bounds
        self.max_possible_cost = self._estimate_max_cost()
        self.max_satisfaction_per_customer = 1.0
        self.max_unserved_penalty_per_customer = 200.0

        # Depot location
        self.depot = (map_size / 2, map_size / 2)

        # State components
        self.customers: List[Customer] = []
        self.vehicles: List[Vehicle] = []
        self.drones: List[Drone] = []
        self.served_customers = set()

        # Track drone missions
        self.drone_missions: Dict[int, Dict] = {}

        # For tracking solution quality
        self.total_cost = 0.0
        self.total_satisfaction = 0.0

        # Action and observation spaces
        self.action_space = spaces.Discrete(num_customers + 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self._get_state_dim(),), dtype=np.float32
        )

    def _estimate_max_cost(self) -> float:
        """
        Estimate maximum possible cost for normalization
        Conservative upper bound based on worst-case routing
        """
        # Diagonal distance of map (worst case single trip)
        diagonal = np.sqrt(2) * self.map_size

        # Worst case: each vehicle visits all customers with returns
        # This is highly conservative but ensures we don't exceed bounds
        max_cost = diagonal * self.num_customers * 2

        return max_cost

    def _get_state_dim(self) -> int:
        """Calculate state dimension"""
        customer_features = 7
        vehicle_features = 7
        drone_features = 5

        state_dim = (
            self.num_customers * customer_features
            + self.num_vehicles * vehicle_features
            + self.num_drones * drone_features
            + 2
        )
        return state_dim

    def _calculate_reward(
        self, cost: float, satisfaction: float, unserved_penalty: float = 0.0
    ) -> float:
        """
        Calculate properly normalized multi-objective reward

        Args:
            cost: Travel cost (distance)
            satisfaction: Time window satisfaction score [0, 1]
            unserved_penalty: Additional penalty for unserved customers

        Returns:
            Normalized reward in approximately [-1, 1] range
        """
        if not self.normalize_rewards:
            # Legacy behavior for backward compatibility
            return -cost + 10.0 * satisfaction - unserved_penalty

        # Normalize cost to [0, 1] where 0 = worst (max cost), 1 = best (zero cost)
        normalized_cost = 1.0 - min(cost / self.max_possible_cost, 1.0)

        # Satisfaction already in [0, 1]
        normalized_satisfaction = satisfaction

        # Normalize unserved penalty
        max_total_penalty = self.max_unserved_penalty_per_customer * self.num_customers
        normalized_penalty = min(unserved_penalty / max_total_penalty, 1.0)

        # Weighted combination
        # Both objectives in [0, 1], weights sum to 1.0
        reward = (
            self.cost_weight * normalized_cost
            + self.satisfaction_weight * normalized_satisfaction
            - normalized_penalty  # Penalty reduces reward
        )

        return reward

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.customers = []
        self.vehicles = []
        self.drones = []
        self.served_customers = set()
        self.drone_missions = {}
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
                    demand=np.random.uniform(5, 15),
                    time_window_start=np.random.uniform(0, 100),
                    time_window_end=np.random.uniform(100, 200),
                    customer_type=customer_type,
                )
            )

        # Initialize vehicles at depot - FULLY LOADED
        for i in range(self.num_vehicles):
            self.vehicles.append(
                Vehicle(
                    id=i,
                    capacity=self.vehicle_capacity,
                    current_load=self.vehicle_capacity,
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
                    speed=2.0,
                    is_available=True,
                    x=self.depot[0],
                    y=self.depot[1],
                )
            )

        return self._get_state(), {}

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
                    customer.demand / 20.0,
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
                    float(len(vehicle.route) > 0),
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
        """Calculate satisfaction score based on time window adherence"""
        if tw_start <= arrival_time <= tw_end:
            return 1.0
        else:
            deviation = min(abs(arrival_time - tw_start), abs(arrival_time - tw_end))
            tw_width = (tw_end - tw_start) / 2
            return np.exp(-deviation / (tw_width + 1e-6))

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        vehicle_id = action["vehicle_id"]
        customer_id = action["customer_id"]
        use_drone = action.get("use_drone", False)
        drone_id = action.get("drone_id", None)

        vehicle = self.vehicles[vehicle_id]

        # Check if returning to depot
        if customer_id == -1:
            result = self._handle_depot_return(vehicle)
            # Check if episode should terminate
            terminated = len(self.served_customers) == self.num_customers
            truncated = result[3]

            # Add penalty for unserved customers if terminating
            state, reward, _, _, info = result
            if terminated or truncated:
                unserved_count = self.num_customers - len(self.served_customers)
                if unserved_count > 0:
                    unserved_penalty = (
                        self.max_unserved_penalty_per_customer * unserved_count
                    )
                    # Recalculate reward with penalty
                    reward = self._calculate_reward(info["cost"], 0.0, unserved_penalty)

            return state, reward, terminated, truncated, info

        customer = self.customers[customer_id]

        # Validate action
        error = self._validate_action(vehicle, customer, use_drone, drone_id)
        if error:
            # Invalid action penalty
            invalid_penalty = 100.0
            reward = self._calculate_reward(0.0, 0.0, invalid_penalty)
            return self._get_state(), reward, False, False, {"error": error}

        # Execute action
        if use_drone and drone_id is not None:
            return self._handle_drone_delivery(
                vehicle, customer, drone_id, action.get("drone_return_location", -1)
            )
        else:
            return self._handle_vehicle_delivery(vehicle, customer)

    def _validate_action(
        self,
        vehicle: Vehicle,
        customer: Customer,
        use_drone: bool,
        drone_id: Optional[int],
    ) -> Optional[str]:
        """Validate if action is feasible"""

        if customer.id in self.served_customers:
            return "customer_already_served"

        if customer.customer_type == CustomerType.LINEHAUL:
            if vehicle.current_load < customer.demand:
                return "insufficient_load_for_delivery"
        else:  # BACKHAUL
            if not vehicle.visited_linehaul:
                return "must_visit_linehaul_first"
            if vehicle.current_load + customer.demand > vehicle.capacity:
                return "capacity_exceeded_for_pickup"

        if use_drone and drone_id is not None:
            drone = self.drones[drone_id]
            if not drone.is_available:
                return "drone_not_available"
            if customer.demand > drone.capacity:
                return "drone_capacity_exceeded"

        return None

    def _handle_depot_return(
        self, vehicle: Vehicle
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Handle vehicle returning to depot"""
        distance = self.calculate_distance(
            vehicle.x, vehicle.y, self.depot[0], self.depot[1]
        )
        travel_time = distance / 1.0
        cost = distance

        vehicle.x = self.depot[0]
        vehicle.y = self.depot[1]
        vehicle.current_time += travel_time
        vehicle.current_load = 0.0

        self.total_cost += cost

        # Use normalized reward calculation
        reward = self._calculate_reward(cost, 0.0, 0.0)

        terminated = len(self.served_customers) == self.num_customers
        truncated = False

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            {"cost": cost, "satisfaction": 0.0},
        )

    def _handle_vehicle_delivery(
        self, vehicle: Vehicle, customer: Customer
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Handle vehicle serving customer directly"""

        distance = self.calculate_distance(vehicle.x, vehicle.y, customer.x, customer.y)
        travel_time = distance / 1.0
        cost = distance

        vehicle.x = customer.x
        vehicle.y = customer.y
        vehicle.current_time += travel_time

        # Update load based on customer type
        if customer.customer_type == CustomerType.LINEHAUL:
            vehicle.current_load -= customer.demand
            vehicle.visited_linehaul = True
        else:
            vehicle.current_load += customer.demand
            vehicle.visited_backhaul = True

        satisfaction = self.calculate_satisfaction(
            vehicle.current_time, customer.time_window_start, customer.time_window_end
        )

        vehicle.route.append(customer.id)
        self.served_customers.add(customer.id)

        self.total_cost += cost
        self.total_satisfaction += satisfaction

        # Use normalized reward calculation
        reward = self._calculate_reward(cost, satisfaction, 0.0)

        terminated = len(self.served_customers) == self.num_customers
        truncated = False

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            {
                "cost": cost,
                "satisfaction": satisfaction,
                "arrival_time": vehicle.current_time,
            },
        )

    def _handle_drone_delivery(
        self,
        vehicle: Vehicle,
        customer: Customer,
        drone_id: int,
        return_location: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Handle drone serving customer"""
        drone = self.drones[drone_id]

        launch_distance = self.calculate_distance(
            vehicle.x, vehicle.y, customer.x, customer.y
        )

        if return_location == -1:
            return_x, return_y = self.depot
        else:
            return_customer = self.customers[return_location]
            return_x, return_y = return_customer.x, return_customer.y

        return_distance = self.calculate_distance(
            customer.x, customer.y, return_x, return_y
        )

        total_distance = launch_distance + return_distance

        battery_cost = total_distance * 0.5
        if battery_cost > drone.current_battery:
            # Battery insufficient penalty
            battery_penalty = 50.0
            reward = self._calculate_reward(0.0, 0.0, battery_penalty)
            return (
                self._get_state(),
                reward,
                False,
                False,
                {"error": "insufficient_battery"},
            )

        travel_time = total_distance / drone.speed
        cost = total_distance * 1.5

        arrival_time = vehicle.current_time + (launch_distance / drone.speed)

        satisfaction = self.calculate_satisfaction(
            arrival_time, customer.time_window_start, customer.time_window_end
        )

        drone.current_battery -= battery_cost
        drone.x = return_x
        drone.y = return_y

        self.served_customers.add(customer.id)

        if customer.customer_type == CustomerType.LINEHAUL:
            vehicle.visited_linehaul = True
            vehicle.current_load -= customer.demand
        else:
            vehicle.visited_backhaul = True
            vehicle.current_load += customer.demand

        self.total_cost += cost
        self.total_satisfaction += satisfaction

        # Use normalized reward calculation
        reward = self._calculate_reward(cost, satisfaction, 0.0)

        terminated = len(self.served_customers) == self.num_customers
        truncated = False

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            {
                "cost": cost,
                "satisfaction": satisfaction,
                "arrival_time": arrival_time,
                "used_drone": True,
            },
        )

    def get_valid_actions(self, vehicle_id: int) -> List[int]:
        """
        Get list of valid customer IDs for current vehicle

        CRITICAL FIX: Only allow depot return when:
        1. Vehicle has started a route, AND
        2. Either no valid customers OR all customers served
        """
        vehicle = self.vehicles[vehicle_id]
        valid = []

        for customer in self.customers:
            if customer.id in self.served_customers:
                continue

            if customer.customer_type == CustomerType.LINEHAUL:
                if vehicle.current_load >= customer.demand:
                    valid.append(customer.id)
            else:  # BACKHAUL
                if not vehicle.visited_linehaul:
                    continue
                if vehicle.current_load + customer.demand <= vehicle.capacity:
                    valid.append(customer.id)

        # CRITICAL: Only allow depot return if route started AND
        # (no valid customers OR all customers served)
        if len(vehicle.route) > 0:
            if len(valid) == 0 or len(self.served_customers) == self.num_customers:
                valid.append(-1)

        return valid
