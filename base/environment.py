import gymnasium as gym
from gymnasium import spaces

import numpy as np
from typing import List, Dict, Tuple

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

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        # Gymnasium requires handling seed parameter
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

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

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        Action format: {
            'vehicle_id': int,
            'customer_id': int,
            'use_drone': bool,
            'drone_id': Optional[int]
        }

        Returns: observation, reward, terminated, truncated, info
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
            terminated = len(self.served_customers) == self.num_customers
            truncated = False

            return (
                self._get_state(),
                reward,
                terminated,
                truncated,
                {"cost": cost, "satisfaction": 0.0},
            )

        customer = self.customers[customer_id]

        # Check feasibility
        if customer.id in self.served_customers:
            return (
                self._get_state(),
                -100.0,
                False,
                False,
                {"error": "customer_already_served"},
            )

        # Check precedence constraint
        if (
            customer.customer_type == CustomerType.BACKHAUL
            and not vehicle.visited_linehaul
        ):
            return (
                self._get_state(),
                -100.0,
                False,
                False,
                {"error": "precedence_violation"},
            )

        # Check capacity
        if vehicle.current_load + customer.demand > vehicle.capacity:
            return (
                self._get_state(),
                -50.0,
                False,
                False,
                {"error": "capacity_exceeded"},
            )

        # Calculate travel distance and time
        if use_drone and drone_id is not None:
            drone = self.drones[drone_id]
            if not drone.is_available or customer.demand > drone.capacity:
                return (
                    self._get_state(),
                    -50.0,
                    False,
                    False,
                    {"error": "drone_not_feasible"},
                )

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
        terminated = len(self.served_customers) == self.num_customers
        truncated = False

        info = {
            "cost": cost,
            "satisfaction": satisfaction,
            "arrival_time": arrival_time,
        }

        return self._get_state(), reward, terminated, truncated, info

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
