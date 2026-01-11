from typing import List, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np


def calculate_euclidean_distance_matrix(nodes):
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for start in nodes:
        for end in nodes:
            start_coord = np.array(start.coord)
            end_coord = np.array(end.coord)
            distance_matrix[start.id][end.id] = np.linalg.norm(start_coord - end_coord)

    return distance_matrix.tolist()


def calculate_manhattan_distance_matrix(nodes):
    num_nodes = len(nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for start in nodes:
        for end in nodes:
            start_coord = np.array(start.coord)
            end_coord = np.array(end.coord)
            distance_matrix[start.id][end.id] = np.linalg.norm(
                start_coord - end_coord, ord=1
            )
    return distance_matrix


@dataclass
class Route:
    nodes: List[Optional[int]]
    arrival: List[Optional[float]]
    departure: List[Optional[float]]
    service: List[Optional[float]]


@dataclass
class Node:
    id: int
    coord: Tuple[float, float]
    demand: int
    time_window: List[Tuple[float, float]]


class Solution:
    def __init__(self, routes: List[Tuple[Route, List[Route]]]):
        self.routes = routes


class VRPBTWProblem:
    nodes: List[Node]
    num_fleet: int

    truck_capacity: float
    drone_capacity: float

    system_duration: float
    drone_trip_duration: float

    truck_speed: float
    drone_speed: float
    distance_matrix: List[List[float]]

    launch_time: float
    land_time: float
    service_time: float

    def __init__(self, data_path):
        self.load_from_generated(data_path)

    def load_from_generated(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        config = data["Config"]

        self.num_fleet = config["Vehicles"]["NUM_TRUCKS"]
        self.truck_capacity = config["Vehicles"]["CAPACITY_TRUCK"]
        self.drone_capacity = config["Vehicles"]["CAPACITY_DRONE"]

        self.system_duration = config["General"]["T_MAX_SYSTEM_H"]
        self.drone_trip_duration = config["Vehicles"]["DRONE_DURATION_H"]

        self.truck_speed = config["Vehicles"]["V_TRUCK_KM_H"]
        self.drone_speed = config["Vehicles"]["V_DRONE_KM_H"]

        self.launch_time = config["Vehicles"]["DRONE_TAKEOFF_MIN"] / 60
        self.land_time = config["Vehicles"]["DRONE_LANDING_MIN"] / 60
        self.service_time = config["Vehicles"]["SERVICE_TIME_MIN"] / 60

        self.nodes = []
        depot = config["Depot"]
        self.nodes.append(Node(depot["id"], depot["coord"], 0, depot["time_window_h"]))
        nodes = data["Nodes"]
        for node in nodes:
            self.nodes.append(
                Node(
                    node["id"],
                    node["coord"],
                    node["demand"],
                    node["tw_h"],
                )
            )
        self.distance_matrix = calculate_euclidean_distance_matrix(self.nodes)

    @staticmethod
    def fitness(solution) -> float:
        return float("inf")
