from enum import Enum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import random
import math


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
