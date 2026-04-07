from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Tier(str, Enum):
    NORMAL = "normal"
    PREMIUM = "premium"
    VIP = "vip"


class CarType(str, Enum):
    BASIC = "basic"
    SEDAN = "sedan"
    XUV = "xuv"
    ARMORED = "armored"


class ActionType(str, Enum):
    ASSIGN = "assign"
    POOL = "pool"
    REPOSITION = "reposition"
    WAIT = "wait"
    NO_OP = "no_op"


@dataclass
class RiderState:
    rider_id: str
    tier: Tier
    urgency: str
    group_size: int
    pickup_zone: int
    drop_zone: int
    share_allowed: bool
    arrival_time: float
    wait_time: float = 0.0
    assigned_car: Optional[str] = None
    assigned_driver: Optional[str] = None
    cancelled: bool = False
    completed: bool = False
    safe_drop: bool = False
    pooled_with: list[str] = field(default_factory=list)

    @property
    def waiting(self) -> bool:
        return not self.cancelled and not self.completed and self.assigned_car is None


@dataclass
class CarState:
    car_id: str
    car_type: CarType
    capacity: int
    location_zone: int
    driver_id: Optional[str] = None
    available: bool = True
    busy_until: float = 0.0
    current_rider_ids: list[str] = field(default_factory=list)


@dataclass
class DriverState:
    driver_id: str
    zone: int
    available: bool = True
    busy_until: float = 0.0
    busy_time: float = 0.0
    idle_time: float = 0.0
    on_break: bool = False


class RiderView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rider_id: str
    tier: str
    urgency: str
    group_size: int
    pickup_zone: int
    drop_zone: int
    share_allowed: bool
    wait_time: float
    assigned_car: Optional[str]
    assigned_driver: Optional[str]
    cancelled: bool
    completed: bool
    safe_drop: bool


class CarView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    car_id: str
    car_type: str
    capacity: int
    location_zone: int
    driver_id: Optional[str]
    available: bool
    busy_until: float
    current_rider_ids: list[str]


class DriverView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    driver_id: str
    zone: int
    available: bool
    busy_until: float
    busy_time: float
    idle_time: float
    on_break: bool


class Observation(BaseModel):
    current_time: float
    task_id: str
    traffic_factor: float
    waiting_riders: list[RiderView]
    available_cars: list[CarView]
    driver_status: list[DriverView]
    metrics: dict[str, float]
    recent_events: list[str]


class Action(BaseModel):
    action_type: ActionType
    rider_id: Optional[str] = None
    rider_ids: Optional[list[str]] = None
    car_id: Optional[str] = None
    driver_id: Optional[str] = None
    target_zone: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        if isinstance(data.get("action_type"), Enum):
            data["action_type"] = data["action_type"].value
        return data


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]
