"""Robot agent classes for warehouse fulfilment simulation."""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Tuple


class RobotStatus(Enum):
    IDLE = auto()
    MOVING_TO_SHELF = auto()
    PICKING = auto()
    MOVING_TO_PACKING = auto()
    DELIVERING = auto()
    DELAYED = auto()


class Task:
    def __init__(
        self,
        task_id: int,
        sku: str,
        shelf: tuple[int, int],
        packing: tuple[int, int],
        priority: float = 1.0,
        zone: str = "A",
        shortage_delay: int = 0,
    ):
        self.task_id = task_id
        self.sku = sku
        self.shelf = shelf
        self.packing = packing
        self.priority = priority
        self.zone = zone
        self.shortage_delay = shortage_delay
        self.assigned_robot: Optional[int] = None
        self.completed = False
        self.start_step: Optional[int] = None
        self.pick_step: Optional[int] = None
        self.complete_step: Optional[int] = None
        self.created_step: Optional[int] = None
        self.queue_delay: int = 0

    @property
    def total_time(self) -> Optional[int]:
        if self.start_step is not None and self.complete_step is not None:
            return self.complete_step - self.start_step
        return None


class Robot:
    """Autonomous warehouse robot agent."""

    def __init__(self, robot_id: int, start_pos: tuple[int, int]):
        self.robot_id = robot_id
        self.position = start_pos
        self.status = RobotStatus.IDLE
        self.current_task: Optional[Task] = None
        self.path: list[tuple[int, int]] = []
        self.path_idx = 0
        self.steps_taken = 0
        self.tasks_completed = 0
        self.distance_travelled = 0
        self.idle_steps = 0
        self.picking_steps_remaining = 0
        self.delay_steps_remaining = 0

    @property
    def is_idle(self) -> bool:
        return self.status == RobotStatus.IDLE

    def assign_task(self, task: Task, path_to_shelf: List[Tuple[int, int]], current_step: int) -> None:
        self.current_task = task
        task.assigned_robot = self.robot_id
        task.start_step = current_step
        if path_to_shelf and len(path_to_shelf) > 1:
            self.path = path_to_shelf
            self.path_idx = 1
            self.status = RobotStatus.MOVING_TO_SHELF
        else:
            self.status = RobotStatus.PICKING
            self.picking_steps_remaining = 2 + task.shortage_delay

    def step(self, path_to_packing: Optional[List[Tuple[int, int]]] = None, current_step: int = 0) -> bool:
        self.steps_taken += 1
        completed = False

        if self.status == RobotStatus.IDLE:
            self.idle_steps += 1
            return completed

        if self.status == RobotStatus.MOVING_TO_SHELF:
            if self.path_idx < len(self.path):
                self.position = self.path[self.path_idx]
                self.path_idx += 1
                self.distance_travelled += 1
            else:
                self.status = RobotStatus.PICKING
                self.picking_steps_remaining = 2 + (self.current_task.shortage_delay if self.current_task else 0)
                if self.current_task:
                    self.current_task.pick_step = current_step
            return completed

        if self.status == RobotStatus.PICKING:
            self.picking_steps_remaining -= 1
            if self.picking_steps_remaining <= 0:
                if self.current_task and self.current_task.shortage_delay > 0:
                    self.delay_steps_remaining = self.current_task.shortage_delay
                    self.status = RobotStatus.DELAYED
                elif path_to_packing and len(path_to_packing) > 1:
                    self.status = RobotStatus.MOVING_TO_PACKING
                    self.path = path_to_packing
                    self.path_idx = 1
                else:
                    self.status = RobotStatus.DELIVERING
            return completed

        if self.status == RobotStatus.DELAYED:
            self.delay_steps_remaining -= 1
            if self.delay_steps_remaining <= 0:
                if path_to_packing and len(path_to_packing) > 1:
                    self.status = RobotStatus.MOVING_TO_PACKING
                    self.path = path_to_packing
                    self.path_idx = 1
                else:
                    self.status = RobotStatus.DELIVERING
            return completed

        if self.status == RobotStatus.MOVING_TO_PACKING:
            if self.path_idx < len(self.path):
                self.position = self.path[self.path_idx]
                self.path_idx += 1
                self.distance_travelled += 1
            else:
                self.status = RobotStatus.DELIVERING
            return completed

        if self.status == RobotStatus.DELIVERING:
            if self.current_task:
                self.current_task.completed = True
                self.current_task.complete_step = current_step
            self.tasks_completed += 1
            self.current_task = None
            self.status = RobotStatus.IDLE
            completed = True
        return completed

    def state_snapshot(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "row": self.position[0],
            "col": self.position[1],
            "status": self.status.name,
            "tasks_completed": self.tasks_completed,
            "distance_travelled": self.distance_travelled,
            "sku": self.current_task.sku if self.current_task else None,
            "zone": self.current_task.zone if self.current_task else None,
        }
