"""Robot agent class for warehouse simulation."""

from enum import Enum, auto
from typing import Optional, Tuple, List


class RobotStatus(Enum):
    IDLE = auto()
    MOVING_TO_SHELF = auto()
    PICKING = auto()
    MOVING_TO_PACKING = auto()
    DELIVERING = auto()


class Task:
    def __init__(self, task_id: int, shelf: Tuple[int, int],
                 packing: Tuple[int, int], priority: float = 1.0):
        self.task_id = task_id
        self.shelf = shelf
        self.packing = packing
        self.priority = priority
        self.assigned_robot: Optional[int] = None
        self.completed = False
        self.start_step: Optional[int] = None
        self.pick_step: Optional[int] = None
        self.complete_step: Optional[int] = None

    @property
    def total_time(self) -> Optional[int]:
        if self.start_step is not None and self.complete_step is not None:
            return self.complete_step - self.start_step
        return None


class Robot:
    """Autonomous warehouse robot agent."""

    def __init__(self, robot_id: int, start_pos: Tuple[int, int]):
        self.robot_id = robot_id
        self.position: Tuple[int, int] = start_pos
        self.status: RobotStatus = RobotStatus.IDLE
        self.current_task: Optional[Task] = None
        self.path: List[Tuple[int, int]] = []
        self.path_idx: int = 0
        self.steps_taken: int = 0
        self.tasks_completed: int = 0
        self.distance_travelled: int = 0
        self.idle_steps: int = 0
        self.picking_steps_remaining: int = 0

    @property
    def is_idle(self) -> bool:
        return self.status == RobotStatus.IDLE

    def assign_task(self, task: Task, path_to_shelf: List[Tuple[int, int]],
                    current_step: int):
        """Assign a new task and set path to shelf."""
        self.current_task = task
        task.assigned_robot = self.robot_id
        task.start_step = current_step

        if path_to_shelf and len(path_to_shelf) > 1:
            self.path = path_to_shelf
            self.path_idx = 1  # skip current position
            self.status = RobotStatus.MOVING_TO_SHELF
        else:
            # Already at shelf
            self.status = RobotStatus.PICKING
            self.picking_steps_remaining = 2

    def step(self, path_to_packing: List[Tuple[int, int]] = None,
             current_step: int = 0) -> bool:
        """
        Advance robot one time step.

        Returns True if a task was just completed.
        """
        self.steps_taken += 1
        completed = False

        if self.status == RobotStatus.IDLE:
            self.idle_steps += 1

        elif self.status == RobotStatus.MOVING_TO_SHELF:
            if self.path_idx < len(self.path):
                self.position = self.path[self.path_idx]
                self.path_idx += 1
                self.distance_travelled += 1
            else:
                # Arrived at shelf
                self.status = RobotStatus.PICKING
                self.picking_steps_remaining = 2
                if self.current_task:
                    self.current_task.pick_step = current_step

        elif self.status == RobotStatus.PICKING:
            self.picking_steps_remaining -= 1
            if self.picking_steps_remaining <= 0:
                self.status = RobotStatus.MOVING_TO_PACKING
                if path_to_packing and len(path_to_packing) > 1:
                    self.path = path_to_packing
                    self.path_idx = 1
                else:
                    self.status = RobotStatus.DELIVERING

        elif self.status == RobotStatus.MOVING_TO_PACKING:
            if self.path_idx < len(self.path):
                self.position = self.path[self.path_idx]
                self.path_idx += 1
                self.distance_travelled += 1
            else:
                self.status = RobotStatus.DELIVERING

        elif self.status == RobotStatus.DELIVERING:
            # One step to deliver
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
        }
