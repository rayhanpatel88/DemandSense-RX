"""Main warehouse simulation engine."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from src.simulation.warehouse import Warehouse
from src.simulation.robot import Robot, Task, RobotStatus
from src.simulation.pathfinding import astar
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WarehouseSimulator:
    """
    Multi-robot warehouse fulfilment simulator.

    Robots autonomously fulfil pick-and-deliver tasks using A* pathfinding.
    """

    def __init__(self, config: dict, demand_priorities: dict = None):
        sim_cfg = config["simulation"]
        self.warehouse = Warehouse(
            width=sim_cfg.get("grid_width", 20),
            height=sim_cfg.get("grid_height", 15),
            n_packing=sim_cfg.get("n_packing_stations", 3),
            seed=sim_cfg.get("seed", 42),
        )
        self.n_robots = sim_cfg.get("n_robots", 3)
        self.time_steps = sim_cfg.get("time_steps", 100)
        self.orders_per_step = sim_cfg.get("orders_per_step", 0.3)
        self.seed = sim_cfg.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.demand_priorities = demand_priorities or {}

        self.robots: List[Robot] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.all_tasks: List[Task] = []
        self.step_count: int = 0
        self.history: List[List[dict]] = []  # robot snapshots per step
        self.congestion_map: np.ndarray = np.zeros(
            (self.warehouse.height, self.warehouse.width), dtype=int
        )
        self._task_id_counter = 0
        self._obstacle_grid = self.warehouse.obstacle_grid()
        self._shelf_positions = self.warehouse.get_shelf_positions()
        self._packing_positions = self.warehouse.get_packing_positions()

    def reset(self):
        """Initialise robots at bottom of warehouse."""
        self.robots = []
        self.task_queue = []
        self.completed_tasks = []
        self.all_tasks = []
        self.step_count = 0
        self.history = []
        self.congestion_map = np.zeros(
            (self.warehouse.height, self.warehouse.width), dtype=int
        )
        self._task_id_counter = 0
        self.rng = np.random.default_rng(self.seed)

        pack_row = self.warehouse.height - 2
        for i in range(self.n_robots):
            col = int(self.warehouse.width * (i + 1) / (self.n_robots + 1))
            self.robots.append(Robot(robot_id=i, start_pos=(pack_row, col)))

    def _generate_task(self) -> Task:
        """Create a new pick-and-deliver task."""
        shelf = self._shelf_positions[
            int(self.rng.integers(0, len(self._shelf_positions)))
        ]
        # Find a walkable cell adjacent to the shelf for robot to stand
        shelf_access = self._find_shelf_access(shelf)
        packing = self._packing_positions[
            int(self.rng.integers(0, len(self._packing_positions)))
        ]
        priority = float(self.rng.uniform(0.5, 2.0))
        task = Task(self._task_id_counter, shelf_access, packing, priority)
        self._task_id_counter += 1
        return task

    def _find_shelf_access(self, shelf: tuple) -> tuple:
        """Find a walkable cell adjacent to a shelf."""
        r, c = shelf
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.warehouse.height and
                    0 <= nc < self.warehouse.width and
                    self._obstacle_grid[nr][nc] == 0):
                return (nr, nc)
        return (r, c)  # fallback

    def _assign_tasks(self):
        """Assign queued tasks to idle robots (highest priority first)."""
        if not self.task_queue:
            return
        idle_robots = [r for r in self.robots if r.is_idle]
        if not idle_robots:
            return

        self.task_queue.sort(key=lambda t: -t.priority)

        for robot in idle_robots:
            if not self.task_queue:
                break
            task = self.task_queue.pop(0)
            occupied = {r.position for r in self.robots if r.robot_id != robot.robot_id}
            path = astar(self._obstacle_grid, robot.position, task.shelf, occupied)
            if path is None:
                # No path found, skip and try again next step
                self.task_queue.insert(0, task)
                continue
            robot.assign_task(task, path, self.step_count)

    def step(self):
        """Advance simulation by one time step."""
        # Generate new orders stochastically
        if self.rng.random() < self.orders_per_step:
            n_new = int(self.rng.integers(1, 4))
            for _ in range(n_new):
                task = self._generate_task()
                self.task_queue.append(task)
                self.all_tasks.append(task)

        # Assign tasks to idle robots
        self._assign_tasks()

        # Step each robot
        for robot in self.robots:
            path_to_packing = None
            if robot.status == RobotStatus.PICKING and robot.current_task:
                occupied = {r.position for r in self.robots if r.robot_id != robot.robot_id}
                packing = robot.current_task.packing
                path_to_packing = astar(
                    self._obstacle_grid, robot.position, packing, occupied
                ) or [robot.position, packing]

            completed = robot.step(path_to_packing, self.step_count)
            if completed and robot.current_task is None:
                # Task was just completed (current_task cleared inside robot.step)
                pass

            # Update congestion map
            r, c = robot.position
            self.congestion_map[r][c] += 1

        # Move completed tasks
        for task in self.all_tasks:
            if task.completed and task not in self.completed_tasks:
                self.completed_tasks.append(task)

        # Snapshot state
        self.history.append([r.state_snapshot() for r in self.robots])
        self.step_count += 1

    def run(self) -> "WarehouseSimulator":
        """Run the full simulation."""
        self.reset()
        logger.info(f"Running simulation: {self.time_steps} steps, {self.n_robots} robots")
        for _ in range(self.time_steps):
            self.step()
        logger.info(f"Simulation complete: {len(self.completed_tasks)} tasks completed")
        return self

    def get_metrics(self) -> dict:
        """Compute summary performance metrics."""
        completed = self.completed_tasks
        n_completed = len(completed)
        n_total = len(self.all_tasks)

        total_times = [t.total_time for t in completed if t.total_time is not None]
        avg_pick_time = float(np.mean(total_times)) if total_times else 0.0

        robot_utils = []
        for robot in self.robots:
            active = robot.steps_taken - robot.idle_steps
            util = active / max(robot.steps_taken, 1)
            robot_utils.append(util)

        avg_util = float(np.mean(robot_utils)) if robot_utils else 0.0
        fulfilment_rate = n_completed / max(n_total, 1)
        delayed = sum(1 for t in completed if (t.total_time or 0) > 20)

        return {
            "tasks_generated": n_total,
            "tasks_completed": n_completed,
            "tasks_pending": len(self.task_queue),
            "fulfilment_rate": round(fulfilment_rate * 100, 1),
            "avg_fulfilment_time": round(avg_pick_time, 1),
            "avg_robot_utilisation": round(avg_util * 100, 1),
            "delayed_orders": delayed,
            "total_steps": self.step_count,
        }

    def get_robot_paths_df(self) -> pd.DataFrame:
        """Return robot positions over time as a DataFrame."""
        rows = []
        for step_idx, snapshots in enumerate(self.history):
            for snap in snapshots:
                rows.append({"step": step_idx, **snap})
        return pd.DataFrame(rows)

    def get_congestion_df(self) -> pd.DataFrame:
        """Return congestion heatmap as a DataFrame."""
        rows = []
        for r in range(self.warehouse.height):
            for c in range(self.warehouse.width):
                rows.append({
                    "row": r, "col": c,
                    "visits": int(self.congestion_map[r][c]),
                    "cell_type": int(self.warehouse.grid[r][c]),
                })
        return pd.DataFrame(rows)
