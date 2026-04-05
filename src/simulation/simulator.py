"""Warehouse simulation engine linked to forecast demand and inventory pressure."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from src.simulation.pathfinding import astar
from src.simulation.robot import Robot, RobotStatus, Task
from src.simulation.warehouse import Warehouse
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WarehouseSimulator:
    """Multi-robot warehouse simulator driven by forecast volume."""

    def __init__(
        self,
        config: dict,
        forecast_df: Optional[pd.DataFrame] = None,
        inventory_df: Optional[pd.DataFrame] = None,
        slotting_df: Optional[pd.DataFrame] = None,
    ):
        sim_cfg = config["simulation"]
        self.warehouse = Warehouse(
            width=sim_cfg.get("grid_width", 20),
            height=sim_cfg.get("grid_height", 15),
            n_packing=sim_cfg.get("n_packing_stations", 3),
            seed=sim_cfg.get("seed", 42),
        )
        self.n_robots = int(sim_cfg.get("n_robots", 3))
        self.time_steps = int(sim_cfg.get("time_steps", 120))
        self.seed = int(sim_cfg.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)
        self.orders_per_step = float(sim_cfg.get("orders_per_step", 0.3))
        self.forecast_df = forecast_df.copy() if forecast_df is not None else pd.DataFrame()
        self.inventory_df = inventory_df.copy() if inventory_df is not None else pd.DataFrame()
        self.slotting_df = slotting_df.copy() if slotting_df is not None else pd.DataFrame()

        self.robots: list[Robot] = []
        self.task_queue: list[Task] = []
        self.completed_tasks: list[Task] = []
        self.all_tasks: list[Task] = []
        self.history: list[list[dict]] = []
        self.step_count = 0
        self.congestion_map = np.zeros((self.warehouse.height, self.warehouse.width), dtype=int)
        self._task_id_counter = 0
        self._obstacle_grid = self.warehouse.obstacle_grid()
        self._packing_positions = self.warehouse.get_packing_positions()
        self._sku_weights = self._build_sku_weights()
        self._slotting = self._build_slotting_map()
        self._inventory_pressure = self._build_inventory_pressure()
        self._tasks_by_zone = defaultdict(int)

    def reset(self) -> None:
        self.robots = []
        self.task_queue = []
        self.completed_tasks = []
        self.all_tasks = []
        self.history = []
        self.step_count = 0
        self.congestion_map = np.zeros((self.warehouse.height, self.warehouse.width), dtype=int)
        self._task_id_counter = 0
        self._tasks_by_zone = defaultdict(int)
        self.rng = np.random.default_rng(self.seed)

        start_row = self.warehouse.height - 2
        for index in range(self.n_robots):
            col = int(self.warehouse.width * (index + 1) / (self.n_robots + 1))
            self.robots.append(Robot(robot_id=index, start_pos=(start_row, col)))

    def run(self) -> "WarehouseSimulator":
        self.reset()
        logger.info("Running warehouse simulation for %s steps with %s robots", self.time_steps, self.n_robots)
        for _ in range(self.time_steps):
            self.step()
        return self

    def step(self) -> None:
        self._generate_tasks()
        self._assign_tasks()

        for robot in self.robots:
            path_to_packing = None
            if robot.current_task and robot.status in {RobotStatus.PICKING, RobotStatus.DELAYED, RobotStatus.MOVING_TO_PACKING}:
                occupied = {other.position for other in self.robots if other.robot_id != robot.robot_id}
                path_to_packing = astar(self._obstacle_grid, robot.position, robot.current_task.packing, occupied) or [
                    robot.position,
                    robot.current_task.packing,
                ]
            robot.step(path_to_packing=path_to_packing, current_step=self.step_count)
            row, col = robot.position
            self.congestion_map[row][col] += 1

        for task in self.task_queue:
            task.queue_delay += 1

        for task in self.all_tasks:
            if task.completed and task not in self.completed_tasks:
                self.completed_tasks.append(task)

        self.history.append([robot.state_snapshot() for robot in self.robots])
        self.step_count += 1

    def get_metrics(self) -> dict:
        completed = self.completed_tasks
        total = len(self.all_tasks)
        completed_total = len(completed)
        total_times = [task.total_time for task in completed if task.total_time is not None]
        queue_delays = [task.queue_delay for task in self.all_tasks]
        robot_utils = [
            (robot.steps_taken - robot.idle_steps) / max(robot.steps_taken, 1)
            for robot in self.robots
        ]
        delayed_orders = sum(1 for task in completed if (task.total_time or 0) > 18 or task.shortage_delay > 0)
        backlog = len(self.task_queue)

        zone_pressure = pd.Series(self._tasks_by_zone).sort_index().to_dict()
        return {
            "tasks_generated": total,
            "tasks_completed": completed_total,
            "tasks_pending": backlog,
            "fulfilment_rate": round(completed_total / max(total, 1) * 100, 1),
            "avg_fulfilment_time": round(float(np.mean(total_times)) if total_times else 0.0, 1),
            "avg_queue_delay": round(float(np.mean(queue_delays)) if queue_delays else 0.0, 1),
            "avg_robot_utilisation": round(float(np.mean(robot_utils)) * 100 if robot_utils else 0.0, 1),
            "delayed_orders": delayed_orders,
            "zone_task_mix": zone_pressure,
            "total_steps": self.step_count,
            "inventory_linked_delays": int(sum(task.shortage_delay > 0 for task in self.all_tasks)),
        }

    def get_robot_paths_df(self) -> pd.DataFrame:
        rows: list[dict] = []
        for step_idx, snapshots in enumerate(self.history):
            for snapshot in snapshots:
                rows.append({"step": step_idx, **snapshot})
        return pd.DataFrame(rows)

    def get_congestion_df(self) -> pd.DataFrame:
        rows: list[dict] = []
        for row in range(self.warehouse.height):
            for col in range(self.warehouse.width):
                rows.append(
                    {
                        "row": row,
                        "col": col,
                        "visits": int(self.congestion_map[row][col]),
                        "cell_type": int(self.warehouse.grid[row][col]),
                        "zone": self.warehouse.zone_for_position((row, col)),
                    }
                )
        return pd.DataFrame(rows)

    def get_task_log(self) -> pd.DataFrame:
        rows = [
            {
                "task_id": task.task_id,
                "sku": task.sku,
                "zone": task.zone,
                "priority": task.priority,
                "shortage_delay": task.shortage_delay,
                "queue_delay": task.queue_delay,
                "total_time": task.total_time,
                "completed": task.completed,
            }
            for task in self.all_tasks
        ]
        return pd.DataFrame(rows)

    def _generate_tasks(self) -> None:
        if not self._sku_weights:
            return
        expected_orders = max(self.orders_per_step * len(self._sku_weights), 0.2)
        n_new = int(self.rng.poisson(expected_orders))
        if n_new == 0:
            return

        skus = list(self._sku_weights.keys())
        probs = np.array(list(self._sku_weights.values()), dtype=float)
        probs = probs / probs.sum()
        selected = self.rng.choice(skus, size=n_new, replace=True, p=probs)

        for sku in selected:
            slot = self._slotting[sku]
            packing = self._packing_positions[int(self.rng.integers(0, len(self._packing_positions)))]
            pressure = self._inventory_pressure.get(sku, {})
            shortage_delay = int(pressure.get("shortage_delay", 0))
            zone_penalty = 1.0 + self._tasks_by_zone[slot["zone"]] * 0.015
            priority = float(self._sku_weights[sku] * 10 * zone_penalty)
            task = Task(
                task_id=self._task_id_counter,
                sku=sku,
                shelf=slot["access"],
                packing=packing,
                priority=priority,
                zone=slot["zone"],
                shortage_delay=shortage_delay,
            )
            task.created_step = self.step_count
            self.task_queue.append(task)
            self.all_tasks.append(task)
            self._tasks_by_zone[slot["zone"]] += 1
            self._task_id_counter += 1

    def _assign_tasks(self) -> None:
        idle_robots = [robot for robot in self.robots if robot.is_idle]
        if not idle_robots or not self.task_queue:
            return

        self.task_queue.sort(key=lambda task: (-task.priority, task.queue_delay, task.task_id))
        for robot in idle_robots:
            if not self.task_queue:
                break
            task = self.task_queue.pop(0)
            occupied = {other.position for other in self.robots if other.robot_id != robot.robot_id}
            path = astar(self._obstacle_grid, robot.position, task.shelf, occupied)
            if path is None:
                self.task_queue.append(task)
                continue
            robot.assign_task(task, path, self.step_count)

    def _build_sku_weights(self) -> dict[str, float]:
        if self.forecast_df.empty:
            return {}
        demand_weights = self.forecast_df.groupby("sku")["forecast"].sum()
        demand_weights = demand_weights / demand_weights.sum()
        return demand_weights.to_dict()

    def _build_slotting_map(self) -> dict[str, dict]:
        shelves = self.warehouse.get_shelf_positions()
        access_points = [self._find_shelf_access(shelf) for shelf in shelves]
        if self.slotting_df.empty:
            ranked_skus = sorted(self._sku_weights, key=self._sku_weights.get, reverse=True)
        else:
            ranked_skus = self.slotting_df.sort_values("slot_rank")["sku"].tolist()

        pack_row = self.warehouse.height - 1
        ranked_access = sorted(access_points, key=lambda pos: (abs(pos[0] - pack_row), pos[1]))
        slotting: dict[str, dict] = {}
        for index, sku in enumerate(ranked_skus):
            access = ranked_access[index % len(ranked_access)]
            slotting[sku] = {"access": access, "zone": self.warehouse.zone_for_position(access)}
        return slotting

    def _build_inventory_pressure(self) -> dict[str, dict]:
        pressure: dict[str, dict] = {}
        if self.inventory_df.empty:
            return pressure
        for _, row in self.inventory_df.iterrows():
            risk = row.get("stockout_risk", "low")
            if risk == "critical":
                delay = 4
            elif risk == "high":
                delay = 2
            else:
                delay = 0
            pressure[row["sku"]] = {"shortage_delay": delay}
        return pressure

    def _find_shelf_access(self, shelf: tuple[int, int]) -> tuple[int, int]:
        row, col = shelf
        for d_row, d_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_row, next_col = row + d_row, col + d_col
            if (
                0 <= next_row < self.warehouse.height
                and 0 <= next_col < self.warehouse.width
                and self._obstacle_grid[next_row][next_col] == 0
            ):
                return next_row, next_col
        return shelf
