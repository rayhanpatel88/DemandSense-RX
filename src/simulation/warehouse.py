"""Warehouse grid environment definition."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Cell(IntEnum):
    EMPTY = 0
    SHELF = 1
    AISLE = 2
    PACKING = 3


class Warehouse:
    """2D warehouse layout with aisle-accessible shelf slots."""

    def __init__(self, width: int = 20, height: int = 15, n_packing: int = 3, seed: int = 42):
        self.width = width
        self.height = height
        self.n_packing = n_packing
        self.seed = seed
        self.grid: list[list[int]] = []
        self.shelf_positions: list[tuple[int, int]] = []
        self.packing_positions: list[tuple[int, int]] = []
        self._build()

    def _build(self) -> None:
        grid = [[Cell.EMPTY] * self.width for _ in range(self.height)]
        for row in range(1, self.height - 3):
            for col in range(1, self.width - 1):
                if row % 3 == 1:
                    grid[row][col] = Cell.SHELF
                    self.shelf_positions.append((row, col))
                else:
                    grid[row][col] = Cell.AISLE

        pack_cols = [int(self.width * (index + 1) / (self.n_packing + 1)) for index in range(self.n_packing)]
        for col in pack_cols:
            grid[self.height - 1][col] = Cell.PACKING
            self.packing_positions.append((self.height - 1, col))
        self.grid = grid

    def obstacle_grid(self) -> list[list[int]]:
        return [[1 if cell == Cell.SHELF else 0 for cell in row] for row in self.grid]

    def get_shelf_positions(self) -> list[tuple[int, int]]:
        return list(self.shelf_positions)

    def get_packing_positions(self) -> list[tuple[int, int]]:
        return list(self.packing_positions)

    def zone_for_position(self, position: tuple[int, int]) -> str:
        _, col = position
        if col < self.width / 3:
            return "A"
        if col < 2 * self.width / 3:
            return "B"
        return "C"

    def to_numpy(self) -> np.ndarray:
        return np.array(self.grid, dtype=int)
