"""Warehouse grid environment definition."""

import numpy as np
from enum import IntEnum
from typing import List, Tuple


class Cell(IntEnum):
    EMPTY = 0
    SHELF = 1        # obstacle / pick location
    AISLE = 2        # walkable aisle (rendered differently)
    PACKING = 3      # delivery destination


class Warehouse:
    """
    2D warehouse grid.

    Layout (default 20 wide × 15 tall):
      - Row 0: empty top buffer
      - Rows 1–11: shelf rows with aisles in between
      - Rows 12–14: bottom packing/staging area
    """

    def __init__(self, width: int = 20, height: int = 15, n_packing: int = 3, seed: int = 42):
        self.width = width
        self.height = height
        self.n_packing = n_packing
        self.seed = seed
        self.grid: List[List[int]] = []
        self.shelf_positions: List[Tuple[int, int]] = []
        self.packing_positions: List[Tuple[int, int]] = []
        self._build()

    def _build(self):
        grid = [[Cell.EMPTY] * self.width for _ in range(self.height)]

        # Place shelf rows in pairs with an aisle between each pair
        shelf_rows = range(1, self.height - 3, 3)  # rows 1, 4, 7, 10
        for row in shelf_rows:
            for col in range(1, self.width - 1):
                grid[row][col] = Cell.SHELF
                self.shelf_positions.append((row, col))

        # Packing stations at bottom
        pack_cols = [int(self.width * (i + 1) / (self.n_packing + 1))
                     for i in range(self.n_packing)]
        pack_row = self.height - 1
        for col in pack_cols:
            grid[pack_row][col] = Cell.PACKING
            self.packing_positions.append((pack_row, col))

        self.grid = grid

    def is_obstacle(self, row: int, col: int) -> bool:
        """Return True if cell is a shelf (not passable for robots)."""
        return self.grid[row][col] == Cell.SHELF

    def obstacle_grid(self) -> List[List[int]]:
        """Return grid with 1 for obstacles (shelves), 0 for walkable."""
        return [
            [1 if cell == Cell.SHELF else 0 for cell in row]
            for row in self.grid
        ]

    def get_shelf_positions(self) -> List[Tuple[int, int]]:
        return list(self.shelf_positions)

    def get_packing_positions(self) -> List[Tuple[int, int]]:
        return list(self.packing_positions)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.grid, dtype=int)
