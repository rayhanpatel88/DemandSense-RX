"""Unit tests for A* pathfinding algorithm."""

import pytest
from src.simulation.pathfinding import astar


def empty_grid(rows, cols):
    return [[0] * cols for _ in range(rows)]


def test_direct_path():
    grid = empty_grid(5, 5)
    path = astar(grid, (0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_path_length_manhattan():
    grid = empty_grid(5, 5)
    path = astar(grid, (0, 0), (0, 4))
    # Manhattan distance = 4, path has 5 nodes
    assert path is not None
    assert len(path) == 5


def test_obstacle_avoidance():
    grid = empty_grid(5, 5)
    # Block column 2 except top row, so robot must go around via row 0
    for r in range(1, 5):
        grid[r][2] = 1
    path = astar(grid, (2, 0), (2, 4))
    assert path is not None
    assert path[-1] == (2, 4)
    # Path must not pass through blocked cells
    for r, c in path:
        assert grid[r][c] != 1, "Path should not pass through obstacles"


def test_no_path():
    grid = empty_grid(3, 3)
    # Completely wall off the goal
    grid[0][2] = 1
    grid[1][2] = 1
    grid[2][2] = 1
    path = astar(grid, (1, 0), (1, 2))
    assert path is None


def test_start_equals_goal():
    grid = empty_grid(5, 5)
    path = astar(grid, (2, 2), (2, 2))
    assert path is not None
    assert path == [(2, 2)]


def test_adjacent_goal():
    grid = empty_grid(5, 5)
    path = astar(grid, (2, 2), (2, 3))
    assert path is not None
    assert len(path) == 2


def test_occupied_cells_avoided():
    grid = empty_grid(5, 5)
    occupied = {(2, 1), (2, 2), (2, 3)}
    path = astar(grid, (2, 0), (2, 4), occupied=occupied)
    # Should find a path around occupied cells (goal is always reachable)
    assert path is not None
    assert path[-1] == (2, 4)
    for pos in path[1:-1]:  # Don't check start/goal
        assert pos not in occupied, "Path should avoid occupied cells"


def test_path_continuity():
    grid = empty_grid(7, 7)
    path = astar(grid, (0, 0), (6, 6))
    assert path is not None
    # Each step should be adjacent (Manhattan distance 1)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Each step must be adjacent"
