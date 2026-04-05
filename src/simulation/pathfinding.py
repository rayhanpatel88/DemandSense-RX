"""A* pathfinding algorithm for warehouse robot navigation."""

import heapq
from typing import List, Tuple, Optional


def astar(
    grid: list,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    occupied: set = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on a 2D grid.

    Parameters
    ----------
    grid : 2D list where 1 = obstacle, 0 = walkable
    start : (row, col) start position
    goal : (row, col) goal position
    occupied : set of (row, col) positions blocked by other robots

    Returns
    -------
    List of (row, col) waypoints from start to goal (inclusive), or None if no path.
    """
    if occupied is None:
        occupied = set()

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def neighbours(pos):
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 1:  # not a wall
                    if (nr, nc) not in occupied or (nr, nc) == goal:
                        yield (nr, nc)

    open_heap = []
    heapq.heappush(open_heap, (h(start), 0, start, [start]))
    visited = {start: 0}

    while open_heap:
        f, g, current, path = heapq.heappop(open_heap)

        if current == goal:
            return path

        for nb in neighbours(current):
            new_g = g + 1
            if nb not in visited or new_g < visited[nb]:
                visited[nb] = new_g
                new_f = new_g + h(nb)
                heapq.heappush(open_heap, (new_f, new_g, nb, path + [nb]))

    return None  # No path found
