import heapq


def _heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _get_neighbors(map_obj, coord):
    """Returns list of (neighbor_coord, weight) for navigable neighbors from coord."""
    neighbors = []
    for edge in map_obj.edges:
        if (edge.src.x, edge.src.y) == coord and not edge.isBlock:
            dest_coord = (edge.dest.x, edge.dest.y)
            dest_node = map_obj.nodes.get(dest_coord)
            if dest_node and not dest_node.isBlock:
                neighbors.append((dest_coord, edge.weight))
    return neighbors


def astar(map_obj, start, goal):
    """
    A* pathfinding on a LineMap.

    Args:
        map_obj (LineMap): The map to search on.
        start (tuple): Starting coordinate (x, y).
        goal (tuple): Goal coordinate (x, y).

    Returns:
        list: Ordered list of (x, y) coordinates from start to goal (inclusive),
              or None if no path exists.
    """
    if start not in map_obj.nodes or goal not in map_obj.nodes:
        return None
    if map_obj.nodes[goal].isBlock:
        return None
    if start == goal:
        return [start]

    # Each entry: (f_score, g_score, coord)
    open_set = []
    heapq.heappush(open_set, (_heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Skip stale heap entries
        if cost > g_score.get(current, float('inf')):
            continue

        for neighbor, weight in _get_neighbors(map_obj, current):
            tentative_g = g_score[current] + weight
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None  # No path found
