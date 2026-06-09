import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_library import canonical_edge, save_map_json
from src.map import LineMap


OUTPUT_DIR = os.path.join(ROOT_DIR, "maps", "training")


def build_line_path(start, targets):
    path = [start]
    current = start
    for target in targets:
        x, y = current
        tx, ty = target
        while x != tx:
            x += 1 if tx > x else -1
            path.append((x, y))
        while y != ty:
            y += 1 if ty > y else -1
            path.append((x, y))
        current = target
    return path


def protected_edges_from_path(path):
    return {
        canonical_edge(path[index], path[index + 1])
        for index in range(len(path) - 1)
    }


def create_map(width, height, blocked_nodes=None, blocked_edges=None):
    line_map = LineMap()
    line_map.createMap((width, height))
    for node in blocked_nodes or []:
        line_map.set_obstacles(node=node)
    for edge in blocked_edges or []:
        line_map.set_obstacles(edge=edge)
    return line_map


def save_training_map(index, slug, width, height, start, goal, checkpoints, blocked_nodes, blocked_edges, difficulty):
    line_map = create_map(width, height, blocked_nodes, blocked_edges)
    filename = f"map_{index:02d}_{slug}.json"
    file_path = os.path.join(OUTPUT_DIR, filename)
    save_map_json(
        file_path=file_path,
        map_obj=line_map,
        start=start,
        goal=goal,
        checkpoints=checkpoints,
        name=slug.replace("_", " ").title(),
        metadata={
            "difficulty": difficulty,
            "training_order": index,
            "lesson": "Students can edit reward, sensor handling, and map data.",
        },
    )


def complex_map(index):
    complex_index = index - 5
    width = 8 + (complex_index % 5)
    height = 8 + ((complex_index * 2) % 5)
    start = (0, 0)
    goal = (width - 1, height - 1)

    checkpoint_pool = [
        (1, height - 2),
        (width // 2, height // 2),
        (width - 2, 1),
        (max(1, width - 3), max(1, height - 3)),
    ]
    checkpoint_count = 2 + (complex_index % 3)
    checkpoints = []
    for coord in checkpoint_pool:
        if coord not in checkpoints and coord not in (start, goal):
            checkpoints.append(coord)
        if len(checkpoints) == checkpoint_count:
            break

    safe_path = build_line_path(start, checkpoints + [goal])
    safe_nodes = set(safe_path)
    safe_edges = protected_edges_from_path(safe_path)
    protected_nodes = safe_nodes | {start, goal, *checkpoints}

    blocked_nodes = []
    for x in range(width):
        for y in range(height):
            coord = (x, y)
            if coord in protected_nodes:
                continue
            value = (x * 37 + y * 17 + complex_index * 13) % 10
            if value in (0, 3):
                blocked_nodes.append(coord)

    blocked_edges = []
    for x in range(width):
        for y in range(height):
            for neighbor in ((x + 1, y), (x, y + 1)):
                if neighbor[0] >= width or neighbor[1] >= height:
                    continue
                edge = canonical_edge((x, y), neighbor)
                if edge in safe_edges:
                    continue
                value = (x * 19 + y * 23 + neighbor[0] * 29 + neighbor[1] * 31 + complex_index * 7) % 13
                if value in (0, 5):
                    blocked_edges.append(((x, y), neighbor))

    return (
        f"complex_multi_goal_{complex_index:02d}",
        width,
        height,
        start,
        goal,
        checkpoints,
        blocked_nodes,
        blocked_edges,
        "complex",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    starter_maps = [
        (
            "empty_grid",
            5,
            5,
            (0, 0),
            (4, 4),
            [],
            [],
            [],
            "starter",
        ),
        (
            "sparse_obstacles",
            5,
            5,
            (0, 0),
            (4, 4),
            [],
            [(2, 2)],
            [((1, 1), (1, 2)), ((3, 3), (3, 4))],
            "starter",
        ),
        (
            "empty_with_checkpoints",
            6,
            6,
            (0, 0),
            (5, 5),
            [(0, 5), (5, 0)],
            [],
            [],
            "starter",
        ),
        (
            "sparse_with_checkpoints",
            6,
            6,
            (0, 0),
            (5, 5),
            [(1, 4), (4, 1)],
            [(2, 2), (3, 3)],
            [((0, 2), (1, 2)), ((4, 3), (5, 3))],
            "starter",
        ),
        (
            "intro_complex",
            7,
            7,
            (0, 0),
            (6, 6),
            [(1, 5), (5, 1), (3, 3)],
            [(2, 1), (2, 2), (4, 4), (5, 4)],
            [((0, 3), (1, 3)), ((3, 1), (3, 2)), ((4, 5), (5, 5))],
            "intro_complex",
        ),
    ]

    for index, definition in enumerate(starter_maps, start=1):
        save_training_map(index, *definition)

    for index in range(6, 31):
        save_training_map(index, *complex_map(index))


if __name__ == "__main__":
    main()
