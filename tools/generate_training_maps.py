import os

from Algo.Astar import astar_route
from rl_library import canonical_edge, save_map_json
from src.map import LineMap


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "maps", "training")
CUSTOM_OUTPUT_PATH = os.path.join(
    ROOT_DIR,
    "maps",
    "custom",
    "map_18x12_challenge.json",
)

MAP_WIDTH = 18
MAP_HEIGHT = 12
START = (0, 0)
GOAL = (MAP_WIDTH - 1, MAP_HEIGHT - 1)


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


def create_map(blocked_nodes=None, blocked_edges=None):
    line_map = LineMap()
    line_map.createMap((MAP_WIDTH, MAP_HEIGHT))
    for node in blocked_nodes or []:
        line_map.set_obstacles(node=node)
    for edge in blocked_edges or []:
        line_map.set_obstacles(edge=edge)
    return line_map


def generate_obstacles(checkpoints, seed, node_level, edge_level):
    safe_path = build_line_path(START, list(checkpoints) + [GOAL])
    protected_nodes = set(safe_path)
    protected_edges = protected_edges_from_path(safe_path)

    blocked_nodes = []
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            coord = (x, y)
            if coord in protected_nodes:
                continue
            value = (x * 37 + y * 17 + seed * 13 + x * y * 3) % 23
            if value < node_level:
                blocked_nodes.append(coord)

    blocked_edges = []
    for x in range(MAP_WIDTH):
        for y in range(MAP_HEIGHT):
            for neighbor in ((x + 1, y), (x, y + 1)):
                if neighbor[0] >= MAP_WIDTH or neighbor[1] >= MAP_HEIGHT:
                    continue
                edge = canonical_edge((x, y), neighbor)
                if edge in protected_edges:
                    continue
                value = (
                    x * 19
                    + y * 23
                    + neighbor[0] * 29
                    + neighbor[1] * 31
                    + seed * 7
                ) % 29
                if value < edge_level:
                    blocked_edges.append(((x, y), neighbor))

    return blocked_nodes, blocked_edges


def save_training_map(
    index,
    slug,
    checkpoints,
    blocked_nodes,
    blocked_edges,
    difficulty,
):
    line_map = create_map(blocked_nodes, blocked_edges)
    if astar_route(line_map, START, GOAL, checkpoints) is None:
        raise ValueError(f"Generated training map {index} is not solvable.")

    filename = f"map_{index:02d}_{slug}.json"
    file_path = os.path.join(OUTPUT_DIR, filename)
    save_map_json(
        file_path=file_path,
        map_obj=line_map,
        start=START,
        goal=GOAL,
        checkpoints=checkpoints,
        name=slug.replace("_", " ").title(),
        metadata={
            "difficulty": difficulty,
            "training_order": index,
            "dimensions": "18x12",
            "lesson": "Students can edit reward, sensor handling, and map data.",
        },
    )


def starter_maps():
    map_2_nodes, map_2_edges = generate_obstacles(
        checkpoints=[],
        seed=2,
        node_level=1,
        edge_level=1,
    )
    map_4_checkpoints = [(2, 9), (15, 2)]
    map_4_nodes, map_4_edges = generate_obstacles(
        checkpoints=map_4_checkpoints,
        seed=4,
        node_level=2,
        edge_level=2,
    )
    map_5_checkpoints = [(2, 9), (15, 2), (9, 6)]
    map_5_nodes, map_5_edges = generate_obstacles(
        checkpoints=map_5_checkpoints,
        seed=5,
        node_level=3,
        edge_level=3,
    )

    return [
        ("empty_grid", [], [], [], "starter"),
        (
            "sparse_obstacles",
            [],
            map_2_nodes,
            map_2_edges,
            "starter",
        ),
        (
            "empty_with_checkpoints",
            [(1, 10), (16, 1)],
            [],
            [],
            "starter",
        ),
        (
            "sparse_with_checkpoints",
            map_4_checkpoints,
            map_4_nodes,
            map_4_edges,
            "starter",
        ),
        (
            "intro_complex",
            map_5_checkpoints,
            map_5_nodes,
            map_5_edges,
            "intro_complex",
        ),
    ]


def complex_map(index):
    complex_index = index - 5
    checkpoint_pool = [
        (1, 10),
        (16, 10),
        (16, 1),
        (9, 6),
        (5, 8),
        (12, 3),
    ]
    offset = (complex_index * 2) % len(checkpoint_pool)
    rotated_pool = (
        checkpoint_pool[offset:]
        + checkpoint_pool[:offset]
    )
    checkpoint_count = 2 + (complex_index % 3)
    checkpoints = rotated_pool[:checkpoint_count]

    difficulty_stage = (complex_index - 1) // 8
    obstacle_level = min(4, 2 + difficulty_stage)
    blocked_nodes, blocked_edges = generate_obstacles(
        checkpoints=checkpoints,
        seed=complex_index + 20,
        node_level=obstacle_level,
        edge_level=obstacle_level,
    )
    return (
        f"complex_multi_goal_{complex_index:02d}",
        checkpoints,
        blocked_nodes,
        blocked_edges,
        "complex",
    )


def save_custom_map():
    checkpoints = [(2, 10), (15, 9), (15, 2), (8, 5)]
    blocked_nodes, blocked_edges = generate_obstacles(
        checkpoints=checkpoints,
        seed=101,
        node_level=4,
        edge_level=4,
    )
    line_map = create_map(blocked_nodes, blocked_edges)
    if astar_route(line_map, START, GOAL, checkpoints) is None:
        raise ValueError("Generated custom map is not solvable.")

    save_map_json(
        file_path=CUSTOM_OUTPUT_PATH,
        map_obj=line_map,
        start=START,
        goal=GOAL,
        checkpoints=checkpoints,
        name="18x12 Custom Challenge",
        metadata={
            "source": "Summer School 2026 map generator",
            "difficulty": "custom_challenge",
            "dimensions": "18x12",
        },
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for index, definition in enumerate(starter_maps(), start=1):
        save_training_map(index, *definition)
    for index in range(6, 31):
        save_training_map(index, *complex_map(index))
    save_custom_map()


if __name__ == "__main__":
    main()
