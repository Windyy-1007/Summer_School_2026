import json
import os
from dataclasses import dataclass, field

try:
    from src.map import LineMap
except ImportError:
    from map import LineMap


CHECKPOINT_SCR = 200
GOAL_SCR = 400
UNFINISH_SCR = -1000
MOVE_SCR = -4
TURN_SCR = -1

UP = (0, 1)
RIGHT = (1, 0)
DOWN = (0, -1)
LEFT = (-1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

FORWARD = "forward"
BACKWARD = "backward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
ACTIONS = [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT]

ACTION_LABELS = {
    FORWARD: "Move forward",
    BACKWARD: "Move backward",
    TURN_LEFT: "Turn left",
    TURN_RIGHT: "Turn right",
}


@dataclass(frozen=True)
class RLState:
    x: int
    y: int
    dx: int
    dy: int
    checkpoints_mask: int = 0

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def direction(self):
        return (self.dx, self.dy)


@dataclass
class TransitionResult:
    state: RLState
    action: str
    next_state: RLState
    reward: float
    moved: bool = False
    turned: bool = False
    hit_obstacle: bool = False
    reached_goal: bool = False
    reached_checkpoints: int = 0
    detected_node: tuple | None = None
    detected_edge: tuple | None = None
    message: str = ""


@dataclass
class MapData:
    line_map: LineMap
    name: str
    width: int
    height: int
    start: tuple
    goal: tuple
    checkpoints: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def manhattan(coord_a, coord_b):
    return abs(coord_a[0] - coord_b[0]) + abs(coord_a[1] - coord_b[1])


def min_manhattan(path, goal):
    if not path:
        return manhattan((0, 0), goal)
    return min(manhattan(coord, goal) for coord in path)


def calculate_score(
    num_moves,
    num_turns,
    checkpoints_reached,
    reached_goal,
    min_manhattan_distance,
):
    if reached_goal:
        move_penalty = num_moves * MOVE_SCR + num_turns * TURN_SCR
        checkpoint_score = checkpoints_reached * CHECKPOINT_SCR
        return move_penalty + checkpoint_score + GOAL_SCR
    return UNFINISH_SCR - int(min_manhattan_distance)


def turn_left(direction):
    dx, dy = direction
    return (-dy, dx)


def turn_right(direction):
    dx, dy = direction
    return (dy, -dx)


def direction_name(direction):
    names = {
        UP: "up",
        RIGHT: "right",
        DOWN: "down",
        LEFT: "left",
    }
    return names.get(direction, str(direction))


def make_checkpoint_mask(reached_checkpoints=None, checkpoints=None):
    if reached_checkpoints is None:
        return 0
    if isinstance(reached_checkpoints, int):
        return reached_checkpoints

    checkpoint_list = list(checkpoints or [])
    reached = set(tuple(coord) for coord in reached_checkpoints)
    mask = 0
    for index, coord in enumerate(checkpoint_list):
        if tuple(coord) in reached:
            mask |= 1 << index
    return mask


def update_checkpoint_mask(position, checkpoints=None, checkpoints_mask=0):
    mask = checkpoints_mask
    for index, checkpoint in enumerate(checkpoints or []):
        if tuple(checkpoint) == tuple(position):
            mask |= 1 << index
    return mask


def checkpoint_count(checkpoints_mask):
    return int(checkpoints_mask).bit_count()


def checkpoints_from_mask(checkpoints_mask, checkpoints=None):
    reached = []
    for index, checkpoint in enumerate(checkpoints or []):
        if checkpoints_mask & (1 << index):
            reached.append(tuple(checkpoint))
    return reached


def get_state(position, direction=UP, checkpoints=None, reached_checkpoints=None):
    mask = make_checkpoint_mask(reached_checkpoints, checkpoints)
    mask = update_checkpoint_mask(position, checkpoints, mask)
    return RLState(position[0], position[1], direction[0], direction[1], mask)


def ensure_state(state):
    if isinstance(state, RLState):
        return state
    if isinstance(state, tuple) and len(state) == 5:
        return RLState(state[0], state[1], state[2], state[3], state[4])
    raise TypeError("state must be an RLState or a 5-item tuple: (x, y, dx, dy, checkpoints_mask)")


def valid_actions(state=None, map_obj=None):
    return list(ACTIONS)


def map_dimensions(map_obj):
    if not map_obj.nodes:
        return (0, 0)
    width = max(x for x, _ in map_obj.nodes) + 1
    height = max(y for _, y in map_obj.nodes) + 1
    return (width, height)


def get_edge(map_obj, src_coord, dest_coord):
    for edge in map_obj.edges:
        if (edge.src.x, edge.src.y) == src_coord and (edge.dest.x, edge.dest.y) == dest_coord:
            return edge
    return None


def canonical_edge(coord_a, coord_b):
    first, second = sorted([tuple(coord_a), tuple(coord_b)])
    return (first, second)


def is_blocked_edge(map_obj, coord_a, coord_b):
    edge = get_edge(map_obj, coord_a, coord_b)
    return edge is None or bool(edge.isBlock)


def default_reward(next_state, goal):
    return -manhattan(next_state.position, goal)


def transition(map_obj, state, action, goal, checkpoints=None):
    state = ensure_state(state)
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    current = state.position
    direction = state.direction
    checkpoints = list(checkpoints or [])

    if action == TURN_LEFT:
        next_direction = turn_left(direction)
        next_state = RLState(state.x, state.y, next_direction[0], next_direction[1], state.checkpoints_mask)
        reward = default_reward(next_state, goal)
        return TransitionResult(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            turned=True,
            reached_goal=current == goal,
            reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
            message=f"Turn left. Heading is now {direction_name(next_direction)}.",
        )

    if action == TURN_RIGHT:
        next_direction = turn_right(direction)
        next_state = RLState(state.x, state.y, next_direction[0], next_direction[1], state.checkpoints_mask)
        reward = default_reward(next_state, goal)
        return TransitionResult(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            turned=True,
            reached_goal=current == goal,
            reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
            message=f"Turn right. Heading is now {direction_name(next_direction)}.",
        )

    move_direction = direction
    if action == BACKWARD:
        move_direction = (-direction[0], -direction[1])

    target = (state.x + move_direction[0], state.y + move_direction[1])
    next_state = state
    hit_obstacle = False
    detected_node = None
    detected_edge = None

    if current not in map_obj.nodes:
        hit_obstacle = True
        message = f"Current node {current} is outside the map."
    elif target not in map_obj.nodes:
        hit_obstacle = True
        message = f"Target {target} is outside the map. Robot stays at {current}."
    else:
        edge = get_edge(map_obj, current, target)
        if edge is None or edge.isBlock:
            hit_obstacle = True
            detected_edge = canonical_edge(current, target)
            if action == FORWARD:
                message = f"Sensor sees a blocked edge before moving from {current} to {target}."
            else:
                message = f"Backward move hits a blocked edge from {current} to {target}."
        elif map_obj.nodes[target].isBlock:
            hit_obstacle = True
            detected_node = target
            message = f"Robot moves halfway, detects blocked node {target}, then returns to {current}."
        else:
            mask = update_checkpoint_mask(target, checkpoints, state.checkpoints_mask)
            next_state = RLState(target[0], target[1], direction[0], direction[1], mask)
            message = f"Robot moves from {current} to {target}."

    reward = default_reward(next_state, goal)
    return TransitionResult(
        state=state,
        action=action,
        next_state=next_state,
        reward=reward,
        moved=not hit_obstacle and next_state.position != current,
        turned=False,
        hit_obstacle=hit_obstacle,
        reached_goal=next_state.position == goal,
        reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
        detected_node=detected_node,
        detected_edge=detected_edge,
        message=message,
    )


def describe_transition(result):
    return (
        f"Transition: {result.state} -> {result.action} -> "
        f"{result.next_state}; reward={result.reward}; {result.message}"
    )


def _coord_from_json(value, field_name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a coordinate like [x, y].")
    x, y = value
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"{field_name} coordinates must be integers.")
    return (x, y)


def _coord_to_json(coord):
    return [int(coord[0]), int(coord[1])]


def _validate_coord_in_bounds(coord, width, height, field_name):
    if coord[0] < 0 or coord[1] < 0 or coord[0] >= width or coord[1] >= height:
        raise ValueError(f"{field_name} {coord} is outside the {width}x{height} map.")


def _blocked_nodes(map_obj):
    return sorted(coord for coord, node in map_obj.nodes.items() if bool(node.isBlock))


def _blocked_edges(map_obj):
    blocked = set()
    for edge in map_obj.edges:
        if edge.isBlock:
            blocked.add(canonical_edge((edge.src.x, edge.src.y), (edge.dest.x, edge.dest.y)))
    return sorted(blocked)


def map_to_dict(map_obj, start, goal, checkpoints=None, name="custom_map", metadata=None):
    width, height = map_dimensions(map_obj)
    return {
        "version": 1,
        "name": name,
        "width": width,
        "height": height,
        "start": _coord_to_json(start),
        "goal": _coord_to_json(goal),
        "checkpoints": [_coord_to_json(coord) for coord in checkpoints or []],
        "blocked_nodes": [_coord_to_json(coord) for coord in _blocked_nodes(map_obj)],
        "blocked_edges": [
            [_coord_to_json(edge[0]), _coord_to_json(edge[1])]
            for edge in _blocked_edges(map_obj)
        ],
        "metadata": dict(metadata or {}),
    }


def save_map_json(file_path, map_obj, start, goal, checkpoints=None, name=None, metadata=None):
    map_name = name or os.path.splitext(os.path.basename(file_path))[0] or "custom_map"
    data = map_to_dict(map_obj, start, goal, checkpoints, map_name, metadata)
    folder = os.path.dirname(os.path.abspath(file_path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return data


def load_map_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("Map JSON must contain an object.")
    if data.get("version") != 1:
        raise ValueError("Only map JSON version 1 is supported.")

    width = data.get("width")
    height = data.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2:
        raise ValueError("width and height must be integers greater than 1.")

    start = _coord_from_json(data.get("start"), "start")
    goal = _coord_from_json(data.get("goal"), "goal")
    checkpoints = [
        _coord_from_json(coord, f"checkpoints[{index}]")
        for index, coord in enumerate(data.get("checkpoints", []))
    ]

    protected_nodes = {start, goal, *checkpoints}
    for field_name, coord in [("start", start), ("goal", goal)]:
        _validate_coord_in_bounds(coord, width, height, field_name)
    for index, coord in enumerate(checkpoints):
        _validate_coord_in_bounds(coord, width, height, f"checkpoints[{index}]")

    line_map = LineMap()
    line_map.createMap((width, height))

    for index, raw_node in enumerate(data.get("blocked_nodes", [])):
        coord = _coord_from_json(raw_node, f"blocked_nodes[{index}]")
        _validate_coord_in_bounds(coord, width, height, f"blocked_nodes[{index}]")
        if coord not in protected_nodes:
            line_map.set_obstacles(node=coord)

    for index, raw_edge in enumerate(data.get("blocked_edges", [])):
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            raise ValueError(f"blocked_edges[{index}] must be [[x1, y1], [x2, y2]].")
        coord_a = _coord_from_json(raw_edge[0], f"blocked_edges[{index}][0]")
        coord_b = _coord_from_json(raw_edge[1], f"blocked_edges[{index}][1]")
        _validate_coord_in_bounds(coord_a, width, height, f"blocked_edges[{index}][0]")
        _validate_coord_in_bounds(coord_b, width, height, f"blocked_edges[{index}][1]")
        if manhattan(coord_a, coord_b) != 1:
            raise ValueError(f"blocked_edges[{index}] endpoints must be adjacent.")
        line_map.set_obstacles(edge=(coord_a, coord_b))

    return MapData(
        line_map=line_map,
        name=str(data.get("name") or "loaded_map"),
        width=width,
        height=height,
        start=start,
        goal=goal,
        checkpoints=checkpoints,
        metadata=dict(data.get("metadata") or {}),
    )


def load_map_json(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return load_map_from_dict(data)
