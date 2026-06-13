import json
import os
from collections import deque
from dataclasses import dataclass, field

from src.map import LineMap


CHECKPOINT_SCR = 200
GOAL_SCR = 400
UNFINISH_SCR = -1000
MOVE_SCR = -4
TURN_SCR = -1
OBSTACLE_SCR = -25
PROGRESS_SCR = 5
RECOVERY_MOVE_SCR = 8
BLOCKED_TURN_SCR = 2
REPEATED_BLOCK_SCR = -8
TURN_REVERSAL_SCR = -2

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

FRONT_BLOCKED = 1
RIGHT_BLOCKED = 2
BACK_BLOCKED = 4
LEFT_BLOCKED = 8
MICRO_PATTERN_CONTEXT = "__micro_pattern__"

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
    context: str = ""
    obstacle_mask: int = 0
    target_forward: int = 0
    target_right: int = 0
    target_is_checkpoint: bool = False
    last_action: str = ""
    recovery_mode: bool = False

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


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def relative_directions(direction):
    return (
        direction,
        turn_right(direction),
        (-direction[0], -direction[1]),
        turn_left(direction),
    )


def is_blocked_move(map_obj, position, direction):
    if map_obj is None:
        return False
    target = (
        position[0] + direction[0],
        position[1] + direction[1],
    )
    if position not in map_obj.nodes or target not in map_obj.nodes:
        return True
    edge = get_edge(map_obj, position, target)
    return (
        edge is None
        or bool(edge.isBlock)
        or bool(map_obj.nodes[target].isBlock)
    )


def local_obstacle_mask(map_obj, position, direction):
    mask = 0
    for bit, relative_direction in zip(
        (FRONT_BLOCKED, RIGHT_BLOCKED, BACK_BLOCKED, LEFT_BLOCKED),
        relative_directions(direction),
    ):
        if is_blocked_move(map_obj, position, relative_direction):
            mask |= bit
    return mask


def current_target(position, goal, checkpoints=None, checkpoints_mask=0):
    remaining = [
        tuple(checkpoint)
        for index, checkpoint in enumerate(checkpoints or [])
        if not checkpoints_mask & (1 << index)
    ]
    if remaining:
        return (
            min(remaining, key=lambda target: manhattan(position, target)),
            True,
        )
    return tuple(goal) if goal is not None else tuple(position), False


def relative_target_features(
    position,
    direction,
    goal,
    checkpoints=None,
    checkpoints_mask=0,
):
    target, is_checkpoint = current_target(
        position,
        goal,
        checkpoints,
        checkpoints_mask,
    )
    world_dx = target[0] - position[0]
    world_dy = target[1] - position[1]
    forward = world_dx * direction[0] + world_dy * direction[1]
    right = world_dx * direction[1] - world_dy * direction[0]
    return sign(forward), sign(right), is_checkpoint


def get_state(
    position,
    direction=UP,
    checkpoints=None,
    reached_checkpoints=None,
    context="",
    map_obj=None,
    goal=None,
    last_action="",
    recovery_mode=False,
):
    mask = make_checkpoint_mask(reached_checkpoints, checkpoints)
    mask = update_checkpoint_mask(position, checkpoints, mask)
    target_forward, target_right, target_is_checkpoint = (
        relative_target_features(
            position,
            direction,
            goal,
            checkpoints,
            mask,
        )
    )
    return RLState(
        position[0],
        position[1],
        direction[0],
        direction[1],
        mask,
        str(context or ""),
        local_obstacle_mask(map_obj, position, direction),
        target_forward,
        target_right,
        target_is_checkpoint,
        str(last_action or ""),
        bool(recovery_mode),
    )


def ensure_state(state):
    if isinstance(state, RLState):
        return state
    if isinstance(state, tuple) and len(state) == 5:
        return RLState(state[0], state[1], state[2], state[3], state[4])
    if isinstance(state, tuple) and len(state) == 6:
        return RLState(state[0], state[1], state[2], state[3], state[4], state[5])
    if isinstance(state, tuple) and len(state) == 12:
        return RLState(*state)
    raise TypeError(
        "state must be an RLState or a tuple: "
        "(x, y, dx, dy, checkpoints_mask[, context, micro features...])"
    )


def micro_pattern_state(state):
    return RLState(
        -1,
        -1,
        0,
        0,
        checkpoint_count(state.checkpoints_mask),
        MICRO_PATTERN_CONTEXT,
        state.obstacle_mask,
        state.target_forward,
        state.target_right,
        state.target_is_checkpoint,
        state.last_action,
        state.recovery_mode,
    )


def exact_policy_state(state):
    return RLState(
        state.x,
        state.y,
        state.dx,
        state.dy,
        state.checkpoints_mask,
        state.context,
        state.obstacle_mask,
        state.target_forward,
        state.target_right,
        state.target_is_checkpoint,
        "",
        state.recovery_mode,
    )


def legacy_state_candidates(state):
    candidates = [
        state,
        exact_policy_state(state),
        RLState(
            state.x,
            state.y,
            state.dx,
            state.dy,
            state.checkpoints_mask,
            state.context,
        ),
        RLState(
            state.x,
            state.y,
            state.dx,
            state.dy,
            state.checkpoints_mask,
            "",
            state.obstacle_mask,
            state.target_forward,
            state.target_right,
            state.target_is_checkpoint,
            state.last_action,
            state.recovery_mode,
        ),
        RLState(
            state.x,
            state.y,
            state.dx,
            state.dy,
            state.checkpoints_mask,
        ),
        micro_pattern_state(state),
    ]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def ensure_action_values(qtable, state):
    if state not in qtable:
        qtable[state] = {action: 0.0 for action in ACTIONS}
    return qtable[state]


def find_action_values(qtable, state):
    first_values = None
    for candidate in legacy_state_candidates(state):
        values = qtable.get(candidate)
        if values is None:
            continue
        if first_values is None:
            first_values = values
        if any(abs(values.get(action, 0.0)) > 1e-9 for action in ACTIONS):
            return values
    return first_values


def has_learned_action(qtable, state):
    values = find_action_values(qtable, state)
    return values is not None and any(
        abs(values.get(action, 0.0)) > 1e-9
        for action in ACTIONS
    )


def max_learned_q(qtable, state):
    values = find_action_values(qtable, state)
    if values is None:
        return 0.0
    return max(values.get(action, 0.0) for action in ACTIONS)


def update_q_values(
    qtable,
    state,
    action,
    target,
    alpha,
    pattern_alpha_scale=0.35,
):
    action_values = ensure_action_values(qtable, exact_policy_state(state))
    old_q = action_values[action]
    action_values[action] = old_q + alpha * (target - old_q)

    pattern = micro_pattern_state(state)
    if pattern == state:
        return
    pattern_values = ensure_action_values(qtable, pattern)
    pattern_q = pattern_values[action]
    pattern_alpha = alpha * pattern_alpha_scale
    pattern_values[action] = pattern_q + pattern_alpha * (target - pattern_q)


def shortest_known_path(map_obj, start, goal):
    if start not in map_obj.nodes or goal not in map_obj.nodes:
        return None
    if map_obj.nodes[start].isBlock or map_obj.nodes[goal].isBlock:
        return None
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        for direction in DIRECTIONS:
            neighbor = (
                current[0] + direction[0],
                current[1] + direction[1],
            )
            if neighbor in came_from:
                continue
            if is_blocked_move(map_obj, current, direction):
                continue
            came_from[neighbor] = current
            queue.append(neighbor)
    return None


def planned_action(map_obj, state, goal, checkpoints=None):
    targets = remaining_checkpoints(state, checkpoints) or [tuple(goal)]
    routes = [
        route
        for target in targets
        if (route := shortest_known_path(map_obj, state.position, target))
        is not None
    ]
    if not routes:
        return None
    route = min(routes, key=len)
    if len(route) < 2:
        return None
    next_coord = route[1]
    move_direction = (
        next_coord[0] - state.x,
        next_coord[1] - state.y,
    )
    if move_direction == state.direction:
        return FORWARD
    if move_direction == (-state.dx, -state.dy):
        return BACKWARD
    if move_direction == turn_left(state.direction):
        return TURN_LEFT
    if move_direction == turn_right(state.direction):
        return TURN_RIGHT
    return None


def action_visit_key(state, action):
    return (
        state.x,
        state.y,
        state.dx,
        state.dy,
        state.checkpoints_mask,
        action,
    )


def greedy_action(qtable, state, map_obj=None, goal=None, checkpoints=None):
    action_values = find_action_values(qtable, state)
    if action_values is None:
        action_values = {action: 0.0 for action in ACTIONS}
    actions = valid_actions(state, map_obj)
    best_value = max(action_values.get(action, 0.0) for action in actions)
    best_actions = [
        action
        for action in actions
        if action_values.get(action, 0.0) == best_value
    ]
    if map_obj is None or goal is None:
        return best_actions[0]
    return max(
        best_actions,
        key=lambda action: action_tie_breaker(
            map_obj,
            state,
            action,
            goal,
            checkpoints,
        ),
    )


def exploration_action(
    map_obj,
    state,
    goal,
    checkpoints=None,
    action_counts=None,
    position_counts=None,
):
    action_counts = action_counts or {}
    position_counts = position_counts or {}

    def score(action):
        result = transition(map_obj, state, action, goal, checkpoints)
        gained_checkpoints = (
            checkpoint_count(result.next_state.checkpoints_mask)
            - checkpoint_count(state.checkpoints_mask)
        )
        distance_gain = (
            target_distance(state, goal, checkpoints)
            - target_distance(result.next_state, goal, checkpoints)
        )
        action_visits = action_counts.get(action_visit_key(state, action), 0)
        position_visits = position_counts.get(result.next_state.position, 0)
        value = 0
        if is_terminal_state(result.next_state, goal, checkpoints):
            value += 100000
        value += gained_checkpoints * 10000
        value -= action_visits * 250
        value -= position_visits * 35
        value += distance_gain * 20
        if result.moved:
            value += 45
        if result.hit_obstacle:
            value -= 10000
        if state.recovery_mode and result.moved:
            value += 35
        if state.obstacle_mask & FRONT_BLOCKED and result.turned:
            value += 25
        if (
            action in (TURN_LEFT, TURN_RIGHT)
            and result.next_state.obstacle_mask & FRONT_BLOCKED
        ):
            value -= 30
        if action == BACKWARD and state.obstacle_mask & FRONT_BLOCKED:
            value += 12
        if action == TURN_RIGHT and state.target_right > 0:
            value += 10
        if action == TURN_LEFT and state.target_right < 0:
            value += 10
        if (
            state.last_action == TURN_LEFT
            and action == TURN_RIGHT
        ) or (
            state.last_action == TURN_RIGHT
            and action == TURN_LEFT
        ):
            value -= 15
        return value

    return max(ACTIONS, key=score)


def runtime_action(
    qtable,
    map_obj,
    state,
    goal,
    checkpoints=None,
    action_counts=None,
    position_counts=None,
):
    source = "policy"
    if has_learned_action(qtable, state):
        action = greedy_action(
            qtable,
            state,
            map_obj,
            goal,
            checkpoints,
        )
    else:
        action = planned_action(map_obj, state, goal, checkpoints)
        source = "planner"

    should_recover = action is None
    if action is not None:
        result = transition(map_obj, state, action, goal, checkpoints)
        should_recover = (
            result.hit_obstacle
            or (action_counts or {}).get(action_visit_key(state, action), 0) > 0
        )

    if should_recover:
        action = exploration_action(
            map_obj,
            state,
            goal,
            checkpoints,
            action_counts,
            position_counts,
        )
        source = "recovery"
    return action, source


def valid_actions(state=None, map_obj=None):
    return list(ACTIONS)


def map_dimensions(map_obj):
    if not map_obj.nodes:
        return (0, 0)
    width = max(x for x, _ in map_obj.nodes) + 1
    height = max(y for _, y in map_obj.nodes) + 1
    return (width, height)


def get_edge(map_obj, src_coord, dest_coord):
    cache_key = (id(map_obj.edges), len(map_obj.edges))
    if getattr(map_obj, "_edge_lookup_cache_key", None) != cache_key:
        map_obj._edge_lookup_cache = {
            ((edge.src.x, edge.src.y), (edge.dest.x, edge.dest.y)): edge
            for edge in map_obj.edges
        }
        map_obj._edge_lookup_cache_key = cache_key
    return map_obj._edge_lookup_cache.get((src_coord, dest_coord))


def canonical_edge(coord_a, coord_b):
    first, second = sorted([tuple(coord_a), tuple(coord_b)])
    return (first, second)


def is_blocked_edge(map_obj, coord_a, coord_b):
    edge = get_edge(map_obj, coord_a, coord_b)
    return edge is None or bool(edge.isBlock)


def default_reward(next_state, goal):
    return -manhattan(next_state.position, goal)


def all_checkpoints_reached(state, checkpoints=None):
    return checkpoint_count(state.checkpoints_mask) >= len(checkpoints or [])


def is_terminal_state(state, goal, checkpoints=None):
    return state.position == tuple(goal) and all_checkpoints_reached(state, checkpoints)


def remaining_checkpoints(state, checkpoints=None):
    return [
        tuple(checkpoint)
        for index, checkpoint in enumerate(checkpoints or [])
        if not state.checkpoints_mask & (1 << index)
    ]


def target_distance(state, goal, checkpoints=None):
    targets = remaining_checkpoints(state, checkpoints) or [tuple(goal)]
    return min(manhattan(state.position, target) for target in targets)


def transition_reward(result, goal, checkpoints=None):
    reward = 0.0
    if result.moved:
        reward += MOVE_SCR
    if result.turned:
        reward += TURN_SCR
    if result.hit_obstacle:
        reward += OBSTACLE_SCR
    if result.state.recovery_mode and result.moved:
        reward += RECOVERY_MOVE_SCR
    if result.state.obstacle_mask & FRONT_BLOCKED and result.turned:
        reward += BLOCKED_TURN_SCR
    if result.state.recovery_mode and result.hit_obstacle:
        reward += REPEATED_BLOCK_SCR
    if (
        result.state.last_action == TURN_LEFT
        and result.action == TURN_RIGHT
    ) or (
        result.state.last_action == TURN_RIGHT
        and result.action == TURN_LEFT
    ):
        reward += TURN_REVERSAL_SCR

    reward += PROGRESS_SCR * (
        target_distance(result.state, goal, checkpoints)
        - target_distance(result.next_state, goal, checkpoints)
    )

    newly_reached = (
        checkpoint_count(result.next_state.checkpoints_mask)
        - checkpoint_count(result.state.checkpoints_mask)
    )
    reward += max(0, newly_reached) * CHECKPOINT_SCR

    if is_terminal_state(result.next_state, goal, checkpoints):
        reward += GOAL_SCR
    return reward


def action_tie_breaker(map_obj, state, action, goal, checkpoints=None):
    result = transition(map_obj, state, action, goal, checkpoints)
    gained_checkpoints = (
        checkpoint_count(result.next_state.checkpoints_mask)
        - checkpoint_count(state.checkpoints_mask)
    )
    distance_gain = (
        target_distance(state, goal, checkpoints)
        - target_distance(result.next_state, goal, checkpoints)
    )
    return (
        is_terminal_state(result.next_state, goal, checkpoints),
        gained_checkpoints,
        distance_gain,
        result.moved,
        not result.hit_obstacle,
        not result.turned,
    )


def transition(map_obj, state, action, goal, checkpoints=None):
    state = ensure_state(state)
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    current = state.position
    direction = state.direction
    checkpoints = list(checkpoints or [])

    if action == TURN_LEFT:
        next_direction = turn_left(direction)
        next_state = get_state(
            current,
            next_direction,
            checkpoints,
            reached_checkpoints=state.checkpoints_mask,
            context=state.context,
            map_obj=map_obj,
            goal=goal,
            last_action=action,
            recovery_mode=state.recovery_mode,
        )
        result = TransitionResult(
            state=state,
            action=action,
            next_state=next_state,
            reward=0.0,
            turned=True,
            reached_goal=current == goal,
            reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
            message=f"Turn left. Heading is now {direction_name(next_direction)}.",
        )
        result.reward = transition_reward(result, goal, checkpoints)
        return result

    if action == TURN_RIGHT:
        next_direction = turn_right(direction)
        next_state = get_state(
            current,
            next_direction,
            checkpoints,
            reached_checkpoints=state.checkpoints_mask,
            context=state.context,
            map_obj=map_obj,
            goal=goal,
            last_action=action,
            recovery_mode=state.recovery_mode,
        )
        result = TransitionResult(
            state=state,
            action=action,
            next_state=next_state,
            reward=0.0,
            turned=True,
            reached_goal=current == goal,
            reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
            message=f"Turn right. Heading is now {direction_name(next_direction)}.",
        )
        result.reward = transition_reward(result, goal, checkpoints)
        return result

    move_direction = direction
    if action == BACKWARD:
        move_direction = (-direction[0], -direction[1])

    target = (state.x + move_direction[0], state.y + move_direction[1])
    next_position = current
    next_mask = state.checkpoints_mask
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
            next_position = target
            next_mask = update_checkpoint_mask(
                target,
                checkpoints,
                state.checkpoints_mask,
            )
            message = f"Robot moves from {current} to {target}."

    next_state = get_state(
        next_position,
        direction,
        checkpoints,
        reached_checkpoints=next_mask,
        context=state.context,
        map_obj=map_obj,
        goal=goal,
        last_action=action,
        recovery_mode=hit_obstacle,
    )
    result = TransitionResult(
        state=state,
        action=action,
        next_state=next_state,
        reward=0.0,
        moved=not hit_obstacle and next_position != current,
        turned=False,
        hit_obstacle=hit_obstacle,
        reached_goal=next_state.position == goal,
        reached_checkpoints=checkpoint_count(next_state.checkpoints_mask),
        detected_node=detected_node,
        detected_edge=detected_edge,
        message=message,
    )
    result.reward = transition_reward(result, goal, checkpoints)
    return result


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
