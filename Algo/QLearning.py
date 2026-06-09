import random
from collections import defaultdict

# Possible movement actions as (dx, dy)
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left

# Reward values
REWARD_GOAL = 400
REWARD_SUB_GOAL = 150
REWARD_STEP = -3
REWARD_OBSTACLE = -10


def _simulate_step(map_obj, coord, action):
    """
    Simulates one step on the map without modifying it.

    Returns:
        (next_coord, hit_obstacle):
            next_coord    -- resulting position (unchanged if move is invalid)
            hit_obstacle  -- True if the move was blocked or out of bounds
    """
    to_coord = (coord[0] + action[0], coord[1] + action[1])

    if to_coord not in map_obj.nodes:
        return coord, True

    if map_obj.nodes[to_coord].isBlock:
        return coord, True

    for edge in map_obj.edges:
        if (edge.src.x, edge.src.y) == coord and (edge.dest.x, edge.dest.y) == to_coord:
            if edge.isBlock:
                return coord, True
            return to_coord, False

    return coord, True  # No connecting edge found


def train(map_obj, start, goal,
          episodes=2000, alpha=0.1, gamma=0.9,
          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    """
    Trains a Q-table via Q-learning on map_obj.

    Args:
        map_obj (LineMap):      The map to train on.
        start (tuple):          Starting coordinate (x, y).
        goal (tuple):           Goal coordinate (x, y).
        episodes (int):         Number of training episodes.
        alpha (float):          Learning rate.
        gamma (float):          Discount factor.
        epsilon_start (float):  Initial exploration rate.
        epsilon_end (float):    Minimum exploration rate.
        epsilon_decay (float):  Multiplicative epsilon decay per episode.

    Returns:
        dict: Trained Q-table mapping state -> {action -> q_value}.
    """
    Q = defaultdict(lambda: defaultdict(float))
    epsilon = epsilon_start
    max_steps = len(map_obj.nodes) * 2  # cap steps per episode to avoid infinite loops

    for _ in range(episodes):
        state = start

        for _ in range(max_steps):
            if state == goal:
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(ACTIONS, key=lambda a: Q[state][a])

            next_state, hit_obstacle = _simulate_step(map_obj, state, action)

            if hit_obstacle:
                reward = REWARD_OBSTACLE
            elif next_state == goal:
                reward = REWARD_GOAL
            else:
                reward = REWARD_STEP

            # Q-learning update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
            best_next_q = max(Q[next_state][a] for a in ACTIONS)
            Q[state][action] += alpha * (reward + gamma * best_next_q - Q[state][action])

            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q


def get_path(map_obj, start, goal,
             sub_goals=None,
             episodes=2000, alpha=0.1, gamma=0.9,
             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    """
    Finds a path from start to goal using Q-learning on a LineMap.
    Trains a Q-table, then follows the greedy policy to extract the path.

    Args:
        map_obj (LineMap):      The map to navigate.
        start (tuple):          Starting coordinate (x, y).
        goal (tuple):           Goal coordinate (x, y).
        episodes (int):         Training episodes.
        alpha (float):          Learning rate.
        gamma (float):          Discount factor.
        epsilon_start (float):  Initial exploration rate.
        epsilon_end (float):    Minimum exploration rate.
        epsilon_decay (float):  Multiplicative epsilon decay per episode.

    Returns:
        list: Ordered list of (x, y) coordinates from start to goal (inclusive),
              or None if the goal could not be reached.
    """
    if sub_goals:
        route = [start]
        current = start
        for target in list(sub_goals) + [goal]:
            segment = get_path(
                map_obj,
                current,
                target,
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay
            )
            if segment is None:
                return None
            route.extend(segment[1:])
            current = target
        return route

    if start not in map_obj.nodes or goal not in map_obj.nodes:
        return None
    if map_obj.nodes[goal].isBlock:
        return None
    if start == goal:
        return [start]

    Q = train(map_obj, start, goal, episodes, alpha, gamma,
              epsilon_start, epsilon_end, epsilon_decay)

    # Greedy path extraction from trained Q-table
    path = [start]
    state = start
    visited = {start}
    max_steps = len(map_obj.nodes)

    for _ in range(max_steps):
        if state == goal:
            return path

        action = max(ACTIONS, key=lambda a: Q[state][a])
        next_state, hit_obstacle = _simulate_step(map_obj, state, action)

        if hit_obstacle or next_state in visited:
            return None  # Stuck or loop detected

        path.append(next_state)
        visited.add(next_state)
        state = next_state

    return None  # Goal not reached within max steps
