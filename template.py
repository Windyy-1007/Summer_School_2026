import random
from dataclasses import dataclass, field

from rl_library import (
    ACTIONS,
    UP,
    calculate_score,
    checkpoint_count,
    default_reward,
    get_state,
    manhattan,
    transition,
    valid_actions,
)


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    score: int
    reached_goal: bool
    checkpoints_reached: int
    min_manhattan: int
    moves: int
    turns: int
    steps: int
    epsilon: float
    path: list = field(default_factory=list)
    final_state: object | None = None


@dataclass
class TrainingResult:
    qtable: dict
    episodes: int
    best_result: EpisodeResult | None
    last_result: EpisodeResult | None
    logs: list = field(default_factory=list)


def create_qtable():
    return {}


def _ensure_action_values(qtable, state):
    if state not in qtable:
        qtable[state] = {action: 0.0 for action in ACTIONS}
    return qtable[state]


def best_action(qtable, state):
    action_values = _ensure_action_values(qtable, state)
    return max(ACTIONS, key=lambda action: action_values[action])


def choose_action(qtable, state, epsilon, rng, map_obj=None):
    actions = valid_actions(state, map_obj)
    if rng.random() < epsilon:
        return rng.choice(actions)
    action_values = _ensure_action_values(qtable, state)
    return max(actions, key=lambda action: action_values[action])


def _is_better_result(candidate, current_best):
    if current_best is None:
        return True
    if candidate.reached_goal != current_best.reached_goal:
        return candidate.reached_goal
    if candidate.score != current_best.score:
        return candidate.score > current_best.score
    if candidate.checkpoints_reached != current_best.checkpoints_reached:
        return candidate.checkpoints_reached > current_best.checkpoints_reached
    if candidate.min_manhattan != current_best.min_manhattan:
        return candidate.min_manhattan < current_best.min_manhattan
    return candidate.total_reward > current_best.total_reward


def is_better_result(candidate, current_best):
    return _is_better_result(candidate, current_best)


def run_episode(
    qtable,
    map_obj,
    start,
    goal,
    checkpoints=None,
    episode=1,
    epsilon=0.1,
    alpha=0.2,
    gamma=0.9,
    max_steps=None,
    rng=None,
    learning=True,
    initial_direction=UP,
):
    rng = rng or random.Random()
    checkpoints = list(checkpoints or [])
    max_steps = max_steps or max(20, len(map_obj.nodes) * 8)

    state = get_state(start, initial_direction, checkpoints)
    path = [start]
    total_reward = 0.0
    moves = 0
    turns = 0
    min_distance = manhattan(start, goal)
    visited_greedy_states = set()

    for step_index in range(max_steps):
        if state.position == goal:
            break

        if learning:
            action = choose_action(qtable, state, epsilon, rng, map_obj)
        else:
            state_key = (state, best_action(qtable, state))
            if state_key in visited_greedy_states:
                break
            visited_greedy_states.add(state_key)
            action = best_action(qtable, state)

        result = transition(map_obj, state, action, goal, checkpoints)
        reward = default_reward(result.next_state, goal)
        total_reward += reward

        if learning:
            action_values = _ensure_action_values(qtable, state)
            next_action_values = _ensure_action_values(qtable, result.next_state)
            next_best_q = 0.0 if result.reached_goal else max(next_action_values.values())
            old_q = action_values[action]
            action_values[action] = old_q + alpha * (reward + gamma * next_best_q - old_q)

        if result.moved:
            moves += 1
            path.append(result.next_state.position)
        if result.turned:
            turns += 1

        state = result.next_state
        min_distance = min(min_distance, manhattan(state.position, goal))

        if result.reached_goal:
            break

    reached_goal = state.position == goal
    checkpoints_reached = checkpoint_count(state.checkpoints_mask)
    score = calculate_score(moves, turns, checkpoints_reached, reached_goal, min_distance)

    return EpisodeResult(
        episode=episode,
        total_reward=total_reward,
        score=score,
        reached_goal=reached_goal,
        checkpoints_reached=checkpoints_reached,
        min_manhattan=min_distance,
        moves=moves,
        turns=turns,
        steps=step_index + 1 if "step_index" in locals() else 0,
        epsilon=epsilon,
        path=path,
        final_state=state,
    )


def simulate_policy(
    qtable,
    map_obj,
    start,
    goal,
    checkpoints=None,
    max_steps=None,
    initial_direction=UP,
):
    return run_episode(
        qtable=qtable,
        map_obj=map_obj,
        start=start,
        goal=goal,
        checkpoints=checkpoints,
        episode=0,
        epsilon=0.0,
        max_steps=max_steps,
        rng=random.Random(0),
        learning=False,
        initial_direction=initial_direction,
    )


def train(
    map_obj,
    start,
    goal,
    checkpoints=None,
    episodes=500,
    alpha=0.2,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    max_steps=None,
    log_every=50,
    log_callback=None,
    seed=42,
):
    rng = random.Random(seed)
    qtable = create_qtable()
    epsilon = epsilon_start
    logs = []
    best_result = None
    last_result = None

    for episode in range(1, episodes + 1):
        last_result = run_episode(
            qtable=qtable,
            map_obj=map_obj,
            start=start,
            goal=goal,
            checkpoints=checkpoints,
            episode=episode,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            max_steps=max_steps,
            rng=rng,
            learning=True,
        )

        if _is_better_result(last_result, best_result):
            best_result = last_result

        if episode == 1 or episode == episodes or episode % log_every == 0:
            log_line = format_episode_log(last_result, total_checkpoints=len(checkpoints or []))
            logs.append(log_line)
            if log_callback is not None:
                log_callback(last_result, qtable, log_line)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return TrainingResult(
        qtable=qtable,
        episodes=episodes,
        best_result=best_result,
        last_result=last_result,
        logs=logs,
    )


def get_policy_path(qtable, map_obj, start, goal, checkpoints=None, max_steps=None):
    result = simulate_policy(qtable, map_obj, start, goal, checkpoints, max_steps)
    return result.path


def load_policy_json(file_path):
    import json

    from rl_library import RLState

    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if data.get("version") != 1:
        raise ValueError("Only StudentRL policy JSON version 1 is supported.")

    qtable = create_qtable()
    for row in data.get("qtable", []):
        state_data = row["state"]
        state = RLState(
            int(state_data["x"]),
            int(state_data["y"]),
            int(state_data["dx"]),
            int(state_data["dy"]),
            int(state_data.get("checkpoints_mask", 0)),
        )
        action_values = row.get("actions", {})
        qtable[state] = {
            action: float(action_values.get(action, 0.0))
            for action in ACTIONS
        }

    return qtable, data


def format_episode_log(result, total_checkpoints=0):
    goal_text = "yes" if result.reached_goal else "no"
    return (
        f"Episode {result.episode}: reward={result.total_reward:.1f}, "
        f"score={result.score}, reached {result.checkpoints_reached}/{total_checkpoints} checkpoints, "
        f"goal={goal_text}, min distance {result.min_manhattan}, "
        f"moves={result.moves}, turns={result.turns}, epsilon={result.epsilon:.3f}"
    )


