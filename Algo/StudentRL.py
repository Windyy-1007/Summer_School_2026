import random
import json
import os
from dataclasses import dataclass, field

from rl_library import (
    ACTIONS,
    RLState,
    UP,
    action_visit_key,
    calculate_score,
    checkpoint_count,
    greedy_action,
    get_state,
    is_terminal_state,
    manhattan,
    max_learned_q,
    runtime_action,
    transition,
    update_q_values,
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
    actions: list = field(default_factory=list)
    action_sources: list = field(default_factory=list)
    final_state: object | None = None
    planner_fallback_steps: int = 0
    recovery_steps: int = 0


@dataclass
class TrainingResult:
    qtable: dict
    episodes: int
    best_result: EpisodeResult | None
    last_result: EpisodeResult | None
    logs: list = field(default_factory=list)


def create_qtable():
    return {}


def best_action(qtable, state, map_obj=None, goal=None, checkpoints=None):
    return greedy_action(qtable, state, map_obj, goal, checkpoints)


def choose_action(
    qtable,
    state,
    epsilon,
    rng,
    map_obj=None,
    goal=None,
    checkpoints=None,
):
    actions = valid_actions(state, map_obj)
    if rng.random() < epsilon:
        return rng.choice(actions)
    return best_action(qtable, state, map_obj, goal, checkpoints)


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
    context="",
    reached_checkpoints=None,
    planner_fallback=False,
    action_counts=None,
    position_counts=None,
    last_action="",
    recovery_mode=False,
):
    rng = rng or random.Random()
    checkpoints = list(checkpoints or [])
    max_steps = max_steps or max(20, len(map_obj.nodes) * 8)

    state = get_state(
        start,
        initial_direction,
        checkpoints,
        reached_checkpoints=reached_checkpoints,
        context=context,
        map_obj=map_obj,
        goal=goal,
        last_action=last_action,
        recovery_mode=recovery_mode,
    )
    path = [start]
    actions_taken = []
    action_sources = []
    total_reward = 0.0
    moves = 0
    turns = 0
    min_distance = manhattan(start, goal)
    planner_fallback_steps = 0
    recovery_steps = 0
    action_counts = dict(action_counts or {})
    position_counts = dict(position_counts or {})
    position_counts[state.position] = position_counts.get(state.position, 0) + 1

    for step_index in range(max_steps):
        if is_terminal_state(state, goal, checkpoints):
            break

        if learning:
            action = choose_action(
                qtable,
                state,
                epsilon,
                rng,
                map_obj,
                goal,
                checkpoints,
            )
            action_source = "learning"
        elif planner_fallback:
            action, action_source = runtime_action(
                qtable,
                map_obj,
                state,
                goal,
                checkpoints,
                action_counts,
                position_counts,
            )
            if action_source == "planner":
                planner_fallback_steps += 1
            elif action_source == "recovery":
                recovery_steps += 1
        else:
            action = best_action(
                qtable,
                state,
                map_obj,
                goal,
                checkpoints,
            )
            action_source = "policy"

        actions_taken.append(action)
        action_sources.append(action_source)
        visit_key = action_visit_key(state, action)
        action_counts[visit_key] = action_counts.get(visit_key, 0) + 1
        result = transition(map_obj, state, action, goal, checkpoints)
        reward = result.reward
        total_reward += reward

        if learning:
            next_best_q = (
                0.0
                if is_terminal_state(result.next_state, goal, checkpoints)
                else max_learned_q(qtable, result.next_state)
            )
            update_q_values(
                qtable,
                state,
                action,
                reward + gamma * next_best_q,
                alpha,
            )

        if result.moved:
            moves += 1
            path.append(result.next_state.position)
        if result.turned:
            turns += 1

        state = result.next_state
        position_counts[state.position] = position_counts.get(state.position, 0) + 1
        min_distance = min(min_distance, manhattan(state.position, goal))

        if is_terminal_state(state, goal, checkpoints):
            break

    reached_goal = is_terminal_state(state, goal, checkpoints)
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
        actions=actions_taken,
        action_sources=action_sources,
        final_state=state,
        planner_fallback_steps=planner_fallback_steps,
        recovery_steps=recovery_steps,
    )


def simulate_policy(
    qtable,
    map_obj,
    start,
    goal,
    checkpoints=None,
    max_steps=None,
    initial_direction=UP,
    context="",
    reached_checkpoints=None,
    planner_fallback=False,
    action_counts=None,
    position_counts=None,
    last_action="",
    recovery_mode=False,
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
        context=context,
        reached_checkpoints=reached_checkpoints,
        planner_fallback=planner_fallback,
        action_counts=action_counts,
        position_counts=position_counts,
        last_action=last_action,
        recovery_mode=recovery_mode,
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
    qtable=None,
    context="",
):
    rng = random.Random(seed)
    qtable = qtable if qtable is not None else create_qtable()
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
            context=context,
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


def get_policy_path(
    qtable,
    map_obj,
    start,
    goal,
    checkpoints=None,
    max_steps=None,
    context="",
):
    result = simulate_policy(
        qtable,
        map_obj,
        start,
        goal,
        checkpoints,
        max_steps,
        context=context,
    )
    return result.path


def save_policy_json(
    file_path,
    qtable,
    episodes,
    map_name="student_rl_policy",
    map_file=None,
    start=None,
    goal=None,
    checkpoints=None,
    best_result=None,
    last_result=None,
    policy_result=None,
    metadata=None,
):
    results = {}
    for label, result in (
        ("best", best_result),
        ("last", last_result),
        ("policy", policy_result),
    ):
        if result is None:
            continue
        results[label] = {
            "episode": result.episode,
            "total_reward": result.total_reward,
            "score": result.score,
            "reached_goal": result.reached_goal,
            "checkpoints_reached": result.checkpoints_reached,
            "min_manhattan": result.min_manhattan,
            "moves": result.moves,
            "turns": result.turns,
            "steps": result.steps,
            "epsilon": result.epsilon,
            "path": [list(coord) for coord in result.path],
            "actions": list(result.actions),
            "action_sources": list(result.action_sources),
            "planner_fallback_steps": result.planner_fallback_steps,
            "recovery_steps": result.recovery_steps,
        }

    rows = []
    for state in sorted(
        qtable,
        key=lambda item: (
            item.context,
            item.x,
            item.y,
            item.dx,
            item.dy,
            item.checkpoints_mask,
            item.obstacle_mask,
            item.target_forward,
            item.target_right,
            item.target_is_checkpoint,
            item.last_action,
            item.recovery_mode,
        ),
    ):
        rows.append(
            {
                "state": {
                    "x": state.x,
                    "y": state.y,
                    "dx": state.dx,
                    "dy": state.dy,
                    "checkpoints_mask": state.checkpoints_mask,
                    "context": state.context,
                    "obstacle_mask": state.obstacle_mask,
                    "target_forward": state.target_forward,
                    "target_right": state.target_right,
                    "target_is_checkpoint": state.target_is_checkpoint,
                    "last_action": state.last_action,
                    "recovery_mode": state.recovery_mode,
                },
                "actions": {
                    action: float(qtable[state].get(action, 0.0))
                    for action in ACTIONS
                },
            }
        )

    data = {
        "version": 1,
        "algorithm": "student_q_learning",
        "map_name": str(map_name),
        "map_file": map_file,
        "start": list(start) if start is not None else None,
        "goal": list(goal) if goal is not None else None,
        "checkpoints": [list(coord) for coord in checkpoints or []],
        "episodes": int(episodes),
        "qtable": rows,
        "results": results,
        "metadata": dict(metadata or {}),
    }
    folder = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(folder, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return data


def load_policy_json(file_path):
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
            str(state_data.get("context", "")),
            int(state_data.get("obstacle_mask", 0)),
            int(state_data.get("target_forward", 0)),
            int(state_data.get("target_right", 0)),
            bool(state_data.get("target_is_checkpoint", False)),
            str(state_data.get("last_action", "")),
            bool(state_data.get("recovery_mode", False)),
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


