import random
import os
import sys

from Algo.Astar import astar
from Algo.QLearning import ACTIONS, get_path as qlearning_path, train as train_qtable

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.agent import Robot
from src.map import LineMap


ACTION_NAMES = {
    (0, 1): "up",
    (1, 0): "right",
    (0, -1): "down",
    (-1, 0): "left",
}


def build_demo_map(size=5):
    line_map = LineMap()
    line_map.createMap(size)

    line_map.set_obstacles(node=(2, 2))
    line_map.set_obstacles(edge=((1, 1), (1, 2)))
    line_map.set_obstacles(edge=((3, 3), (3, 4)))

    return line_map


def format_path(path):
    if path is None:
        return "No path found"
    return " -> ".join(str(coord) for coord in path)


def print_qtable_policy(qtable, path):
    if not path:
        return

    print("\nQ-table greedy policy on the Q-learning path:")
    for state in path[:-1]:
        best_action = max(ACTIONS, key=lambda action: qtable[state][action])
        best_value = qtable[state][best_action]
        print(f"  {state}: {ACTION_NAMES[best_action]} (Q={best_value:.2f})")


def run_path_planners():
    start = (0, 0)
    goal = (4, 4)
    demo_map = build_demo_map()

    print("=== A* on full map ===")
    astar_result = astar(demo_map, start, goal)
    print(format_path(astar_result))

    print("\n=== Q-learning on full map ===")
    random.seed(42)
    qtable = train_qtable(demo_map, start, goal, episodes=5000)
    random.seed(42)
    q_result = qlearning_path(demo_map, start, goal, episodes=5000)
    print(format_path(q_result))
    print_qtable_policy(qtable, q_result)


def run_two_map_navigation():
    start = (0, 0)
    goal = (4, 4)
    ground_truth_map = build_demo_map()
    robot = Robot(start[0], start[1], map_obj=ground_truth_map, direction=(0, 1))

    print("\n=== Two-map robot simulation using A* ===")
    print("Robot plans with known_map, moves on the ground-truth map, and replans after discovering obstacles.")
    reached_goal = robot.navigate_to(goal)
    print(f"Reached goal: {reached_goal}")
    print(f"Final position: ({robot.x}, {robot.y})")
    print(f"Discovered nodes: {sorted(robot.discovered_nodes)}")
    print(f"Discovered edges: {sorted(robot.discovered_edges)}")


if __name__ == "__main__":
    run_path_planners()
    run_two_map_navigation()
