import sys
import os
import random

# Add src folder to path to allow importing map and agent
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from map import LineMap
from agent import Robot
from Algo.Astar import astar_route
from Algo.QLearning import get_path as qlearning_path
from Algo import StudentRL
from rl_library import (
    CHECKPOINT_SCR,
    GOAL_SCR,
    MOVE_SCR,
    TURN_SCR,
    calculate_score,
    load_map_json,
    manhattan,
    save_map_json,
)

FINAL_GOAL_REWARD = GOAL_SCR
SUB_GOAL_REWARD = CHECKPOINT_SCR
MOVE_COST = MOVE_SCR
TURN_COST = TURN_SCR
STUDENT_RL_POLICY_FILENAME = "student_rl_policy.json"

class SimulationService:
    def __init__(self, map_size=5):
        self.map_width = map_size
        self.map_height = map_size
        self.map_size = map_size
        self.map_name = "Default Demo Map"
        self.custom_map_dir = os.path.join(root_dir, "maps", "custom")
        self.training_map_dir = os.path.join(root_dir, "maps", "training")
        self.student_rl_policy_dir = os.path.join(root_dir, "maps", "student_rl_policies")
        self.current_map_file_path = None
        self.start = (0, 0)
        self.goal = (min(4, map_size - 1), min(4, map_size - 1))
        self.sub_goals = [(1, 4), (4, 1)]
        self.map = LineMap()
        self.map.createMap(self.map_size)
        self.student_rl_training = None
        self.student_rl_qtable = None
        self.student_rl_best_result = None
        self.student_rl_run_qtable = None
        self.student_rl_run_signature = None
        self.student_rl_run_policy_data = None
        
        # Gọi phương thức thiết lập vật cản
        self.configure_obstacles()

        # Starts at (0,0) facing UP (0, 1)
        self.reset_robot()

    def configure_obstacles(self):
        """
        Dùng phương thức này để tự thiết lập các vật cản (node hoặc edge) trên bản đồ.
        """
        # =====================================================================
        # KHU VỰC THIẾT LẬP VẬT CẢN (BẠN CÓ THỂ TỰ SỬA/THÊM/BỚT Ở ĐÂY)
        # =====================================================================

        # 1. Chặn các Nút (Block Nodes)
        # Điền tọa độ dạng (x, y) để chặn nút đó
        self.map.set_obstacles(node=(2, 2))
        self.map.set_obstacles(node=(1, 3))

        # 2. Chặn các Cạnh (Block Edges)
        # Điền cặp tọa độ dạng ((x1, y1), (x2, y2)) để chặn cạnh nối giữa hai nút
        self.map.set_obstacles(edge=((0, 1), (1, 1)))
        self.map.set_obstacles(edge=((3, 3), (3, 4)))
        self.map.set_obstacles(edge=((2, 0), (3, 0)))

        # =====================================================================

    def get_map_size(self):
        return self.map_size

    def get_map_width(self):
        return self.map_width

    def get_map_height(self):
        return self.map_height

    def get_map_name(self):
        return self.map_name

    def get_custom_map_dir(self):
        os.makedirs(self.custom_map_dir, exist_ok=True)
        return self.custom_map_dir

    def get_student_rl_policy_dir(self):
        os.makedirs(self.student_rl_policy_dir, exist_ok=True)
        return self.student_rl_policy_dir

    def get_training_map_files(self):
        if not os.path.isdir(self.training_map_dir):
            return []
        return sorted(
            os.path.join(self.training_map_dir, filename)
            for filename in os.listdir(self.training_map_dir)
            if filename.lower().endswith(".json")
        )

    def clear_student_rl_training(self):
        self.student_rl_training = None
        self.student_rl_qtable = None
        self.student_rl_best_result = None
        self.student_rl_run_qtable = None
        self.student_rl_run_signature = None
        self.student_rl_run_policy_data = None

    def save_map(self, file_path):
        data = save_map_json(
            file_path=file_path,
            map_obj=self.map,
            start=self.start,
            goal=self.goal,
            checkpoints=self.sub_goals,
            name=self.map_name,
            metadata={"source": "LineMap Robot Simulator"},
        )
        self.current_map_file_path = file_path
        return f"Saved map '{data['name']}' to {file_path}."

    def load_map(self, file_path):
        map_data = load_map_json(file_path)
        self.map = map_data.line_map
        self.map_width = map_data.width
        self.map_height = map_data.height
        self.map_size = max(self.map_width, self.map_height)
        self.map_name = map_data.name
        self.current_map_file_path = file_path
        self.start = map_data.start
        self.goal = map_data.goal
        self.sub_goals = list(map_data.checkpoints)
        self.active_training_path = None
        self.clear_student_rl_training()
        self.reset_robot()
        return (
            f"Loaded map '{self.map_name}' ({self.map_width}x{self.map_height}) "
            f"with {len(self.sub_goals)} checkpoints."
        )

    def load_training_map(self, training_index):
        training_maps = self.get_training_map_files()
        if not training_maps:
            return None, "No training maps found."
        index = max(1, min(int(training_index), len(training_maps))) - 1
        message = self.load_map(training_maps[index])
        return training_maps[index], message

    def get_start(self):
        return self.start

    def get_goal(self):
        return self.goal

    def get_sub_goals(self):
        return list(self.sub_goals)

    def get_score(self):
        return self.score

    def get_nodes(self):
        return self.map.nodes

    def get_edges(self):
        return self.map.edges

    def reset_robot(self):
        self.robot = Robot(self.start[0], self.start[1], map_obj=self.map, direction=(0, 1))
        self.student_rl_run_qtable = None
        self.student_rl_run_signature = None
        self.student_rl_run_policy_data = None
        self.reset_score_state()

    def reset_score_state(self):
        self.reached_sub_goals = set()
        self.final_goal_reached = False
        self.num_moves = 0
        self.num_turns = 0
        self.min_manhattan_distance = manhattan(self.start, self.goal)
        self.score = self.current_score()

    def current_score(self):
        return calculate_score(
            self.num_moves,
            self.num_turns,
            len(self.reached_sub_goals),
            self.final_goal_reached,
            self.min_manhattan_distance,
        )

    def get_astar_path(self, use_known_map=False):
        map_obj = self.robot.known_map if use_known_map else self.map
        return astar_route(
            map_obj,
            (self.robot.x, self.robot.y),
            self.goal,
            self.get_remaining_sub_goals()
        )

    def get_qlearning_path(self, episodes=2500, use_known_map=False, from_robot=False):
        map_obj = self.robot.known_map if use_known_map else self.map
        start = (self.robot.x, self.robot.y) if from_robot else self.start
        random.seed(42)
        sub_goals = self.get_remaining_sub_goals() if from_robot else self.sub_goals
        return qlearning_path(map_obj, start, self.goal, sub_goals=sub_goals, episodes=episodes)

    def start_student_rl_training(self, episodes=500, batch_size=25, seed=42):
        self.student_rl_qtable, policy_data = self.load_student_rl_policy()
        self.student_rl_best_result = None
        self.student_rl_training = {
            "episode": 0,
            "episodes": max(1, int(episodes)),
            "previous_episodes": int((policy_data or {}).get("episodes", 0)),
            "batch_size": max(1, int(batch_size)),
            "epsilon": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "alpha": 0.2,
            "gamma": 0.9,
            "rng": random.Random(seed),
            "last_result": None,
        }
        action = "Updating existing" if policy_data else "Creating"
        return (
            f"{action} shared Student RL policy for {episodes} more episodes "
            f"on '{self.map_name}'."
        )

    def train_student_rl_batch(self):
        if self.student_rl_training is None or self.student_rl_qtable is None:
            self.start_student_rl_training()

        training = self.student_rl_training
        last_result = training["last_result"]

        for _ in range(training["batch_size"]):
            if training["episode"] >= training["episodes"]:
                break

            training["episode"] += 1
            last_result = StudentRL.run_episode(
                qtable=self.student_rl_qtable,
                map_obj=self.map,
                start=self.start,
                goal=self.goal,
                checkpoints=self.sub_goals,
                episode=training["episode"],
                epsilon=training["epsilon"],
                alpha=training["alpha"],
                gamma=training["gamma"],
                rng=training["rng"],
                learning=True,
            )
            training["last_result"] = last_result
            if StudentRL.is_better_result(last_result, self.student_rl_best_result):
                self.student_rl_best_result = last_result
            training["epsilon"] = max(
                training["epsilon_end"],
                training["epsilon"] * training["epsilon_decay"],
            )

        policy_result = StudentRL.simulate_policy(
            self.student_rl_qtable,
            self.map,
            self.start,
            self.goal,
            self.sub_goals,
        )
        done = training["episode"] >= training["episodes"]
        message = StudentRL.format_episode_log(last_result or policy_result, len(self.sub_goals))
        if done:
            policy_file = self.save_student_rl_policy(
                self.student_rl_qtable,
                training["previous_episodes"] + training["episodes"],
                best_result=self.student_rl_best_result,
                last_result=last_result,
                policy_result=policy_result,
                metadata={
                    "last_training_mode": "current_map",
                    "last_training_map": self.map_name,
                    "last_training_episodes": training["episodes"],
                    "total_recorded_episodes": training["previous_episodes"] + training["episodes"],
                },
            )
            rel_policy_file = os.path.relpath(policy_file, root_dir)
            message += f" Training complete. Saved policy to {rel_policy_file}."
        return done, message, policy_result.path

    def _policy_file_path(self, map_name=None, map_file=None):
        return os.path.join(self.get_student_rl_policy_dir(), STUDENT_RL_POLICY_FILENAME)

    def load_student_rl_policy(self):
        file_path = self._policy_file_path()
        if not os.path.exists(file_path):
            return StudentRL.create_qtable(), None
        qtable, policy_data = StudentRL.load_policy_json(file_path)
        return qtable, policy_data

    def save_student_rl_policy(
        self,
        qtable,
        episodes,
        best_result=None,
        last_result=None,
        policy_result=None,
        map_name=None,
        map_file=None,
        start=None,
        goal=None,
        checkpoints=None,
        metadata=None,
    ):
        file_path = self._policy_file_path()
        map_file_for_json = os.path.relpath(map_file, root_dir) if map_file else None
        StudentRL.save_policy_json(
            file_path=file_path,
            qtable=qtable,
            episodes=episodes,
            map_name="student_rl_shared_policy",
            map_file=map_file_for_json,
            start=start,
            goal=goal,
            checkpoints=checkpoints or [],
            best_result=best_result,
            last_result=last_result,
            policy_result=policy_result,
            metadata={
                "source": "LineMap Robot Simulator",
                **dict(metadata or {}),
            },
        )
        return file_path

    def train_student_rl_all_training_maps(self, episodes=500, seed=42):
        map_files = self.get_training_map_files()
        if not map_files:
            return "No training maps found."

        shared_qtable, policy_data = self.load_student_rl_policy()
        previous_episodes = int((policy_data or {}).get("episodes", 0))
        loaded_maps = []

        for index, file_path in enumerate(map_files, start=1):
            map_data = load_map_json(file_path)
            loaded_maps.append((file_path, map_data))
            StudentRL.train(
                map_obj=map_data.line_map,
                start=map_data.start,
                goal=map_data.goal,
                checkpoints=map_data.checkpoints,
                episodes=episodes,
                log_every=max(1, episodes),
                seed=seed + index,
                qtable=shared_qtable,
            )

        reached = 0
        failed = []
        evaluation = []

        for file_path, map_data in loaded_maps:
            policy_result = StudentRL.simulate_policy(
                shared_qtable,
                map_data.line_map,
                map_data.start,
                map_data.goal,
                map_data.checkpoints,
            )
            rel_map_file = os.path.relpath(file_path, root_dir)
            evaluation.append(
                {
                    "map_name": map_data.name,
                    "map_file": rel_map_file,
                    "reached_goal": policy_result.reached_goal,
                    "checkpoints_reached": policy_result.checkpoints_reached,
                    "total_checkpoints": len(map_data.checkpoints),
                    "score": policy_result.score,
                    "moves": policy_result.moves,
                    "turns": policy_result.turns,
                    "path": [list(coord) for coord in policy_result.path],
                }
            )
            if policy_result.reached_goal:
                reached += 1
            else:
                failed.append(map_data.name)

        policy_file = self.save_student_rl_policy(
            qtable=shared_qtable,
            episodes=previous_episodes + episodes * len(loaded_maps),
            map_file=None,
            start=None,
            goal=None,
            checkpoints=[],
            metadata={
                "last_training_mode": "all_training_maps",
                "episodes_per_map": episodes,
                "previous_recorded_episodes": previous_episodes,
                "total_recorded_episodes": previous_episodes + episodes * len(loaded_maps),
                "training_maps": [
                    {
                        "map_name": map_data.name,
                        "map_file": os.path.relpath(file_path, root_dir),
                    }
                    for file_path, map_data in loaded_maps
                ],
                "evaluation": evaluation,
            },
        )
        rel_policy_file = os.path.relpath(policy_file, root_dir)
        message = (
            f"Trained one shared Student RL model on {len(map_files)} maps "
            f"({episodes} episodes each). Saved one policy to {rel_policy_file}. "
            f"Greedy policy reached goal on {reached}/{len(map_files)} maps."
        )
        if failed:
            message += " Failed: " + ", ".join(failed[:3])
            if len(failed) > 3:
                message += f", +{len(failed) - 3} more"
            message += "."
        return message

    def get_student_rl_policy_path(self):
        if not self.student_rl_qtable:
            return None
        return StudentRL.get_policy_path(
            self.student_rl_qtable,
            self.map,
            self.start,
            self.goal,
            self.sub_goals,
        )

    def stop_student_rl_training(self):
        self.student_rl_training = None
        return "Student RL training stopped."

    def reset_student_rl_policy(self):
        self.clear_student_rl_training()
        policy_file = self._policy_file_path()
        if os.path.exists(policy_file):
            os.remove(policy_file)
            rel_policy_file = os.path.relpath(policy_file, root_dir)
            return f"Reset Student RL training. Deleted {rel_policy_file}."
        rel_policy_file = os.path.relpath(policy_file, root_dir)
        return f"Reset Student RL training. No saved policy found at {rel_policy_file}."

    def get_remaining_sub_goals(self):
        return [coord for coord in self.sub_goals if coord not in self.reached_sub_goals]

    def is_sub_goal_reached(self, coord):
        return coord in self.reached_sub_goals

    def set_goal(self, coord):
        if coord not in self.map.nodes or coord == self.start:
            return False
        self.map.nodes[coord].isBlock = False
        self.goal = coord
        self.current_map_file_path = None
        self.sub_goals = [sub_goal for sub_goal in self.sub_goals if sub_goal != coord]
        self.clear_student_rl_training()
        self.reset_robot()
        return True

    def toggle_sub_goal(self, coord):
        if coord not in self.map.nodes or coord in (self.start, self.goal):
            return None
        self.map.nodes[coord].isBlock = False
        self.current_map_file_path = None
        if coord in self.sub_goals:
            self.sub_goals.remove(coord)
            self.clear_student_rl_training()
            self.reset_robot()
            return False
        self.sub_goals.append(coord)
        self.clear_student_rl_training()
        self.reset_robot()
        return True

    def toggle_node_obstacle(self, coord):
        if coord not in self.map.nodes or coord in (self.start, self.goal) or coord in self.sub_goals:
            return None
        node = self.map.nodes[coord]
        node.isBlock = not bool(node.isBlock)
        self.current_map_file_path = None
        self.clear_student_rl_training()
        self.reset_robot()
        return bool(node.isBlock)

    def toggle_edge_obstacle(self, coord1, coord2):
        if coord1 not in self.map.nodes or coord2 not in self.map.nodes:
            return None
        if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) != 1:
            return None

        matching_edges = []
        for edge in self.map.edges:
            src = (edge.src.x, edge.src.y)
            dest = (edge.dest.x, edge.dest.y)
            if (src == coord1 and dest == coord2) or (src == coord2 and dest == coord1):
                matching_edges.append(edge)

        if not matching_edges:
            return None

        new_state = not any(bool(edge.isBlock) for edge in matching_edges)
        for edge in matching_edges:
            edge.isBlock = new_state

        self.current_map_file_path = None
        self.clear_student_rl_training()
        self.reset_robot()
        return new_state

    def apply_position_rewards(self):
        coord = (self.robot.x, self.robot.y)
        messages = []
        if coord in self.sub_goals and coord not in self.reached_sub_goals:
            self.reached_sub_goals.add(coord)
            messages.append(f"Checkpoint {coord} recorded (+{SUB_GOAL_REWARD} if goal is reached).")
        if coord == self.goal and not self.final_goal_reached:
            self.final_goal_reached = True
            messages.append(f"Reached final goal (+{FINAL_GOAL_REWARD}); score finalized.")
        if not messages:
            return ""
        return " " + " ".join(messages)

    def apply_action_score(self, before_coord, before_direction):
        if self.final_goal_reached:
            return " Score is locked after reaching the final goal."

        messages = []
        if before_direction != self.robot.direction:
            self.num_turns += 1
            messages.append(f"Turn {TURN_COST}.")
        if before_coord != (self.robot.x, self.robot.y):
            self.num_moves += 1
            current_coord = (self.robot.x, self.robot.y)
            self.min_manhattan_distance = min(
                self.min_manhattan_distance,
                manhattan(current_coord, self.goal),
            )
            messages.append(f"Move {MOVE_COST}.")
            position_message = self.apply_position_rewards()
            if position_message:
                messages.append(position_message.strip())
        self.score = self.current_score()
        if not messages:
            return f" Score={self.score}."
        return f" {' '.join(messages)} Score={self.score}."

    def step_toward_goal_with_known_astar(self):
        if self.final_goal_reached:
            return True, "Reached final goal."

        path = self.get_astar_path(use_known_map=True)
        if path is None or len(path) < 2:
            return False, "No path found on known_map."

        next_coord = path[1]
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        moved = self.robot._moveTo(next_coord)
        score_message = self.apply_action_score(before_coord, before_direction)
        if self.final_goal_reached:
            return True, f"Moved to {next_coord}.{score_message}"
        if moved:
            return None, f"Moved to {next_coord}.{score_message}"
        return None, f"Discovered obstacle near {next_coord}; replanning.{score_message}"

    def step_toward_goal_with_known_qlearning(self, episodes=2500):
        if self.final_goal_reached:
            return True, "Reached final goal.", None

        path = self.get_qlearning_path(
            episodes=episodes,
            use_known_map=True,
            from_robot=True
        )
        if path is None or len(path) < 2:
            return False, "Q-learning found no path on known_map.", path

        next_coord = path[1]
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        moved = self.robot._moveTo(next_coord)
        score_message = self.apply_action_score(before_coord, before_direction)
        if self.final_goal_reached:
            return True, f"Q-learning moved to {next_coord}.{score_message}", path
        if moved:
            return None, f"Q-learning moved to {next_coord}.{score_message}", path
        return None, f"Q-learning discovered obstacle near {next_coord}; replanning.{score_message}", path

    def _student_rl_run_state_signature(self):
        return (
            tuple(sorted(self.robot.discovered_nodes)),
            tuple(sorted(self.robot.discovered_edges)),
            tuple(self.get_remaining_sub_goals()),
            self.goal,
        )

    def _get_known_student_rl_policy(self, episodes=500):
        if self.robot.known_map is None:
            return None, None

        signature = self._student_rl_run_state_signature()
        start = (self.robot.x, self.robot.y)
        remaining_sub_goals = self.get_remaining_sub_goals()

        if self.student_rl_run_qtable is None or signature != self.student_rl_run_signature:
            self.student_rl_run_qtable, self.student_rl_run_policy_data = self.load_student_rl_policy()
            if self.student_rl_run_policy_data is None:
                self.student_rl_run_qtable = None
                return None, None
            self.student_rl_run_signature = signature

        policy = StudentRL.simulate_policy(
            self.student_rl_run_qtable,
            self.robot.known_map,
            start,
            self.goal,
            remaining_sub_goals,
            initial_direction=self.robot.direction,
        )
        return policy, self.student_rl_run_policy_data

    def step_toward_goal_with_known_student_rl(self, episodes=500):
        if self.final_goal_reached:
            return True, "Reached final goal.", None

        policy, policy_data = self._get_known_student_rl_policy(episodes=episodes)
        if policy is None:
            return (
                False,
                "No saved Student RL policy found. Click Train Student RL or Train One RL Model first.",
                None,
            )

        path = policy.path
        if not policy.reached_goal or len(path) < 2:
            episodes_text = policy_data.get("episodes", 0) if policy_data else 0
            return (
                False,
                f"Saved Student RL policy found no complete path on known_map "
                f"(trained episodes recorded: {episodes_text}).",
                path,
            )

        next_coord = path[1]
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        moved = self.robot._moveTo(next_coord)
        score_message = self.apply_action_score(before_coord, before_direction)

        if self.final_goal_reached:
            return True, f"Saved Student RL policy moved to {next_coord}.{score_message}", path
        if moved:
            if self._student_rl_run_state_signature() != self.student_rl_run_signature:
                self.student_rl_run_qtable = None
            return None, f"Saved Student RL policy moved to {next_coord}.{score_message}", path

        self.student_rl_run_qtable = None
        self.student_rl_run_signature = None
        self.student_rl_run_policy_data = None
        return None, f"Saved Student RL policy discovered obstacle near {next_coord}; replanning.{score_message}", path

    def get_robot_x(self):
        return self.robot.x

    def get_robot_y(self):
        return self.robot.y

    def get_robot_direction(self):
        return self.robot.direction

    def get_robot_direction_name(self):
        return self.robot._get_direction_name()

    def forward(self):
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        moved = self.robot.forward(1)
        score_message = self.apply_action_score(before_coord, before_direction)
        return moved, score_message

    def backward(self):
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        moved = self.robot.backward(1)
        score_message = self.apply_action_score(before_coord, before_direction)
        return moved, score_message

    def turn_left(self):
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        self.robot.turnLeft(90)
        score_message = self.apply_action_score(before_coord, before_direction)
        return self.get_robot_direction_name(), score_message

    def turn_right(self):
        before_coord = (self.robot.x, self.robot.y)
        before_direction = self.robot.direction
        self.robot.turnRight(90)
        score_message = self.apply_action_score(before_coord, before_direction)
        return self.get_robot_direction_name(), score_message

    def is_edge_discovered(self, coord1, coord2):
        edge_key = tuple(sorted([coord1, coord2]))
        return edge_key in self.robot.discovered_edges

    def is_node_discovered(self, coord):
        return coord in self.robot.discovered_nodes

