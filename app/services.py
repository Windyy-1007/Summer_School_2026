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

FINAL_GOAL_REWARD = 400
SUB_GOAL_REWARD = 150
MOVE_COST = -3
TURN_COST = -1

class SimulationService:
    def __init__(self, map_size=5):
        self.map_size = map_size
        self.start = (0, 0)
        self.goal = (min(4, map_size - 1), min(4, map_size - 1))
        self.sub_goals = [(1, 4), (4, 1)]
        self.reached_sub_goals = set()
        self.final_goal_reached = False
        self.score = 0
        self.map = LineMap()
        self.map.createMap(self.map_size)
        
        # Gọi phương thức thiết lập vật cản
        self.configure_obstacles()

        # Starts at (0,0) facing UP (0, 1)
        self.robot = Robot(self.start[0], self.start[1], map_obj=self.map, direction=(0, 1))

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
        self.reached_sub_goals = set()
        self.final_goal_reached = False
        self.score = 0

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

    def get_remaining_sub_goals(self):
        return [coord for coord in self.sub_goals if coord not in self.reached_sub_goals]

    def is_sub_goal_reached(self, coord):
        return coord in self.reached_sub_goals

    def set_goal(self, coord):
        if coord not in self.map.nodes or coord == self.start:
            return False
        self.map.nodes[coord].isBlock = False
        self.goal = coord
        self.sub_goals = [sub_goal for sub_goal in self.sub_goals if sub_goal != coord]
        self.reset_robot()
        return True

    def toggle_sub_goal(self, coord):
        if coord not in self.map.nodes or coord in (self.start, self.goal):
            return None
        self.map.nodes[coord].isBlock = False
        if coord in self.sub_goals:
            self.sub_goals.remove(coord)
            self.reset_robot()
            return False
        self.sub_goals.append(coord)
        self.reset_robot()
        return True

    def toggle_node_obstacle(self, coord):
        if coord not in self.map.nodes or coord in (self.start, self.goal) or coord in self.sub_goals:
            return None
        node = self.map.nodes[coord]
        node.isBlock = not bool(node.isBlock)
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

        self.reset_robot()
        return new_state

    def apply_position_rewards(self):
        coord = (self.robot.x, self.robot.y)
        if coord in self.sub_goals and coord not in self.reached_sub_goals:
            self.reached_sub_goals.add(coord)
            self.score += SUB_GOAL_REWARD
            return f" Reached sub-goal {coord}: +{SUB_GOAL_REWARD}."
        if coord == self.goal and not self.final_goal_reached:
            self.final_goal_reached = True
            self.score += FINAL_GOAL_REWARD
            return f" Reached final goal: +{FINAL_GOAL_REWARD}."
        return ""

    def apply_action_score(self, before_coord, before_direction):
        message = ""
        if before_direction != self.robot.direction:
            self.score += TURN_COST
            message += f" Turn {TURN_COST}."
        if before_coord != (self.robot.x, self.robot.y):
            self.score += MOVE_COST
            message += f" Move {MOVE_COST}."
            message += self.apply_position_rewards()
        return message

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

