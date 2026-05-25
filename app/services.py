import sys
import os

# Add src folder to path to allow importing map and agent
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from map import LineMap
from agent import Robot

class SimulationService:
    def __init__(self, map_size=5):
        self.map_size = map_size
        self.map = LineMap()
        self.map.createMap(self.map_size)
        
        # Gọi phương thức thiết lập vật cản
        self.configure_obstacles()

        # Starts at (0,0) facing UP (0, 1)
        self.robot = Robot(0, 0, map_obj=self.map, direction=(0, 1))

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

    def get_nodes(self):
        return self.map.nodes

    def get_edges(self):
        return self.map.edges

    def get_robot_x(self):
        return self.robot.x

    def get_robot_y(self):
        return self.robot.y

    def get_robot_direction(self):
        return self.robot.direction

    def get_robot_direction_name(self):
        return self.robot._get_direction_name()

    def forward(self):
        return self.robot.forward(1)

    def backward(self):
        return self.robot.backward(1)

    def turn_left(self):
        self.robot.turnLeft(90)
        return self.get_robot_direction_name()

    def turn_right(self):
        self.robot.turnRight(90)
        return self.get_robot_direction_name()

    def is_edge_discovered(self, coord1, coord2):
        edge_key = tuple(sorted([coord1, coord2]))
        return edge_key in self.robot.discovered_edges

    def is_node_discovered(self, coord):
        return coord in self.robot.discovered_nodes

