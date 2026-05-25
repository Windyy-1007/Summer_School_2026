from map import Node, Edge, LineMap

class Robot:
    def __init__(self, currX, currY, map_obj=None, direction=(0, 1)):
        """
        Initializes the robot at (currX, currY).
        
        Args:
            currX (int): Initial X coordinate.
            currY (int): Initial Y coordinate.
            map_obj (LineMap, optional): The map the robot is moving in.
            direction (tuple, optional): Initial heading vector, defaults to (0, 1) [UP/NORTH].
                                        Can be (0, 1) [UP], (1, 0) [RIGHT], (0, -1) [DOWN], (-1, 0) [LEFT].
        """
        self.x = currX
        self.y = currY
        self.map = map_obj
        self.direction = direction
        self.discovered_nodes = set()
        self.discovered_edges = set()
        self.check_vision()

    def check_vision(self):
        """
        Looks in front of the robot. If there is a blocked edge in front, it discovers and remembers it.
        """
        if not self.map:
            return
        curr_coord = (self.x, self.y)
        dx, dy = self.direction
        front_coord = (self.x + dx, self.y + dy)
        
        edge = self._get_edge(curr_coord, front_coord)
        if edge and edge.isBlock == 1:
            edge_key = tuple(sorted([curr_coord, front_coord]))
            self.discovered_edges.add(edge_key)

    def _get_direction_name(self):
        """
        Maps the direction vector to a friendly string name (up, right, down, left).
        """
        mapping = {
            (0, 1): "up",
            (1, 0): "right",
            (0, -1): "down",
            (-1, 0): "left"
        }
        return mapping.get(self.direction, str(self.direction))

    def turnRight(self, angle=90):
        """
        Rotates the robot clockwise by the specified angle (must be a multiple of 90).
        """
        steps = (angle // 90) % 4
        for _ in range(steps):
            # Rotates (dx, dy) to (dy, -dx)
            self.direction = (self.direction[1], -self.direction[0])
        print(f"Robot turned right. New direction: {self._get_direction_name()}")
        self.check_vision()

    def turnLeft(self, angle=90):
        """
        Rotates the robot counter-clockwise by the specified angle (must be a multiple of 90).
        """
        steps = (angle // 90) % 4
        for _ in range(steps):
            # Rotates (dx, dy) to (-dy, dx)
            self.direction = (-self.direction[1], self.direction[0])
        print(f"Robot turned left. New direction: {self._get_direction_name()}")
        self.check_vision()

    def _get_edge(self, src_coord, dest_coord):
        """
        Helper method to retrieve the edge between two coordinates if it exists.
        """
        if not self.map:
            return None
        for e in self.map.edges:
            # Check matching coordinates
            if (e.src.x, e.src.y) == src_coord and (e.dest.x, e.dest.y) == dest_coord:
                return e
        return None

    def _step(self, move_forward=True):
        """
        Helper method to attempt moving 1 unit step (forward or backward).
        Implements obstacle vision rules for edges and nodes.
        
        Returns:
            bool: True if step succeeded, False otherwise.
        """
        if not self.map:
            print("Error: No map is set for the robot.")
            return False

        current_coord = (self.x, self.y)
        
        # Calculate target position based on direction and movement type
        dx, dy = self.direction
        if move_forward:
            target_coord = (self.x + dx, self.y + dy)
            movement_dir = (dx, dy)
        else:
            target_coord = (self.x - dx, self.y - dy)
            movement_dir = (-dx, -dy)

        # 0. Check if target coordinates are within map boundaries
        if target_coord not in self.map.nodes:
            print(f"Cannot move: Destination {target_coord} is out of map bounds.")
            return False

        # 1. Edge obstacle detection rule:
        # "hướng của robot phải đúng hướng của cạnh bị chặn thì robot mới nhận biết được là cạnh đó không đi được"
        edge = self._get_edge(current_coord, target_coord)
        if edge and edge.isBlock == 1:
            # Robot heading is self.direction.
            # Heading aligns with edge direction if movement_dir matches self.direction.
            is_aligned_with_heading = (movement_dir == self.direction)
            
            # Learn that the edge is blocked
            edge_key = tuple(sorted([current_coord, target_coord]))
            self.discovered_edges.add(edge_key)

            if is_aligned_with_heading:
                print(f"Edge obstacle detected: The edge {current_coord} -> {target_coord} is blocked in the robot's heading direction. Action: Stayed at {current_coord}.")
                return False
            else:
                # Moving backward, heading does not align with edge direction.
                # Robot fails to recognize it beforehand, tries to move, but gets blocked by the physical obstacle.
                print(f"Failed to recognize blocked edge (moving backward). Attempted to move {current_coord} -> {target_coord} but was physically blocked. Action: Stayed at {current_coord}.")
                return False

        # 2. Node obstacle detection rule:
        # "robot phải đi lên cạnh, mới thấy node bị chắn, và làm hành động quay trở lại node mới đi"
        target_node = self.map.nodes[target_coord]
        if target_node.isBlock == 1:
            # The robot starts moving onto the edge, then spots that the destination node is blocked,
            # and performs the action to return to the original starting node.
            
            # Learn that the node is blocked
            self.discovered_nodes.add(target_coord)
            
            print(f"Moved onto edge towards {target_coord}, detected that target Node {target_coord} is blocked. Action: Returned to original node {current_coord}.")
            return False

        # If both edge and node are clear, move successfully
        self.x, self.y = target_coord
        self.check_vision()
        print(f"Successfully moved to {target_coord}.")
        return True

    def forward(self, dist=1):
        """
        Moves the robot forward by `dist` units, step-by-step.
        """
        print(f"Attempting to move forward {dist} steps...")
        for i in range(dist):
            if not self._step(move_forward=True):
                print(f"Movement interrupted at step {i+1}/{dist}. Current position: ({self.x}, {self.y})")
                return False
        return True

    def backward(self, dist=1):
        """
        Moves the robot backward by `dist` units, step-by-step.
        """
        print(f"Attempting to move backward {dist} steps...")
        for i in range(dist):
            if not self._step(move_forward=False):
                print(f"Movement interrupted at step {i+1}/{dist}. Current position: ({self.x}, {self.y})")
                return False
        return True

    def _moveTo(self, target_coord):
        """
        Internal helper to move the robot to an adjacent coordinate (target_coord).
        If the target is to the right of the current heading, it turns right first, then moves forward.
        If the target is to the left of the current heading, it turns left first, then moves forward.
        If the target is in front, it moves forward.
        If the target is behind, it moves backward.
        """
        if not self.map:
            print("Error: No map is set for the robot.")
            return False

        if target_coord not in self.map.nodes:
            print(f"Error: Target {target_coord} is not in the map.")
            return False

        dx = target_coord[0] - self.x
        dy = target_coord[1] - self.y

        # Target must be a direct neighbor
        if abs(dx) + abs(dy) != 1:
            print(f"Error: Target {target_coord} is not adjacent to current position ({self.x}, {self.y}).")
            return False

        target_dir = (dx, dy)

        # Determine relative action based on current direction
        right_dir = (self.direction[1], -self.direction[0])
        left_dir = (-self.direction[1], self.direction[0])
        opposite_dir = (-self.direction[0], -self.direction[1])

        if target_dir == self.direction:
            return self.forward(1)
        elif target_dir == opposite_dir:
            return self.backward(1)
        elif target_dir == right_dir:
            self.turnRight(90)
            return self.forward(1)
        elif target_dir == left_dir:
            self.turnLeft(90)
            return self.forward(1)

        return False


