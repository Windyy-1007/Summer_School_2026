import unittest
from src.map import Map, LineMap, Node, Edge
from src.agent import Robot

class TestMap(unittest.TestCase):
    def test_map_abstract_class(self):
        """Verify Map cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Map()

    def test_linemap_creation_valid(self):
        """Verify LineMap creates correct grid of nodes and edges."""
        lm = LineMap()
        lm.createMap(3) # 3x3 grid
        
        # Grid should have 9 nodes
        self.assertEqual(len(lm.nodes), 9)
        # Coordinate bounds checks
        self.assertIn((0,0), lm.nodes)
        self.assertIn((2,2), lm.nodes)
        self.assertNotIn((3,3), lm.nodes)
        self.assertNotIn((-1,0), lm.nodes)
        
        # 3x3 grid should have 24 directed edges
        self.assertEqual(len(lm.edges), 24)

    def test_linemap_creation_invalid(self):
        """Verify LineMap raises errors for invalid sizes."""
        lm = LineMap()
        with self.assertRaises(ValueError):
            lm.createMap("invalid_size")
        with self.assertRaises(ValueError):
            lm.createMap((1, 2, 3)) # Too many dimensions

    def test_set_obstacles_nodes(self):
        """Verify blocking nodes works correctly with nodes and coordinates."""
        lm = LineMap()
        lm.createMap(3)

        # Block via coordinates tuple
        lm.set_obstacles(node=(1, 1))
        self.assertEqual(lm.nodes[(1, 1)].isBlock, 1)

        # Block via Node object
        node_obj = lm.nodes[(0, 0)]
        lm.set_obstacles(node=node_obj)
        self.assertEqual(node_obj.isBlock, 1)

        # Block invalid node coord
        with self.assertRaises(ValueError):
            lm.set_obstacles(node=(5, 5))
            
        # Invalid node type
        with self.assertRaises(TypeError):
            lm.set_obstacles(node="not_a_node")

    def test_set_obstacles_edges(self):
        """Verify blocking edges works and blocks both directions."""
        lm = LineMap()
        lm.createMap(3)

        # Block via coordinate pair
        lm.set_obstacles(edge=((0,0), (0,1)))
        
        # Check both directions are blocked
        blocked_count = 0
        for e in lm.edges:
            src_coord = (e.src.x, e.src.y)
            dest_coord = (e.dest.x, e.dest.y)
            if (src_coord == (0,0) and dest_coord == (0,1)) or (src_coord == (0,1) and dest_coord == (0,0)):
                self.assertEqual(e.isBlock, 1)
                blocked_count += 1
        
        self.assertEqual(blocked_count, 2)

        # Block invalid edge
        with self.assertRaises(ValueError):
            lm.set_obstacles(edge=((0,0), (2,2))) # No edge exists diagonally
            
        # Invalid edge type
        with self.assertRaises(TypeError):
            lm.set_obstacles(edge="not_an_edge")


class TestRobot(unittest.TestCase):
    def setUp(self):
        self.lm = LineMap()
        self.lm.createMap(3)
        self.robot = Robot(0, 0, map_obj=self.lm, direction=(0, 1)) # facing UP

    def test_robot_initialization(self):
        """Verify robot initializes with correct positions and direction."""
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 0)
        self.assertEqual(self.robot.direction, (0, 1))

    def test_robot_rotations(self):
        """Verify turnRight and turnLeft rotate robot correctly."""
        # Initial UP (0, 1)
        self.robot.turnRight(90) # RIGHT (1, 0)
        self.assertEqual(self.robot.direction, (1, 0))
        
        self.robot.turnRight(180) # LEFT (-1, 0)
        self.assertEqual(self.robot.direction, (-1, 0))

        self.robot.turnLeft(90) # DOWN (0, -1)
        self.assertEqual(self.robot.direction, (0, -1))

        self.robot.turnLeft(270) # LEFT (-1, 0)
        self.assertEqual(self.robot.direction, (-1, 0))

    def test_robot_movement_clear_path(self):
        """Verify robot moves forward and backward on a clear path."""
        # Move forward 2 steps: (0,0) -> (0,1) -> (0,2)
        success = self.robot.forward(2)
        self.assertTrue(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 2)

        # Move backward 1 step: (0,2) -> (0,1)
        success = self.robot.backward(1)
        self.assertTrue(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 1)

    def test_robot_error_no_map(self):
        """Verify robot fails to move if no map is configured."""
        headless_bot = Robot(0, 0)
        self.assertFalse(headless_bot.forward(1))
        self.assertFalse(headless_bot.backward(1))

    def test_robot_out_of_bounds(self):
        """Verify robot cannot move outside the map bounds."""
        # Initial position (0,0) facing UP (0, 1). Backward move leads to (0, -1) (out of bounds)
        success = self.robot.backward(1)
        self.assertFalse(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 0)

        # Move forward 3 steps (from (0,0), limit is 2 since size is 3)
        success = self.robot.forward(3)
        self.assertFalse(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 2) # Interrupted at step 2

    def test_robot_blocked_edge_aligned(self):
        """Verify forward move fails when facing a blocked edge."""
        # Block edge between (0,0) and (0,1)
        self.lm.set_obstacles(edge=((0,0), (0,1)))
        
        # Robot is at (0,0) facing UP (0,1), which matches the blocked edge direction
        success = self.robot.forward(1)
        self.assertFalse(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 0)

    def test_robot_blocked_edge_unaligned_backward(self):
        """Verify backward move fails when facing away from blocked edge."""
        # Block edge between (0,1) and (0,0)
        self.lm.set_obstacles(edge=((0,0), (0,1)))
        
        # Move robot to (0,1) via other path or directly setting coords
        self.robot.x = 0
        self.robot.y = 1
        self.robot.direction = (0, 1) # Facing UP
        
        # Robot moves backward (UP heading, moves DOWN to (0,0)). Cautious: opposite heading.
        success = self.robot.backward(1)
        self.assertFalse(success)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 1)

    def test_robot_blocked_node_rollback(self):
        """Verify rollback when robot encounters a blocked node."""
        # Block node (0, 1)
        self.lm.set_obstacles(node=(0, 1))

        # Try to move forward from (0,0) to (0,1)
        success = self.robot.forward(1)
        self.assertFalse(success)
        # Position should roll back to (0, 0)
        self.assertEqual(self.robot.x, 0)
        self.assertEqual(self.robot.y, 0)

    def test_obstacle_discovery_memory(self):
        """Verify robot dynamically discovers and remembers node and edge blocks."""
        # 1. Edge obstacle discovery:
        # Block edge between (0, 1) and (0, 2)
        self.lm.set_obstacles(edge=((0, 1), (0, 2)))
        edge_key = tuple(sorted([(0, 1), (0, 2)]))
        
        # Initially, robot is at (0,0) facing UP (0,1). It doesn't see (0,1)-(0,2) yet
        self.assertNotIn(edge_key, self.robot.discovered_edges)
        
        # Move robot to (0,1) facing UP
        self.robot.forward(1)
        # Now at (0,1) facing UP. It is directly facing the blocked edge, so it should discover it!
        self.assertIn(edge_key, self.robot.discovered_edges)

        # 2. Node obstacle discovery:
        # Block node (1, 0)
        self.lm.set_obstacles(node=(1, 0))
        
        # Robot is at (0,1) facing UP. It doesn't know (1,0) is blocked.
        self.assertNotIn((1, 0), self.robot.discovered_nodes)
        
        # Move robot back to (0,0)
        self.robot.backward(1) # Now at (0,0) facing UP
        
        # Turn right to face (1,0)
        self.robot.turnRight(90) # Now at (0,0) facing RIGHT
        self.assertNotIn((1, 0), self.robot.discovered_nodes)
        
        # Attempt to move forward to (1,0) (blocked node)
        # Robot moves onto edge towards (1,0), detects it is blocked, and rolls back to (0,0)
        success = self.robot.forward(1)
        self.assertFalse(success)
        # Now it must know (1,0) is blocked!
        self.assertIn((1, 0), self.robot.discovered_nodes)



if __name__ == "__main__":
    unittest.main()
