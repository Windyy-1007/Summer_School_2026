import unittest
import random

from src.map import LineMap
from src.agent import Robot
from Algo.Astar import astar, astar_route
from Algo.QLearning import get_path as ql_get_path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def is_valid_path(map_obj, path):
    """
    Returns True if every consecutive pair in path is:
      - adjacent (Manhattan distance 1)
      - connected by an unblocked edge
      - destination node is unblocked
    """
    if not path:
        return False
    for i in range(len(path) - 1):
        curr, nxt = path[i], path[i + 1]
        if abs(curr[0] - nxt[0]) + abs(curr[1] - nxt[1]) != 1:
            return False
        if map_obj.nodes[nxt].isBlock:
            return False
        edge_ok = False
        for edge in map_obj.edges:
            if (edge.src.x, edge.src.y) == curr and (edge.dest.x, edge.dest.y) == nxt:
                if edge.isBlock:
                    return False
                edge_ok = True
                break
        if not edge_ok:
            return False
    return True


# ---------------------------------------------------------------------------
# A* tests
# ---------------------------------------------------------------------------

class TestAstar(unittest.TestCase):
    def setUp(self):
        self.lm = LineMap()
        self.lm.createMap(5)

    def test_clear_path_found(self):
        """A* finds a path on a clear 5x5 grid."""
        path = astar(self.lm, (0, 0), (4, 4))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertTrue(is_valid_path(self.lm, path))

    def test_optimal_path_length(self):
        """A* returns the shortest (Manhattan) path on a clear grid."""
        path = astar(self.lm, (0, 0), (4, 4))
        # Manhattan distance = 8 → 9 nodes
        self.assertEqual(len(path), 9)

    def test_start_equals_goal(self):
        """A* returns a single-element path when start == goal."""
        self.assertEqual(astar(self.lm, (2, 2), (2, 2)), [(2, 2)])

    def test_blocked_goal_returns_none(self):
        """A* returns None when the goal node is blocked."""
        self.lm.set_obstacles(node=(4, 4))
        self.assertIsNone(astar(self.lm, (0, 0), (4, 4)))

    def test_path_around_blocked_node(self):
        """A* finds a detour when a node on the direct path is blocked."""
        self.lm.set_obstacles(node=(2, 2))
        path = astar(self.lm, (0, 2), (4, 2))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], (4, 2))
        self.assertNotIn((2, 2), path)
        self.assertTrue(is_valid_path(self.lm, path))

    def test_path_around_blocked_edge(self):
        """A* finds a detour when an edge on the direct path is blocked."""
        self.lm.set_obstacles(edge=((2, 2), (2, 3)))
        path = astar(self.lm, (2, 2), (2, 4))
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 4))
        self.assertTrue(is_valid_path(self.lm, path))

    def test_enclosed_goal_returns_none(self):
        """A* returns None when the goal is completely surrounded by blocked edges."""
        lm = LineMap()
        lm.createMap(3)
        lm.set_obstacles(edge=((1, 1), (1, 2)))
        lm.set_obstacles(edge=((1, 1), (1, 0)))
        lm.set_obstacles(edge=((1, 1), (0, 1)))
        lm.set_obstacles(edge=((1, 1), (2, 1)))
        self.assertIsNone(astar(lm, (0, 0), (1, 1)))

    def test_invalid_start_returns_none(self):
        """A* returns None for a start coordinate outside the map."""
        self.assertIsNone(astar(self.lm, (9, 9), (4, 4)))

    def test_invalid_goal_returns_none(self):
        """A* returns None for a goal coordinate outside the map."""
        self.assertIsNone(astar(self.lm, (0, 0), (9, 9)))

    def test_route_visits_sub_goals_before_final_goal(self):
        """A* route visits ordered sub-goals before the final goal."""
        path = astar_route(self.lm, (0, 0), (4, 4), sub_goals=[(0, 4), (4, 0)])
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertLess(path.index((0, 4)), path.index((4, 0)))
        self.assertLess(path.index((4, 0)), path.index((4, 4)))
        self.assertTrue(is_valid_path(self.lm, path))


# ---------------------------------------------------------------------------
# Q-Learning tests
# ---------------------------------------------------------------------------

class TestQLearning(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.lm = LineMap()
        self.lm.createMap(5)

    def test_clear_path_found(self):
        """Q-learning finds a valid path on a clear 5x5 grid."""
        path = ql_get_path(self.lm, (0, 0), (4, 4), episodes=3000)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertTrue(is_valid_path(self.lm, path))

    def test_no_duplicate_nodes_in_path(self):
        """Q-learning path contains no repeated coordinates (no loops)."""
        path = ql_get_path(self.lm, (0, 0), (4, 4), episodes=3000)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(set(path)))

    def test_start_equals_goal(self):
        """Q-learning returns single-element path when start == goal."""
        self.assertEqual(ql_get_path(self.lm, (2, 2), (2, 2)), [(2, 2)])

    def test_blocked_goal_returns_none(self):
        """Q-learning returns None when the goal node is blocked."""
        self.lm.set_obstacles(node=(4, 4))
        self.assertIsNone(ql_get_path(self.lm, (0, 0), (4, 4)))

    def test_path_around_blocked_node(self):
        """Q-learning finds a valid path when a node on the direct route is blocked."""
        self.lm.set_obstacles(node=(2, 2))
        path = ql_get_path(self.lm, (0, 0), (4, 4), episodes=3000)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertNotIn((2, 2), path)
        self.assertTrue(is_valid_path(self.lm, path))

    def test_path_around_blocked_edge(self):
        """Q-learning finds a valid path when a direct edge is blocked."""
        self.lm.set_obstacles(edge=((2, 2), (2, 3)))
        path = ql_get_path(self.lm, (2, 2), (2, 4), episodes=3000)
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (2, 4))
        self.assertTrue(is_valid_path(self.lm, path))

    def test_enclosed_goal_returns_none(self):
        """Q-learning returns None when the goal is completely surrounded by blocked edges."""
        lm = LineMap()
        lm.createMap(3)
        lm.set_obstacles(edge=((1, 1), (1, 2)))
        lm.set_obstacles(edge=((1, 1), (1, 0)))
        lm.set_obstacles(edge=((1, 1), (0, 1)))
        lm.set_obstacles(edge=((1, 1), (2, 1)))
        self.assertIsNone(ql_get_path(lm, (0, 0), (1, 1), episodes=3000))

    def test_invalid_start_returns_none(self):
        """Q-learning returns None for a start coordinate outside the map."""
        self.assertIsNone(ql_get_path(self.lm, (9, 9), (4, 4)))

    def test_invalid_goal_returns_none(self):
        """Q-learning returns None for a goal coordinate outside the map."""
        self.assertIsNone(ql_get_path(self.lm, (0, 0), (9, 9)))

    def test_path_visits_sub_goals_before_final_goal(self):
        """Q-learning path visits ordered sub-goals before the final goal."""
        path = ql_get_path(
            self.lm,
            (0, 0),
            (4, 4),
            sub_goals=[(0, 4), (4, 0)],
            episodes=3000
        )
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertLess(path.index((0, 4)), path.index((4, 0)))
        self.assertLess(path.index((4, 0)), path.index((4, 4)))
        self.assertTrue(is_valid_path(self.lm, path))


# ---------------------------------------------------------------------------
# Two-maps approach and navigate_to tests
# ---------------------------------------------------------------------------

class TestTwoMapsAndNavigation(unittest.TestCase):
    def setUp(self):
        self.lm = LineMap()
        self.lm.createMap(5)
        self.robot = Robot(0, 0, map_obj=self.lm, direction=(0, 1))

    def test_known_map_initialized_same_dimensions(self):
        """known_map is created with the same node/edge count as the ground truth."""
        self.assertIsNotNone(self.robot.known_map)
        self.assertEqual(len(self.robot.known_map.nodes), len(self.lm.nodes))
        self.assertEqual(len(self.robot.known_map.edges), len(self.lm.edges))

    def test_known_map_starts_obstacle_free(self):
        """known_map has no obstacles even when ground truth has pre-set ones."""
        lm = LineMap()
        lm.createMap(5)
        lm.set_obstacles(node=(3, 3))
        lm.set_obstacles(edge=((1, 1), (1, 2)))
        robot = Robot(2, 2, map_obj=lm, direction=(1, 0))  # facing away from obstacles
        for node in robot.known_map.nodes.values():
            self.assertFalse(node.isBlock)
        for edge in robot.known_map.edges:
            self.assertFalse(edge.isBlock)

    def test_blocked_edge_in_front_syncs_to_known_map_at_init(self):
        """A blocked edge directly in front at init is discovered by check_vision and synced."""
        lm = LineMap()
        lm.createMap(5)
        lm.set_obstacles(edge=((0, 0), (0, 1)))  # obstacle set BEFORE robot creation
        robot = Robot(0, 0, map_obj=lm, direction=(0, 1))  # facing UP into the obstacle

        # check_vision at init should have detected and synced the edge
        synced = any(
            e.isBlock == 1
            and (e.src.x, e.src.y) == (0, 0)
            and (e.dest.x, e.dest.y) == (0, 1)
            for e in robot.known_map.edges
        )
        self.assertTrue(synced)

    def test_blocked_node_discovery_syncs_to_known_map(self):
        """Discovering a blocked node during movement updates known_map."""
        self.lm.set_obstacles(node=(0, 1))
        # Robot at (0,0) facing UP attempts to enter blocked node (0,1)
        self.robot.forward(1)
        self.assertEqual(self.robot.known_map.nodes[(0, 1)].isBlock, 1)

    def test_blocked_edge_discovery_during_move_syncs_to_known_map(self):
        """Discovering a blocked edge during movement updates known_map."""
        self.lm.set_obstacles(edge=((0, 0), (0, 1)))
        # Obstacle was set after robot init, so not yet in known_map
        self.robot.check_vision()  # explicitly trigger vision
        synced = any(
            e.isBlock == 1 for e in self.robot.known_map.edges
        )
        self.assertTrue(synced)

    def test_navigate_to_clear_path_reaches_target(self):
        """navigate_to successfully reaches the target on a clear map."""
        result = self.robot.navigate_to((4, 4))
        self.assertTrue(result)
        self.assertEqual((self.robot.x, self.robot.y), (4, 4))

    def test_navigate_to_already_at_target(self):
        """navigate_to returns True immediately when already at the target."""
        result = self.robot.navigate_to((0, 0))
        self.assertTrue(result)
        self.assertEqual((self.robot.x, self.robot.y), (0, 0))

    def test_navigate_to_replans_around_discovered_obstacle(self):
        """navigate_to re-plans and still reaches the target after hitting an obstacle."""
        # Block a node on the straight path upward; robot will discover and reroute
        self.lm.set_obstacles(node=(0, 2))
        result = self.robot.navigate_to((0, 4))
        self.assertTrue(result)
        self.assertEqual((self.robot.x, self.robot.y), (0, 4))

    def test_navigate_to_known_blocked_goal_returns_false(self):
        """navigate_to returns False when the goal is known to be blocked."""
        self.lm.set_obstacles(node=(4, 4))
        self.robot.known_map.set_obstacles(node=(4, 4))
        result = self.robot.navigate_to((4, 4))
        self.assertFalse(result)

    def test_navigate_to_no_map_returns_false(self):
        """navigate_to returns False when no map is attached to the robot."""
        headless = Robot(0, 0)
        self.assertFalse(headless.navigate_to((2, 2)))


if __name__ == "__main__":
    unittest.main()
