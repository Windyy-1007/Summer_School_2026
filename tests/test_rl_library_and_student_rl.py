import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from Algo import StudentRL
from rl_library import (
    BACKWARD,
    FORWARD,
    TURN_LEFT,
    calculate_score,
    get_state,
    load_map_from_dict,
    load_map_json,
    save_map_json,
    transition,
)
from map import LineMap


def blocked_edge_exists(map_obj, coord_a, coord_b):
    return any(
        edge.isBlock
        and (
            ((edge.src.x, edge.src.y) == coord_a and (edge.dest.x, edge.dest.y) == coord_b)
            or ((edge.src.x, edge.src.y) == coord_b and (edge.dest.x, edge.dest.y) == coord_a)
        )
        for edge in map_obj.edges
    )


class TestMapJson(unittest.TestCase):
    def test_map_json_round_trip(self):
        line_map = LineMap()
        line_map.createMap((4, 3))
        line_map.set_obstacles(node=(1, 1))
        line_map.set_obstacles(edge=((2, 1), (2, 2)))

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "round_trip.json")
            save_map_json(
                file_path,
                line_map,
                start=(0, 0),
                goal=(3, 2),
                checkpoints=[(0, 2)],
                name="round_trip",
            )
            loaded = load_map_json(file_path)

        self.assertEqual(loaded.name, "round_trip")
        self.assertEqual((loaded.width, loaded.height), (4, 3))
        self.assertEqual(loaded.start, (0, 0))
        self.assertEqual(loaded.goal, (3, 2))
        self.assertEqual(loaded.checkpoints, [(0, 2)])
        self.assertTrue(loaded.line_map.nodes[(1, 1)].isBlock)
        self.assertTrue(blocked_edge_exists(loaded.line_map, (2, 1), (2, 2)))

    def test_load_clears_protected_blocked_nodes(self):
        data = {
            "version": 1,
            "name": "protected_nodes",
            "width": 3,
            "height": 3,
            "start": [0, 0],
            "goal": [2, 2],
            "checkpoints": [[1, 1]],
            "blocked_nodes": [[0, 0], [1, 1], [2, 2], [0, 1]],
            "blocked_edges": [],
            "metadata": {},
        }
        loaded = load_map_from_dict(data)
        self.assertFalse(loaded.line_map.nodes[(0, 0)].isBlock)
        self.assertFalse(loaded.line_map.nodes[(1, 1)].isBlock)
        self.assertFalse(loaded.line_map.nodes[(2, 2)].isBlock)
        self.assertTrue(loaded.line_map.nodes[(0, 1)].isBlock)

    def test_invalid_edge_rejected(self):
        data = {
            "version": 1,
            "name": "bad_edge",
            "width": 3,
            "height": 3,
            "start": [0, 0],
            "goal": [2, 2],
            "checkpoints": [],
            "blocked_nodes": [],
            "blocked_edges": [[[0, 0], [2, 2]]],
            "metadata": {},
        }
        with self.assertRaises(ValueError):
            load_map_from_dict(data)


class TestScoring(unittest.TestCase):
    def test_finished_score_uses_move_turn_checkpoint_goal(self):
        score = calculate_score(
            num_moves=10,
            num_turns=3,
            checkpoints_reached=2,
            reached_goal=True,
            min_manhattan_distance=0,
        )
        self.assertEqual(score, 757)

    def test_unfinished_score_uses_unfinish_minus_min_manhattan_only(self):
        score = calculate_score(
            num_moves=10,
            num_turns=3,
            checkpoints_reached=2,
            reached_goal=False,
            min_manhattan_distance=4,
        )
        self.assertEqual(score, -1004)


class TestStudentTransition(unittest.TestCase):
    def test_forward_clear_move(self):
        line_map = LineMap()
        line_map.createMap(3)
        state = get_state((0, 0))
        result = transition(line_map, state, FORWARD, goal=(2, 2))
        self.assertTrue(result.moved)
        self.assertEqual(result.next_state.position, (0, 1))

    def test_blocked_edge_stays_in_place(self):
        line_map = LineMap()
        line_map.createMap(3)
        line_map.set_obstacles(edge=((0, 0), (0, 1)))
        state = get_state((0, 0))
        result = transition(line_map, state, FORWARD, goal=(2, 2))
        self.assertTrue(result.hit_obstacle)
        self.assertEqual(result.next_state.position, (0, 0))
        self.assertEqual(result.detected_edge, ((0, 0), (0, 1)))

    def test_blocked_node_rolls_back(self):
        line_map = LineMap()
        line_map.createMap(3)
        line_map.set_obstacles(node=(0, 1))
        state = get_state((0, 0))
        result = transition(line_map, state, FORWARD, goal=(2, 2))
        self.assertTrue(result.hit_obstacle)
        self.assertEqual(result.detected_node, (0, 1))
        self.assertEqual(result.next_state.position, (0, 0))

    def test_turn_and_backward(self):
        line_map = LineMap()
        line_map.createMap(3)
        state = get_state((0, 1))
        turned = transition(line_map, state, TURN_LEFT, goal=(2, 2))
        self.assertTrue(turned.turned)
        self.assertEqual(turned.next_state.direction, (-1, 0))

        moved_back = transition(line_map, state, BACKWARD, goal=(2, 2))
        self.assertTrue(moved_back.moved)
        self.assertEqual(moved_back.next_state.position, (0, 0))


class TestStudentRLSmoke(unittest.TestCase):
    def test_first_five_training_maps_train(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        map_dir = os.path.join(root_dir, "maps", "training")
        map_files = sorted(
            os.path.join(map_dir, filename)
            for filename in os.listdir(map_dir)
            if filename.endswith(".json")
        )[:5]
        self.assertEqual(len(map_files), 5)

        for file_path in map_files:
            loaded = load_map_json(file_path)
            result = StudentRL.train(
                loaded.line_map,
                loaded.start,
                loaded.goal,
                loaded.checkpoints,
                episodes=30,
                log_every=10,
                seed=7,
            )
            self.assertTrue(result.qtable)
            self.assertIsNotNone(result.last_result)
            self.assertIsInstance(result.best_result.path, list)
            self.assertEqual(result.best_result.path[0], loaded.start)


if __name__ == "__main__":
    unittest.main()
