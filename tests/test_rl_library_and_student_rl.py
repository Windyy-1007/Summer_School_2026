import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from Algo import StudentRL
from app.services import SimulationService
from rl_library import (
    ACTIONS,
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

    def test_policy_json_round_trip(self):
        line_map = LineMap()
        line_map.createMap(4)
        checkpoints = [(0, 2)]
        result = StudentRL.train(
            line_map,
            start=(0, 0),
            goal=(3, 3),
            checkpoints=checkpoints,
            episodes=60,
            log_every=60,
            seed=11,
        )
        policy = StudentRL.simulate_policy(
            result.qtable,
            line_map,
            start=(0, 0),
            goal=(3, 3),
            checkpoints=checkpoints,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "student_policy.json")
            saved = StudentRL.save_policy_json(
                file_path=file_path,
                qtable=result.qtable,
                episodes=result.episodes,
                map_name="policy_round_trip",
                map_file="maps/training/policy_round_trip.json",
                start=(0, 0),
                goal=(3, 3),
                checkpoints=checkpoints,
                best_result=result.best_result,
                last_result=result.last_result,
                policy_result=policy,
            )
            loaded_qtable, loaded = StudentRL.load_policy_json(file_path)

        self.assertEqual(saved["algorithm"], "student_q_learning")
        self.assertEqual(loaded["map_name"], "policy_round_trip")
        self.assertEqual(len(loaded_qtable), len(result.qtable))
        loaded_policy = StudentRL.simulate_policy(
            loaded_qtable,
            line_map,
            start=(0, 0),
            goal=(3, 3),
            checkpoints=checkpoints,
        )
        self.assertEqual(loaded_policy.path, policy.path)

    def test_service_train_all_training_maps_saves_one_shared_policy_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_dir = os.path.join(tmpdir, "training")
            policy_dir = os.path.join(tmpdir, "policies")
            os.makedirs(training_dir)

            for index in range(2):
                line_map = LineMap()
                line_map.createMap(3)
                save_map_json(
                    os.path.join(training_dir, f"map_{index + 1}.json"),
                    line_map,
                    start=(0, 0),
                    goal=(2, 2),
                    checkpoints=[],
                    name=f"temp_map_{index + 1}",
                )

            service = SimulationService(map_size=5)
            service.training_map_dir = training_dir
            service.student_rl_policy_dir = policy_dir

            message = service.train_student_rl_all_training_maps(episodes=5, seed=3)
            policy_files = sorted(
                filename
                for filename in os.listdir(policy_dir)
                if filename.endswith(".json")
            )
            qtable, policy_data = StudentRL.load_policy_json(
                os.path.join(policy_dir, policy_files[0])
            )

        self.assertIn("Trained one shared Student RL model on 2 maps", message)
        self.assertEqual(policy_files, ["student_rl_policy.json"])
        self.assertTrue(qtable)
        self.assertEqual(policy_data["map_name"], "student_rl_shared_policy")
        self.assertEqual(policy_data["episodes"], 10)
        self.assertEqual(len(policy_data["metadata"]["training_maps"]), 2)

    def test_current_map_training_continues_single_shared_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(map_size=5)
            service.student_rl_policy_dir = tmpdir

            service.start_student_rl_training(episodes=5, batch_size=5, seed=9)
            done, _, _ = service.train_student_rl_batch()
            self.assertTrue(done)

            policy_path = os.path.join(tmpdir, "student_rl_policy.json")
            first_qtable, first_policy = StudentRL.load_policy_json(policy_path)

            service.start_student_rl_training(episodes=7, batch_size=7, seed=10)
            done, _, _ = service.train_student_rl_batch()
            self.assertTrue(done)

            policy_files = sorted(
                filename
                for filename in os.listdir(tmpdir)
                if filename.endswith(".json")
            )
            second_qtable, second_policy = StudentRL.load_policy_json(policy_path)

        self.assertEqual(policy_files, ["student_rl_policy.json"])
        self.assertTrue(first_qtable)
        self.assertTrue(second_qtable)
        self.assertEqual(first_policy["episodes"], 5)
        self.assertEqual(second_policy["episodes"], 12)

    def test_run_student_rl_requires_saved_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(map_size=5)
            service.student_rl_policy_dir = tmpdir

            status, message, path = service.step_toward_goal_with_known_student_rl()

        self.assertFalse(status)
        self.assertIn("No saved Student RL policy", message)
        self.assertIsNone(path)

    def test_run_student_rl_uses_saved_policy_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(map_size=5)
            service.student_rl_policy_dir = tmpdir
            service.map = LineMap()
            service.map.createMap(3)
            service.map_width = 3
            service.map_height = 3
            service.map_size = 3
            service.start = (0, 0)
            service.goal = (0, 1)
            service.sub_goals = []
            service.reset_robot()

            state = get_state((0, 0))
            qtable = {state: {action: 0.0 for action in ACTIONS}}
            qtable[state][FORWARD] = 100.0
            StudentRL.save_policy_json(
                file_path=os.path.join(tmpdir, "student_rl_policy.json"),
                qtable=qtable,
                episodes=123,
                map_name="student_rl_shared_policy",
                start=None,
                goal=None,
                checkpoints=[],
            )

            status, message, path = service.step_toward_goal_with_known_student_rl()

        self.assertTrue(status)
        self.assertIn("Saved Student RL policy moved", message)
        self.assertEqual(path, [(0, 0), (0, 1)])


if __name__ == "__main__":
    unittest.main()
