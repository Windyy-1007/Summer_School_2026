import os
import tempfile
import unittest

from Algo import StudentRL
from app.services import SimulationService
from rl_library import (
    ACTIONS,
    BACKWARD,
    FRONT_BLOCKED,
    FORWARD,
    LEFT_BLOCKED,
    MICRO_PATTERN_CONTEXT,
    RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    action_visit_key,
    calculate_score,
    find_action_values,
    get_state,
    is_terminal_state,
    load_map_from_dict,
    load_map_json,
    micro_pattern_state,
    save_map_json,
    transition,
    update_q_values,
)
from src.map import LineMap


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
    def test_curriculum_and_custom_challenge_are_18x12(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        training_dir = os.path.join(root_dir, "maps", "training")
        map_files = sorted(
            os.path.join(training_dir, filename)
            for filename in os.listdir(training_dir)
            if filename.endswith(".json")
        )
        map_files.append(
            os.path.join(
                root_dir,
                "maps",
                "custom",
                "map_18x12_challenge.json",
            )
        )

        self.assertEqual(len(map_files), 31)
        for file_path in map_files:
            loaded = load_map_json(file_path)
            self.assertEqual(
                (loaded.width, loaded.height),
                (18, 12),
                os.path.basename(file_path),
            )

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
    def test_obstacle_pattern_is_relative_to_heading(self):
        line_map = LineMap()
        line_map.createMap(5)
        line_map.set_obstacles(node=(2, 3))

        facing_up = get_state(
            (2, 2),
            map_obj=line_map,
            goal=(4, 4),
        )
        facing_right = get_state(
            (2, 2),
            direction=RIGHT,
            map_obj=line_map,
            goal=(4, 4),
        )

        self.assertTrue(facing_up.obstacle_mask & FRONT_BLOCKED)
        self.assertFalse(facing_right.obstacle_mask & FRONT_BLOCKED)
        self.assertTrue(facing_right.obstacle_mask & LEFT_BLOCKED)

    def test_target_direction_is_relative_to_heading(self):
        line_map = LineMap()
        line_map.createMap(5)

        facing_up = get_state(
            (2, 2),
            map_obj=line_map,
            goal=(3, 2),
        )
        facing_right = get_state(
            (2, 2),
            direction=RIGHT,
            map_obj=line_map,
            goal=(3, 2),
        )

        self.assertEqual(
            (facing_up.target_forward, facing_up.target_right),
            (0, 1),
        )
        self.assertEqual(
            (facing_right.target_forward, facing_right.target_right),
            (1, 0),
        )

    def test_micro_pattern_q_values_transfer_between_positions(self):
        line_map = LineMap()
        line_map.createMap(7)
        state_a = get_state(
            (2, 2),
            map_obj=line_map,
            goal=(2, 5),
            context="map_a",
        )
        state_b = get_state(
            (4, 2),
            map_obj=line_map,
            goal=(4, 5),
            context="map_b",
        )
        qtable = {}

        update_q_values(
            qtable,
            state_a,
            FORWARD,
            target=10.0,
            alpha=1.0,
        )

        self.assertNotEqual(state_a, state_b)
        self.assertEqual(
            micro_pattern_state(state_a),
            micro_pattern_state(state_b),
        )
        self.assertAlmostEqual(
            find_action_values(qtable, state_b)[FORWARD],
            3.5,
        )

    def test_recovery_state_rewards_escape_and_penalizes_repeated_collision(self):
        line_map = LineMap()
        line_map.createMap(4)
        line_map.set_obstacles(edge=((1, 1), (1, 2)))
        state = get_state(
            (1, 1),
            map_obj=line_map,
            goal=(3, 3),
        )

        blocked = transition(line_map, state, FORWARD, goal=(3, 3))
        repeated = transition(
            line_map,
            blocked.next_state,
            FORWARD,
            goal=(3, 3),
        )
        turned = transition(
            line_map,
            blocked.next_state,
            TURN_RIGHT,
            goal=(3, 3),
        )
        escaped = transition(
            line_map,
            turned.next_state,
            FORWARD,
            goal=(3, 3),
        )

        self.assertTrue(blocked.next_state.recovery_mode)
        self.assertEqual(blocked.next_state.last_action, FORWARD)
        self.assertEqual(repeated.reward, -33)
        self.assertTrue(turned.next_state.recovery_mode)
        self.assertTrue(escaped.moved)
        self.assertFalse(escaped.next_state.recovery_mode)
        self.assertGreater(escaped.reward, 0)

    def test_forward_clear_move(self):
        line_map = LineMap()
        line_map.createMap(3)
        state = get_state((0, 0))
        result = transition(line_map, state, FORWARD, goal=(2, 2))
        self.assertTrue(result.moved)
        self.assertEqual(result.next_state.position, (0, 1))
        self.assertEqual(result.reward, 1)

    def test_blocked_edge_stays_in_place(self):
        line_map = LineMap()
        line_map.createMap(3)
        line_map.set_obstacles(edge=((0, 0), (0, 1)))
        state = get_state((0, 0))
        result = transition(line_map, state, FORWARD, goal=(2, 2))
        self.assertTrue(result.hit_obstacle)
        self.assertEqual(result.next_state.position, (0, 0))
        self.assertEqual(result.detected_edge, ((0, 0), (0, 1)))
        self.assertEqual(result.reward, -25)

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

    def test_goal_is_terminal_only_after_all_checkpoints(self):
        line_map = LineMap()
        line_map.createMap(3)
        checkpoints = [(2, 2)]
        state = get_state((0, 0), checkpoints=checkpoints)
        result = transition(
            line_map,
            state,
            FORWARD,
            goal=(0, 1),
            checkpoints=checkpoints,
        )
        self.assertTrue(result.reached_goal)
        self.assertFalse(
            is_terminal_state(result.next_state, (0, 1), checkpoints)
        )


class TestStudentRLSmoke(unittest.TestCase):
    def test_partial_policy_still_returns_the_next_action(self):
        line_map = LineMap()
        line_map.createMap(5)

        policy = StudentRL.simulate_policy(
            {},
            line_map,
            start=(0, 0),
            goal=(4, 4),
            max_steps=1,
            planner_fallback=True,
        )

        self.assertFalse(policy.reached_goal)
        self.assertEqual(policy.actions, [FORWARD])
        self.assertEqual(policy.path, [(0, 0), (0, 1)])

    def test_runtime_ignores_learned_action_that_hits_known_obstacle(self):
        line_map = LineMap()
        line_map.createMap(5)
        line_map.set_obstacles(node=(0, 1))
        state = get_state(
            (0, 0),
            map_obj=line_map,
            goal=(0, 4),
        )
        qtable = {
            state: {
                action: 100.0 if action == FORWARD else 0.0
                for action in ACTIONS
            }
        }

        policy = StudentRL.simulate_policy(
            qtable,
            line_map,
            start=(0, 0),
            goal=(0, 4),
            max_steps=1,
            planner_fallback=True,
        )

        self.assertNotEqual(policy.actions[0], FORWARD)
        self.assertEqual(policy.action_sources[0], "recovery")

    def test_runtime_changes_behavior_after_repeating_state_action(self):
        line_map = LineMap()
        line_map.createMap(5)
        state = get_state(
            (2, 2),
            map_obj=line_map,
            goal=(2, 4),
        )
        qtable = {
            state: {
                action: 100.0 if action == FORWARD else 0.0
                for action in ACTIONS
            }
        }

        policy = StudentRL.simulate_policy(
            qtable,
            line_map,
            start=(2, 2),
            goal=(2, 4),
            max_steps=1,
            planner_fallback=True,
            action_counts={action_visit_key(state, FORWARD): 1},
        )

        self.assertNotEqual(policy.actions[0], FORWARD)
        self.assertEqual(policy.action_sources[0], "recovery")

    def test_recovery_backs_out_when_no_complete_route_is_known(self):
        line_map = LineMap()
        line_map.createMap(3)
        line_map.set_obstacles(node=(1, 2))
        line_map.set_obstacles(node=(0, 1))
        line_map.set_obstacles(node=(2, 1))

        policy = StudentRL.simulate_policy(
            {},
            line_map,
            start=(1, 1),
            goal=(1, 2),
            max_steps=1,
            planner_fallback=True,
        )

        self.assertFalse(policy.reached_goal)
        self.assertEqual(policy.actions, [BACKWARD])
        self.assertEqual(policy.path, [(1, 1), (1, 0)])
        self.assertEqual(policy.action_sources, ["recovery"])

    def test_unfamiliar_policy_uses_planner_fallback_around_obstacle(self):
        line_map = LineMap()
        line_map.createMap(5)
        line_map.set_obstacles(node=(0, 1))

        without_fallback = StudentRL.simulate_policy(
            {},
            line_map,
            start=(0, 0),
            goal=(0, 4),
            context="unfamiliar_map",
        )
        with_fallback = StudentRL.simulate_policy(
            {},
            line_map,
            start=(0, 0),
            goal=(0, 4),
            context="unfamiliar_map",
            planner_fallback=True,
        )

        self.assertFalse(without_fallback.reached_goal)
        self.assertTrue(with_fallback.reached_goal)
        self.assertNotIn((0, 1), with_fallback.path)
        self.assertGreater(with_fallback.planner_fallback_steps, 0)

    def test_unfamiliar_policy_targets_checkpoint_before_goal(self):
        line_map = LineMap()
        line_map.createMap(5)
        checkpoint = (4, 0)
        goal = (4, 4)

        policy = StudentRL.simulate_policy(
            {},
            line_map,
            start=(0, 0),
            goal=goal,
            checkpoints=[checkpoint],
            context="unfamiliar_checkpoint_map",
            planner_fallback=True,
        )

        self.assertTrue(policy.reached_goal)
        self.assertLess(
            policy.path.index(checkpoint),
            policy.path.index(goal),
        )

    def test_first_five_training_maps_learn_complete_policy(self):
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
                episodes=500,
                log_every=100,
                seed=7,
            )
            policy = StudentRL.simulate_policy(
                result.qtable,
                loaded.line_map,
                loaded.start,
                loaded.goal,
                loaded.checkpoints,
            )
            self.assertTrue(result.qtable)
            self.assertIsNotNone(result.last_result)
            self.assertIsInstance(result.best_result.path, list)
            self.assertEqual(result.best_result.path[0], loaded.start)
            self.assertTrue(policy.reached_goal)
            self.assertEqual(
                policy.checkpoints_reached,
                len(loaded.checkpoints),
            )

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
        self.assertEqual(set(loaded_qtable), set(result.qtable))
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

    def test_train_all_batches_advance_maps_and_report_progress(self):
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
                    name=f"batch_map_{index + 1}",
                )

            service = SimulationService(map_size=5)
            service.training_map_dir = training_dir
            service.student_rl_policy_dir = policy_dir
            message = service.start_student_rl_all_training_maps(
                episodes=10,
                batch_size=2,
                seed=5,
            )
            self.assertIn("Training map 1/2", message)
            self.assertEqual(
                service.get_student_rl_training_progress(),
                "Map 1/2 | Episode 0/10",
            )

            visited_maps = {service.get_map_name()}
            done = False
            while not done:
                done, message, _ = service.train_student_rl_batch()
                visited_maps.add(service.get_map_name())

            qtable, policy_data = StudentRL.load_policy_json(
                os.path.join(policy_dir, "student_rl_policy.json")
            )

        self.assertEqual(visited_maps, {"batch_map_1", "batch_map_2"})
        self.assertIn("Greedy policy reached goal on", message)
        self.assertFalse(service.is_student_rl_training_active())
        self.assertEqual(len(policy_data["metadata"]["evaluation"]), 2)
        self.assertEqual(
            len(
                {
                    state.context
                    for state in qtable
                    if state.context
                    and state.context != MICRO_PATTERN_CONTEXT
                }
            ),
            2,
        )
        self.assertTrue(
            any(
                state.context == MICRO_PATTERN_CONTEXT
                for state in qtable
            )
        )

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

    def test_run_student_rl_replans_after_unfamiliar_obstacle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(map_size=5)
            service.student_rl_policy_dir = tmpdir
            service.map = LineMap()
            service.map.createMap(5)
            service.map.set_obstacles(node=(0, 1))
            service.map_width = 5
            service.map_height = 5
            service.map_size = 5
            service.map_name = "Unfamiliar Obstacle Map"
            service.current_map_file_path = None
            service.start = (0, 0)
            service.goal = (0, 4)
            service.sub_goals = []
            service.reset_robot()

            StudentRL.save_policy_json(
                file_path=os.path.join(tmpdir, "student_rl_policy.json"),
                qtable={},
                episodes=100,
                map_name="student_rl_shared_policy",
            )

            messages = []
            status = None
            for _ in range(20):
                status, message, _ = (
                    service.step_toward_goal_with_known_student_rl()
                )
                messages.append(message)
                if status is not None:
                    break

        self.assertTrue(status)
        self.assertEqual(
            (service.get_robot_x(), service.get_robot_y()),
            (0, 4),
        )
        self.assertIn((0, 1), service.robot.discovered_nodes)
        self.assertTrue(
            any("replanning" in message for message in messages)
        )
        self.assertTrue(
            any("Hybrid Student RL" in message for message in messages)
        )

    def test_run_student_rl_keeps_acting_without_a_complete_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = SimulationService(map_size=5)
            service.student_rl_policy_dir = tmpdir
            service.map = LineMap()
            service.map.createMap(3)
            service.map.set_obstacles(node=(1, 2))
            service.map.set_obstacles(node=(0, 1))
            service.map.set_obstacles(node=(2, 1))
            service.map_width = 3
            service.map_height = 3
            service.map_size = 3
            service.start = (1, 1)
            service.goal = (1, 2)
            service.sub_goals = []
            service.reset_robot()

            StudentRL.save_policy_json(
                file_path=os.path.join(tmpdir, "student_rl_policy.json"),
                qtable={},
                episodes=1,
                map_name="student_rl_shared_policy",
            )

            first_status, first_message, _ = (
                service.step_toward_goal_with_known_student_rl()
            )
            second_status, second_message, _ = (
                service.step_toward_goal_with_known_student_rl()
            )

        self.assertIsNone(first_status)
        self.assertIn("replanning", first_message)
        self.assertIsNone(second_status)
        self.assertIn("recovery/exploration", second_message)
        self.assertEqual(
            (service.get_robot_x(), service.get_robot_y()),
            (1, 0),
        )


if __name__ == "__main__":
    unittest.main()
