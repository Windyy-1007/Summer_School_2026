# Summer_School_2026

Quick commands to run this repo.

## Requirements

- Python 3
- Tkinter, usually included with Python

## Run the app

From the repo root:

```powershell
.\runApp.ps1
```

Or run the Python entrypoint directly:

```powershell
python .\app\keyboardControl.py
```

In the app:

- `Show A* Path` draws an A* route from the robot to the goal.
- `Show Q-learning Path` draws a Q-learning route from the start to the goal.
- `Run Two-map A*` animates the robot using `known_map` and replans as obstacles are discovered.
- `Run Two-map Q` animates Q-learning using `known_map` and replans as obstacles are discovered.
- `Save Map JSON` saves the current map, goal, checkpoints, and obstacles.
- `Load Map JSON` loads a saved map file.
- `Load Training Map` loads one of the 30 sample maps in `maps/training`.
- `Train Student RL` trains the educational MDP/QTable scaffold and shows milestone paths.
- `Set Goal` lets you click a map node to move the goal.
- `Toggle Sub-goal` lets you click a map node to add or remove a sub-goal.
- `Toggle Node Obstacle` lets you click a node to block or unblock it.
- `Toggle Edge Obstacle` lets you click two adjacent nodes to block or unblock the edge between them.
- `Reset Robot` moves the robot back to the start.

Score rules:

- Checkpoint: `+200`
- Final goal: `+400`
- Each full node-to-node move: `-4`
- Each turn: `-1`
- If the robot reaches the goal: `movePenalty + 200 * checkpointsReached + 400`
- If the robot does not reach the goal: `-1000 - minManhattan`, where `minManhattan` is the closest distance the robot reached to the goal.

## Student RL scaffold

The student-facing helper library is `rl_library.py`. It contains constants, simple actions, state helpers, transition logic, scoring, and JSON map helpers.

The educational QTable trainer is `Algo/StudentRL.py`. Its default reward is intentionally simple:

```python
reward = -manhattan(next_position, goal)
```

Students are expected to improve sensor handling, reward design, and training map data from there.

Train from the console:

```powershell
python .\train_student_rl.py --map .\maps\training\map_01_empty_grid.json --episodes 500
```

Or train all sample maps:

```powershell
python .\train_student_rl.py --episodes 300 --log-every 50
```

## Run A* and Q-learning simulation

```powershell
python .\simulation.py
```

This prints:

- A* path on the full map
- Q-learning path on the full map
- Q-table greedy policy for the learned path
- Two-map robot navigation using A* on `known_map`

## API examples

See `API.py` for sample robot, ultrasonic, line sensor, button, timer, and motion API calls.
