# Summer_School_2026

Quick commands to run this repo.

## Latest updates

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
- `Set Goal` lets you click a map node to move the goal.
- `Toggle Sub-goal` lets you click a map node to add or remove a sub-goal.
- `Toggle Node Obstacle` lets you click a node to block or unblock it.
- `Toggle Edge Obstacle` lets you click two adjacent nodes to block or unblock the edge between them.
- `Reset Robot` moves the robot back to the start.

Score rules:

- Final goal: `+400`
- Each sub-goal: `+150`
- Each move: `-3`
- Each turn: `-1`

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
