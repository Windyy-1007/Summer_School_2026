# Summer School 2026 Robot RL

The project now has one student-facing entrypoint:

```powershell
python .\main.py
```

The PowerShell launcher runs the same file:

```powershell
.\runApp.ps1
```

## Student workflow

1. Edit the learning algorithm in `Algo/StudentRL.py`.
2. Edit reusable MDP state, transition, reward, and map helpers in `rl_library.py`.
3. Run `python .\main.py`.
4. Load a training map, choose episodes, and train the current map or all maps.
5. Run the saved Student RL policy and inspect its path, score, and logs.

`Train All Maps` runs map 1, then map 2, through map N. Training is split into
small UI batches so the current map, greedy path, log, map number, and episode
progress remain visible while training.

When the saved Q-table has no confident action for an unfamiliar state, runtime
navigation switches to a planner fallback on the robot's `known_map`. It still
has to discover real obstacles through movement; each discovery updates
`known_map`, then the hybrid Student RL policy replans toward the next
checkpoint or final goal.

Runtime execution is action-first. The UI executes the policy's next forward,
backward, left-turn, or right-turn action even when the predicted route is only
partial. Repeated state-actions and known collisions trigger local
recovery/exploration, which favors unblocked and less-visited behavior instead
of stopping with a "no complete path" result.

The Q-table learns two state views together. Exact states retain map position,
heading, checkpoint progress, and local geometry for route-specific behavior.
Reusable micro-pattern states describe blocked directions relative to the
robot, relative target direction, previous action, and collision recovery.
This lets behavior learned around one obstacle transfer to a similar situation
at another position or on another map. The supporting state, lookup, reward,
and planner functions remain in `rl_library.py`; `Algo/StudentRL.py` only calls
those helpers from the learning loop.

All 30 curriculum maps use the same 18x12 canvas. They retain the progression
from empty navigation, to sparse obstacles, to checkpoints, and then denser
multi-checkpoint challenges. A generated 18x12 custom challenge is available at
`maps/custom/map_18x12_challenge.json`.

## Main folders

- `Algo/`: A*, baseline Q-learning, and the student RL implementation.
- `app/`: Tkinter UI and the simulation service used by `main.py`.
- `src/`: Core map and robot domain models.
- `maps/`: Custom maps, curriculum maps, and saved Student RL policies.
- `tests/`: Automated tests and the test case sheet.
- `tools/`: Map generation, CLI training, visualization, and algorithm demos.
- `XBot/`: Hardware integration code, kept separate from the simulator.

## UI controls

- `Show A* Reference`: display a reference route for comparison.
- `Run Student RL`: animate the saved Student RL policy.
- `Reset Robot`: return the robot to the map start.
- `Train Current Map`: continue the shared policy on the visible map.
- `Train All Maps`: train sequentially across every curriculum map.
- `Stop RL`: stop after the current UI batch.
- `Reset RL Model`: delete the saved shared policy.
- Map editor controls: set goals, checkpoints, node obstacles, and edge obstacles.

Keyboard controls are `W` forward, `S` backward, `A` turn left, and `D` turn
right.

## Scoring

- Final goal: `+400`
- Each checkpoint: `+200`
- Each move: `-4`
- Each turn: `-1`
- Unfinished route: `-1000 - minimum Manhattan distance`

## Tests

```powershell
python -m unittest discover -s tests -v
```

The test matrix is documented in `tests/TEST_CASES.md`.

## Developer tools

Run these from the repository root:

```powershell
python -m tools.train_student_rl --episodes 500
python -m tools.run_algorithm_demo
python -m tools.visualize_map
python -m tools.generate_training_maps
```
