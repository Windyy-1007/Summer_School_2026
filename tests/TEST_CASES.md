# Test Cases

Run all automated cases from the repository root:

```powershell
python -m unittest discover -s tests -v
```

| ID | Area | Expected result |
| --- | --- | --- |
| MAP-01 | Create square and rectangular maps | Correct nodes and directed edges are created |
| MAP-02 | Save and load JSON maps | Start, goal, checkpoints, nodes, and edges round-trip |
| ROBOT-01 | Move and turn | Position and heading update correctly |
| ROBOT-02 | Discover obstacles | Blocked nodes and edges are copied to `known_map` |
| PLAN-01 | A* navigation | A valid shortest route is returned when reachable |
| PLAN-02 | Baseline Q-learning | A valid route is learned around obstacles |
| RL-01 | MDP transition | Actions produce correct state, sensor, and obstacle results |
| RL-02 | Terminal rule | Goal is terminal only after every checkpoint is reached |
| RL-03 | Student RL training | First five curriculum maps reach goal and all checkpoints |
| RL-04 | Policy persistence | Q-table and metadata save/load without changing behavior |
| RL-05 | Continue training | Existing policy episode count and Q-table are reused |
| RL-06 | Train all maps | Maps run sequentially in batches with visible progress |
| RL-07 | Shared policy isolation | Per-map state contexts coexist in one policy JSON file |
| RL-08 | Unfamiliar obstacle map | Agent discovers the obstacle and uses planner fallback to replan |
| RL-09 | Relative obstacle state | Front/right/back/left obstacle patterns rotate with the agent heading |
| RL-10 | Micro-pattern transfer | Equivalent local situations reuse learned Q-values across positions and maps |
| RL-11 | Collision recovery | Repeated collisions are penalized and successful escape behavior is rewarded |
| RL-12 | Action-first runtime | A partial policy can move, turn, or reverse without requiring a complete predicted path |
| RL-13 | Loop recovery | Repeating the same state-action switches to a less-visited behavior |
| UI-01 | Non-blocking training contract | Each batch returns progress, path, and completion state |
