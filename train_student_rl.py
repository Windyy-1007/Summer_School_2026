import argparse
import glob
import os

from Algo.StudentRL import format_episode_log, simulate_policy, train
from rl_library import load_map_json


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_MAP_DIR = os.path.join(ROOT_DIR, "maps", "training")


def _default_maps():
    return sorted(glob.glob(os.path.join(TRAINING_MAP_DIR, "*.json")))


def _print_map_header(map_data, file_path):
    print("\n" + "=" * 72)
    print(f"Map: {map_data.name}")
    print(f"File: {os.path.relpath(file_path, ROOT_DIR)}")
    print(f"Size: {map_data.width}x{map_data.height}")
    print(f"Start: {map_data.start} | Goal: {map_data.goal} | Checkpoints: {map_data.checkpoints}")
    print("=" * 72)


def train_one_map(file_path, episodes, log_every, seed):
    map_data = load_map_json(file_path)
    _print_map_header(map_data, file_path)

    result = train(
        map_obj=map_data.line_map,
        start=map_data.start,
        goal=map_data.goal,
        checkpoints=map_data.checkpoints,
        episodes=episodes,
        log_every=log_every,
        seed=seed,
    )

    for line in result.logs:
        print(line)

    policy = simulate_policy(
        result.qtable,
        map_data.line_map,
        map_data.start,
        map_data.goal,
        map_data.checkpoints,
    )
    print("\nGreedy policy after training:")
    print(format_episode_log(policy, total_checkpoints=len(map_data.checkpoints)))
    print("Path:", " -> ".join(str(coord) for coord in policy.path))
    return result


def main():
    parser = argparse.ArgumentParser(description="Train the student RL QTable on JSON maps.")
    parser.add_argument("--map", dest="map_path", help="Path to one JSON map. Defaults to all training maps.")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per map.")
    parser.add_argument("--log-every", type=int, default=50, help="Print one milestone every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable training.")
    args = parser.parse_args()

    map_files = [args.map_path] if args.map_path else _default_maps()
    if not map_files:
        raise SystemExit("No training maps found. Run tools/generate_training_maps.py first.")

    for file_path in map_files:
        train_one_map(file_path, args.episodes, args.log_every, args.seed)


if __name__ == "__main__":
    main()
