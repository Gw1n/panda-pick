import argparse
import json
import os
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
import onnxruntime as ort

import gymnasium_robotics  # noqa: F401


def obs_to_vector(obs_dict: dict, keys: list[str]) -> np.ndarray:
    return np.concatenate([obs_dict[k].flatten() for k in keys]).astype(np.float32)


def run_episodes(session, env, obs_keys: list[str], n_episodes: int, seed: int) -> dict:
    successes = []
    episode_lengths = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            obs_vec = obs_to_vector(obs, obs_keys).reshape(1, -1)
            action = session.run(None, {"observation": obs_vec})[0][0]
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        successes.append(float(info.get("is_success", 0.0)))
        episode_lengths.append(ep_length)
        episode_rewards.append(ep_reward)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "n_episodes": n_episodes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="models/policy.onnx")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/latest_eval.json")
    parser.add_argument("--success-threshold", type=float, default=None)
    args = parser.parse_args()

    meta_path = args.policy.replace(".onnx", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    session = ort.InferenceSession(args.policy)
    env = gym.make("FetchPickAndPlace-v4")

    print(f"Evaluating {args.episodes} episodes (seed={args.seed})...")
    metrics = run_episodes(session, env, meta["obs_keys"], args.episodes, args.seed)
    env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"success_rate:        {metrics['success_rate']:.3f}")
    print(f"mean_episode_length: {metrics['mean_episode_length']:.1f}")
    print(f"mean_reward:         {metrics['mean_reward']:.2f}")
    print(f"written: {args.output}")

    if args.success_threshold is not None and metrics["success_rate"] < args.success_threshold:
        print(f"REGRESSION: {metrics['success_rate']:.3f} < threshold {args.success_threshold}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
