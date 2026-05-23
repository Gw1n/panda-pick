import argparse
import json
import os

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import imageio
import numpy as np
import onnxruntime as ort


def run_episode(session, env, obs_keys: list[str], seed: int) -> tuple[list, bool]:
    frames = []
    obs, _ = env.reset(seed=seed)
    done = False
    success = False

    while not done:
        frames.append(env.render())

        obs_vec = np.concatenate(
            [obs[k].flatten() for k in obs_keys]
        ).astype(np.float32).reshape(1, -1)

        action = session.run(None, {"observation": obs_vec})[0][0]
        action = np.clip(action, -1.0, 1.0)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        success = bool(info.get("is_success", False))

    frames.append(env.render())
    return frames, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="models/policy.onnx")
    parser.add_argument("--output", default="results/demo.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--tries", type=int, default=20)
    args = parser.parse_args()

    meta_path = args.policy.replace(".onnx", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    session = ort.InferenceSession(args.policy)
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")

    saved_frames = None
    for attempt in range(args.tries):
        frames, success = run_episode(session, env, meta["obs_keys"], seed=args.seed + attempt)
        if saved_frames is None or success:
            saved_frames = frames
        if success:
            print(f"successful episode at seed={args.seed + attempt}")
            break
    else:
        print(f"no success in {args.tries} tries, saving last attempt")

    env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    imageio.mimsave(args.output, saved_frames, fps=args.fps, loop=0)
    print(f"saved {len(saved_frames)} frames → {args.output}")


if __name__ == "__main__":
    main()
