import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(log_dir: str) -> list[str]:
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def read_scalar(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    files = find_event_files(log_dir)
    if not files:
        return None

    steps, values = [], []
    for path in files:
        ea = EventAccumulator(path)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for event in ea.Scalars(tag):
            steps.append(event.step)
            values.append(event.value)

    if not steps:
        return None

    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="results/tb_logs")
    parser.add_argument("--output", default="results/training_curve.png")
    args = parser.parse_args()

    data = read_scalar(args.log_dir, "eval/success_rate")
    if data is None:
        print("eval/success_rate not found, trying eval/mean_reward")
        data = read_scalar(args.log_dir, "eval/mean_reward")
        ylabel = "Mean reward"
    else:
        ylabel = "Success rate"

    if data is None:
        raise RuntimeError(f"No scalar data found in {args.log_dir}")

    steps, values = data

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps / 1_000, values, linewidth=1.5, color="#2563eb")
    ax.set_xlabel("Timesteps (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_title("FetchPickAndPlace-v4 — SAC + HER")
    ax.grid(True, alpha=0.3)

    if ylabel == "Success rate":
        ax.axhline(y=0.4, color="#dc2626", linewidth=1, linestyle="--", label="CI threshold (0.4)")
        ax.set_ylim(0, 1)
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
