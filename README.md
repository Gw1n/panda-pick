# ROS2 RL Manipulation

Training a reinforcement learning agent to pick and place objects with a Franka Panda arm, then deploying the trained policy as a ROS2 node — with a CI/CD pipeline that automatically evaluates it on every commit.

## Motivation

This is a pet project that sits at the intersection of three things: robot learning, ROS2, and DevOps practices. The goal is not just to train an agent that works, but to build the infrastructure around it — reproducible training, exportable policy, automated regression testing in simulation.

MuJoCo was chosen over Isaac Sim specifically because GitHub Actions runners are CPU-only. Isaac Sim requires an NVIDIA GPU, which makes it unsuitable for headless CI. MuJoCo runs anywhere.

## Stack

| Area | Tools |
|---|---|
| Simulation | MuJoCo via `gymnasium-robotics` (FetchPickAndPlace-v4) |
| RL algorithm | SAC + HER, Stable-Baselines3 |
| Policy export | ONNX |
| Robot framework | ROS2 Humble |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Training monitoring | Weights & Biases |

**Why SAC + HER:** pick-and-place has a sparse reward — the agent only gets feedback when the object actually reaches the goal. Without Hindsight Experience Replay, the agent almost never stumbles on a success by chance, so learning stalls. HER retroactively relabels failed trajectories as successful ones by substituting the achieved state as the goal. In practice this is one line in the SB3 config.

## Current state

The training infrastructure is in place and verified. Both Docker images build. A short test run (2000 steps) completes without errors.

What works:
- `docker compose build` — both `train` and `ros2` images build successfully
- `docker compose run train python rl/train.py --no-wandb` — training runs, checkpoints save to `models/`
- ROS2 package builds with `colcon build` inside the container (stub nodes only for now)

What is not yet implemented:
- `evaluate.py` — headless evaluation with metrics output
- `export_onnx.py` — policy export with VecNormalize baked into the ONNX graph
- ROS2 nodes — `policy_node`, `sim_bridge`, `evaluation_node` are currently stubs
- CI/CD workflows — the YAML files exist but contain only placeholders
- Trained weights — no trained model is committed; training is in progress

## Getting started

Build the training image:

```bash
cd docker
docker compose build train
```

Run a short test (2000 steps, no W&B):

```bash
docker compose run --rm train python rl/train.py --timesteps 2000 --no-wandb
```

Run full training (offline W&B logging):

```bash
docker compose run --rm -e WANDB_MODE=offline train
```

Checkpoints are saved to `models/checkpoints/` every 50k steps. The best model (by eval success rate) is saved to `models/best_model.zip`.

## Repository layout

```
.
├── docker/
│   ├── Dockerfile.train       # RL training environment
│   ├── Dockerfile.ros2        # ROS2 runtime + ONNX inference
│   └── docker-compose.yml
├── rl/
│   ├── train.py               # SAC + HER training entry point
│   ├── evaluate.py            # headless evaluation (not yet implemented)
│   ├── export_onnx.py         # policy export (not yet implemented)
│   ├── envs/
│   │   └── pick_place_env.py  # environment factory
│   └── configs/
│       └── sac_config.yaml    # hyperparameters
├── ros2_ws/
│   └── src/
│       └── manipulation_policy/   # ROS2 package (stubs)
│           ├── policy_node.py
│           ├── sim_bridge.py
│           └── evaluation_node.py
├── models/                    # saved weights (gitignored)
└── results/                   # eval metrics, TensorBoard logs (gitignored)
```

## What's next

**Stage 3 — evaluate and export:**
Write `evaluate.py` to run N headless episodes and output a `metrics.json` with success rate. Write `export_onnx.py` to export the trained policy with VecNormalize normalization baked into the ONNX graph, so the ROS2 node receives raw observations and returns actions without any additional preprocessing.

**Stage 4 — ROS2 nodes:**
Implement the three nodes: `sim_bridge` (MuJoCo ↔ ROS2 topic bridge), `policy_node` (ONNX inference on `/observation`, publishes to `/joint_command`), and `evaluation_node` (orchestrates episodes and reports success rate).

**Stage 5 — CI/CD:**
Wire up the three GitHub Actions workflows: build + test on every push, headless evaluation on merges to main (fails if success rate drops below 0.4), and Docker image publish on version tags.
