import argparse
import os
import signal
import sys

import yaml
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.envs.pick_place_env import make_env

PAUSE_DIR = "models/paused_state"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_base_env(env_id: str, seed: int) -> DummyVecEnv:
    return DummyVecEnv([make_env(env_id, seed=seed)])


def build_vec_env(env_id: str, seed: int, training: bool = True) -> VecNormalize:
    venv = build_base_env(env_id, seed)
    return VecNormalize(venv, norm_obs=True, norm_reward=False, training=training)


def install_pause_handler(model_ref: list, train_env_ref: list) -> None:
    def handler(signum, frame):
        model = model_ref[0]
        train_env = train_env_ref[0]
        os.makedirs(PAUSE_DIR, exist_ok=True)
        print(f"\nPausing at step {model.num_timesteps}. Saving state...")
        model.save(f"{PAUSE_DIR}/model")
        model.save_replay_buffer(f"{PAUSE_DIR}/replay_buffer")
        train_env.save(f"{PAUSE_DIR}/vecnormalize.pkl")
        print(f"Saved to {PAUSE_DIR}/. Resume with: python rl/train.py --resume")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def resume_from_pause(env_id: str, seed: int) -> tuple[SAC, VecNormalize]:
    venv = build_base_env(env_id, seed)
    train_env = VecNormalize.load(f"{PAUSE_DIR}/vecnormalize.pkl", venv)
    train_env.training = True

    model = SAC.load(f"{PAUSE_DIR}/model", env=train_env)

    replay_path = f"{PAUSE_DIR}/replay_buffer.pkl"
    if os.path.exists(replay_path):
        model.load_replay_buffer(replay_path)
        print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")

    print(f"Resuming from step {model.num_timesteps}")
    return model, train_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl/configs/sac_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_timesteps = args.timesteps or cfg["total_timesteps"]

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.resume:
        model, train_env = resume_from_pause(cfg["env_id"], seed=cfg["seed"])
        remaining = total_timesteps - model.num_timesteps
        reset_num_timesteps = False
    else:
        train_env = build_vec_env(cfg["env_id"], seed=cfg["seed"], training=True)
        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": cfg["her"]["n_sampled_goal"],
                "goal_selection_strategy": cfg["her"]["goal_selection_strategy"],
            },
            learning_rate=cfg["sac"]["learning_rate"],
            buffer_size=cfg["sac"]["buffer_size"],
            batch_size=cfg["sac"]["batch_size"],
            tau=cfg["sac"]["tau"],
            gamma=cfg["sac"]["gamma"],
            train_freq=cfg["sac"]["train_freq"],
            gradient_steps=cfg["sac"]["gradient_steps"],
            learning_starts=cfg["sac"]["learning_starts"],
            tensorboard_log="results/tb_logs",
            seed=cfg["seed"],
            verbose=1,
        )
        remaining = total_timesteps
        reset_num_timesteps = True

    model_ref = [model]
    train_env_ref = [train_env]
    install_pause_handler(model_ref, train_env_ref)

    eval_env = build_vec_env(cfg["env_id"], seed=cfg["seed"] + 100, training=False)

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/checkpoints/",
            name_prefix="sac_her",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            eval_freq=10_000,
            n_eval_episodes=20,
            best_model_save_path="models/",
            log_path="results/",
            deterministic=True,
        ),
    ]

    if not args.no_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(project="manipulation-pnp", config=cfg, sync_tensorboard=True)
        callbacks.append(WandbCallback(verbose=0))

    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save("models/policy_final")
    train_env.save("models/vecnormalize.pkl")

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    main()
