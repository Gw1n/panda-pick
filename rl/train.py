import argparse
import os

import yaml
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.envs.pick_place_env import make_env


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_vec_env(env_id: str, seed: int, training: bool = True) -> VecNormalize:
    vec = DummyVecEnv([make_env(env_id, seed=seed)])
    return VecNormalize(vec, norm_obs=True, norm_reward=False, training=training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl/configs/sac_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_timesteps = args.timesteps or cfg["total_timesteps"]

    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_env = build_vec_env(cfg["env_id"], seed=cfg["seed"], training=True)
    eval_env = build_vec_env(cfg["env_id"], seed=cfg["seed"] + 100, training=False)

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

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/checkpoints/",
            name_prefix="sac_her",
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

        run = wandb.init(
            project="manipulation-pnp",
            config=cfg,
            sync_tensorboard=True,
        )
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save("models/policy_final")
    train_env.save("models/vecnormalize.pkl")

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    main()
