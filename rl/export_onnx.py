import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs.pick_place_env import make_env


class PolicyWrapper(nn.Module):
    def __init__(self, actor, obs_mean: np.ndarray, obs_std: np.ndarray, clip_obs: float = 10.0):
        super().__init__()
        self.actor = actor
        self.clip_obs = clip_obs
        self.register_buffer("obs_mean", torch.FloatTensor(obs_mean))
        self.register_buffer("obs_std", torch.FloatTensor(obs_std))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = torch.clamp((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs)
        latent = self.actor.latent_pi(obs_norm)
        mean_actions = self.actor.mu(latent)
        return torch.tanh(mean_actions)


def extract_normalization(vec_normalize: VecNormalize) -> tuple[np.ndarray, np.ndarray, list[str]]:
    obs_rms = vec_normalize.obs_rms
    keys = list(obs_rms.keys())
    means = np.concatenate([obs_rms[k].mean.flatten() for k in keys])
    stds = np.concatenate([np.sqrt(obs_rms[k].var + vec_normalize.epsilon).flatten() for k in keys])
    return means, stds, keys


def verify(wrapper: PolicyWrapper, onnx_path: str, obs_dim: int) -> None:
    dummy = torch.randn(1, obs_dim)
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()

    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {"observation": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_out - ort_out).max())
    print(f"  max output difference (torch vs onnx): {max_diff:.2e}")
    assert max_diff < 1e-5, f"verification failed: max diff {max_diff}"
    print("  verification passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--vecnormalize", default="models/vecnormalize.pkl")
    parser.add_argument("--output", default="models/policy.onnx")
    parser.add_argument("--config", default="rl/configs/sac_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    venv = DummyVecEnv([make_env(cfg["env_id"], seed=0)])
    vec_normalize = VecNormalize.load(args.vecnormalize, venv)

    model = SAC.load(args.model, env=vec_normalize)
    actor = model.policy.actor.eval()

    obs_mean, obs_std, keys = extract_normalization(vec_normalize)
    obs_dim = len(obs_mean)
    print(f"obs keys (concatenation order): {keys}")
    print(f"obs dim: {obs_dim}, action dim: {actor.mu.out_features}")

    wrapper = PolicyWrapper(actor, obs_mean, obs_std).eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        torch.zeros(1, obs_dim),
        args.output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported: {args.output}")

    verify(wrapper, args.output, obs_dim)

    meta = {
        "obs_keys": keys,
        "obs_dim": obs_dim,
        "action_dim": int(actor.mu.out_features),
    }
    meta_path = args.output.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
