import gymnasium as gym
import gymnasium_robotics  # registers FetchPickAndPlace-v3 and others on import


def make_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init
