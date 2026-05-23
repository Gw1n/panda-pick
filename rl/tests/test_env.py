import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np


def test_env_observation_space():
    env = gym.make("FetchPickAndPlace-v4")
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    env.close()


def test_env_step_returns_is_success():
    env = gym.make("FetchPickAndPlace-v4")
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    _, _, _, _, info = env.step(action)
    assert "is_success" in info
    env.close()


def test_obs_concatenation_order():
    obs_dict = {
        "achieved_goal": np.array([1.0, 2.0, 3.0]),
        "desired_goal": np.array([4.0, 5.0, 6.0]),
        "observation": np.array([7.0, 8.0]),
    }
    keys = ["achieved_goal", "desired_goal", "observation"]
    result = np.concatenate([obs_dict[k].flatten() for k in keys])
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    np.testing.assert_array_equal(result, expected)
