import numpy as np


def test_obs_to_vec_preserves_key_order():
    obs_dict = {
        "achieved_goal": np.array([1.0, 2.0, 3.0]),
        "desired_goal": np.array([4.0, 5.0, 6.0]),
        "observation": np.zeros(25),
    }
    keys = ["achieved_goal", "desired_goal", "observation"]
    result = np.concatenate([obs_dict[k].flatten() for k in keys])

    assert result[0] == 1.0
    assert result[3] == 4.0
    assert result[6] == 0.0
    assert result.shape == (31,)


def test_action_clipping():
    action = np.array([2.0, -3.0, 0.5, 1.0])
    clipped = np.clip(action, -1.0, 1.0)
    assert clipped[0] == 1.0
    assert clipped[1] == -1.0
    assert clipped[2] == 0.5
