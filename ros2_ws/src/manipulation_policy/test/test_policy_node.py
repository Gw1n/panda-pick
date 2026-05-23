import numpy as np


def test_obs_reshape_for_inference():
    obs_flat = np.zeros(31, dtype=np.float32)
    batched = obs_flat.reshape(1, -1)
    assert batched.shape == (1, 31)
    assert batched.dtype == np.float32


def test_action_publish_format():
    action = np.array([0.1, -0.2, 0.3, 1.0], dtype=np.float32)
    clipped = np.clip(action, -1.0, 1.0)
    as_list = clipped.tolist()
    assert isinstance(as_list, list)
    assert len(as_list) == 4
