import json
import threading
import time

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String


class SimBridge(Node):
    def __init__(self):
        super().__init__('sim_bridge')

        self.declare_parameter('meta_path', '/models/policy_meta.json')
        meta_path = self.get_parameter('meta_path').get_parameter_value().string_value

        with open(meta_path) as f:
            self._obs_keys = json.load(f)['obs_keys']

        self._env = gym.make('FetchPickAndPlace-v4')
        obs, _ = self._env.reset()
        self._current_obs = obs

        self._action: np.ndarray | None = None
        self._action_event = threading.Event()
        self._episode_running = False
        self._step_reward = 0.0
        self._step_count = 0

        self._obs_pub = self.create_publisher(Float32MultiArray, '/observation', 10)
        self._done_pub = self.create_publisher(String, '/episode_done', 10)
        self.create_subscription(Float32MultiArray, '/joint_command', self._action_cb, 10)
        self.create_subscription(Bool, '/episode_reset', self._reset_cb, 10)

        threading.Thread(target=self._sim_loop, daemon=True).start()
        self.get_logger().info('sim_bridge ready')

    def _action_cb(self, msg: Float32MultiArray) -> None:
        self._action = np.array(msg.data, dtype=np.float32)
        self._action_event.set()

    def _reset_cb(self, _msg: Bool) -> None:
        obs, _ = self._env.reset()
        self._current_obs = obs
        self._step_reward = 0.0
        self._step_count = 0
        self._action = None
        self._action_event.clear()
        self._episode_running = True

    def _obs_to_vec(self, obs_dict: dict) -> np.ndarray:
        return np.concatenate(
            [obs_dict[k].flatten() for k in self._obs_keys]
        ).astype(np.float32)

    def _sim_loop(self) -> None:
        while rclpy.ok():
            if not self._episode_running:
                time.sleep(0.01)
                continue

            msg = Float32MultiArray()
            msg.data = self._obs_to_vec(self._current_obs).tolist()
            self._obs_pub.publish(msg)

            if not self._action_event.wait(timeout=1.0):
                self.get_logger().warn('timeout waiting for action')
                continue
            self._action_event.clear()

            obs, reward, terminated, truncated, info = self._env.step(self._action)
            self._current_obs = obs
            self._step_reward += float(reward)
            self._step_count += 1

            if terminated or truncated:
                self._episode_running = False
                result = String()
                result.data = json.dumps({
                    'success': bool(info.get('is_success', False)),
                    'steps': self._step_count,
                    'reward': self._step_reward,
                })
                self._done_pub.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = SimBridge()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
