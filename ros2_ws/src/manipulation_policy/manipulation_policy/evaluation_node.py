import json
import os
from datetime import datetime, timezone

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')

        self.declare_parameter('n_episodes', 10)
        self.declare_parameter('output_path', '/results/latest_eval.json')

        self._n_episodes = self.get_parameter('n_episodes').get_parameter_value().integer_value
        self._output_path = self.get_parameter('output_path').get_parameter_value().string_value

        self._results: list[dict] = []

        self._reset_pub = self.create_publisher(Bool, '/episode_reset', 10)
        self.create_subscription(String, '/episode_done', self._done_cb, 10)

        self._startup_timer = self.create_timer(2.0, self._start)

    def _start(self) -> None:
        self._startup_timer.cancel()
        self._send_reset()

    def _send_reset(self) -> None:
        msg = Bool()
        msg.data = True
        self._reset_pub.publish(msg)
        self.get_logger().info(f'episode {len(self._results) + 1}/{self._n_episodes}')

    def _done_cb(self, msg: String) -> None:
        result = json.loads(msg.data)
        self._results.append(result)
        self.get_logger().info(
            f'  success={result["success"]}  steps={result["steps"]}  '
            f'reward={result["reward"]:.1f}'
        )

        if len(self._results) >= self._n_episodes:
            self._save_and_exit()
        else:
            self._send_reset()

    def _save_and_exit(self) -> None:
        successes = [r['success'] for r in self._results]
        metrics = {
            'success_rate': float(np.mean(successes)),
            'mean_episode_length': float(np.mean([r['steps'] for r in self._results])),
            'mean_reward': float(np.mean([r['reward'] for r in self._results])),
            'n_episodes': self._n_episodes,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(self._output_path) or '.', exist_ok=True)
        with open(self._output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(
            f'success_rate={metrics["success_rate"]:.3f} '
            f'({sum(successes)}/{self._n_episodes})'
        )
        self.get_logger().info(f'written: {self._output_path}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
