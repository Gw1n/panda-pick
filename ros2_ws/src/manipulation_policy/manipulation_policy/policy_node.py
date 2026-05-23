import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        self.declare_parameter('policy_path', '/models/policy.onnx')
        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value

        self._session = ort.InferenceSession(policy_path)
        self._action_pub = self.create_publisher(Float32MultiArray, '/joint_command', 10)
        self.create_subscription(Float32MultiArray, '/observation', self._obs_cb, 10)
        self.get_logger().info(f'loaded policy: {policy_path}')

    def _obs_cb(self, msg: Float32MultiArray) -> None:
        obs = np.array(msg.data, dtype=np.float32).reshape(1, -1)
        action = self._session.run(None, {'observation': obs})[0][0]
        action = np.clip(action, -1.0, 1.0)

        out = Float32MultiArray()
        out.data = action.tolist()
        self._action_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
