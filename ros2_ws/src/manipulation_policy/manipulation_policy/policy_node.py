import rclpy
from rclpy.node import Node


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
