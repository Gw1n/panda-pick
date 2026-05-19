import rclpy
from rclpy.node import Node


class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
