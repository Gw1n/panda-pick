import rclpy
from rclpy.node import Node


class SimBridge(Node):
    def __init__(self):
        super().__init__('sim_bridge')


def main(args=None):
    rclpy.init(args=args)
    node = SimBridge()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
