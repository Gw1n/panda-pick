from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='manipulation_policy',
            executable='sim_bridge',
            name='sim_bridge',
            parameters=[{'meta_path': '/models/policy_meta.json'}],
        ),
        Node(
            package='manipulation_policy',
            executable='policy_node',
            name='policy_node',
            parameters=[{'policy_path': '/models/policy.onnx'}],
        ),
        Node(
            package='manipulation_policy',
            executable='evaluation_node',
            name='evaluation_node',
            parameters=[{
                'n_episodes': 10,
                'output_path': '/results/latest_eval.json',
            }],
        ),
    ])
