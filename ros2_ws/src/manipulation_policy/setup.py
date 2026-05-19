from setuptools import setup

package_name = 'manipulation_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='RL pick-and-place policy deployed as ROS2 nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = manipulation_policy.policy_node:main',
            'sim_bridge = manipulation_policy.sim_bridge:main',
            'evaluation_node = manipulation_policy.evaluation_node:main',
        ],
    },
)
