"""Main launch file for the come-here system.

Launches all three nodes: audio, perception, behavior.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    audio_params = os.path.join(
        get_package_share_directory('come_here_audio'), 'config', 'audio_params.yaml'
    )
    perception_params = os.path.join(
        get_package_share_directory('come_here_perception'), 'config', 'perception_params.yaml'
    )
    behavior_params = os.path.join(
        get_package_share_directory('come_here_behavior'), 'config', 'behavior_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_mock', default_value='true',
                              description='Use mock sensors (no real hardware)'),

        Node(
            package='come_here_audio',
            executable='audio_node',
            name='audio_node',
            parameters=[audio_params],
            output='screen',
        ),
        Node(
            package='come_here_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[perception_params],
            output='screen',
        ),
        Node(
            package='come_here_perception',
            executable='face_detector_node',
            name='face_detector_node',
            parameters=[perception_params],
            output='screen',
        ),
        Node(
            package='come_here_behavior',
            executable='behavior_node',
            name='behavior_node',
            parameters=[behavior_params],
            output='screen',
        ),
        Node(
            package='come_here_behavior',
            executable='go2_bridge_node',
            name='go2_bridge_node',
            parameters=[behavior_params],
            output='screen',
        ),
    ])
