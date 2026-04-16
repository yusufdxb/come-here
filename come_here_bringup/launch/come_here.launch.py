"""Main launch file for the come-here system.

Launches audio, perception (person + face detector), and behavior nodes.
The GO2 bridge node is only launched when use_mock:=false, so mock mode
does not require the unitree_api package to be installed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


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

    use_mock = LaunchConfiguration('use_mock')
    mock_override = {
        'use_mock': ParameterValue(use_mock, value_type=bool),
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_mock',
            default_value='true',
            description=(
                'When true, all sensor providers run as mocks and the '
                'GO2 hardware bridge is not launched. Set to false on the robot.'
            ),
        ),

        Node(
            package='come_here_audio',
            executable='audio_node',
            name='audio_node',
            parameters=[audio_params, mock_override],
            output='screen',
        ),
        Node(
            package='come_here_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[perception_params, mock_override],
            output='screen',
        ),
        Node(
            package='come_here_perception',
            executable='face_detector_node',
            name='face_detector_node',
            parameters=[perception_params, mock_override],
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
            condition=UnlessCondition(use_mock),
        ),
    ])
