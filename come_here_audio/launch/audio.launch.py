import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('come_here_audio')
    default_params = os.path.join(pkg_dir, 'config', 'audio_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('use_mock', default_value='true'),
        Node(
            package='come_here_audio',
            executable='audio_node',
            name='audio_node',
            parameters=[default_params],
            output='screen',
        ),
    ])
