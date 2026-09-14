"""Class demo launch: "come here" -> find the caller -> ALIGN/WALK -> stop.

    ros2 launch come_here_bringup professor_demo.launch.py                 # dry run
    ros2 launch come_here_bringup professor_demo.launch.py dry_run:=false  # live

Starts the camera publisher, audio_node (Whisper wake phrase), perception_node
(YOLO), behavior_node (state machine + trial log) and go2_bridge_node. No
face detector, no TURN_TO_SOUND, no sit.

dry_run defaults to true: the bridge sends Sport API requests to
/come_here/dry_run/sport_request, which the robot ignores. Motion needs the
explicit dry_run:=false.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('come_here_bringup'), 'config', 'professor_demo.yaml'
    )
    sounds = os.path.join(get_package_share_directory('come_here_audio'), 'sounds')

    dry_run = LaunchConfiguration('dry_run')
    use_mock = LaunchConfiguration('use_mock')
    camera = LaunchConfiguration('camera')
    mock = {'use_mock': ParameterValue(use_mock, value_type=bool)}

    return LaunchDescription([
        DeclareLaunchArgument(
            'dry_run', default_value='true',
            description='true: Sport requests go to a dry-run topic, the robot never moves',
        ),
        DeclareLaunchArgument(
            'use_mock', default_value='false',
            description='true: mock sensors, no bridge (desk testing without the robot)',
        ),
        DeclareLaunchArgument(
            'camera', default_value='true',
            description='start the GO2 front camera publisher script',
        ),
        DeclareLaunchArgument(
            'camera_script', default_value='/home/unitree/go2_video_publisher.py',
            description='publishes sensor_msgs/Image on /camera/image_raw',
        ),
        DeclareLaunchArgument(
            'respeaker_profile', default_value='far_field',
            description='far_field or none: ReSpeaker DSP profile written at startup',
        ),
        DeclareLaunchArgument(
            'adaptive_gate', default_value='true',
            description='false restores the fixed 0.015 RMS wake gate',
        ),
        DeclareLaunchArgument(
            'max_walk_distance_m', default_value='2.0',
            description='commanded walking budget; set to caller start distance minus 0.5 m',
        ),
        DeclareLaunchArgument(
            'trial_log_dir', default_value='~/come_here_trials',
        ),
        DeclareLaunchArgument(
            'skip_turn_to_sound', default_value='false',
            description='true: no turn toward the voice, the caller must start in camera view',
        ),
        DeclareLaunchArgument(
            'doa_offset_deg', default_value='0.0',
            description='software DOA mount offset, from scripts/doa_probe.py (caller ahead)',
        ),
        DeclareLaunchArgument(
            'doa_mirror', default_value='false',
            description='true if the probe reports left callers as right',
        ),
        DeclareLaunchArgument(
            'direction_confidence_threshold', default_value='0.5',
            description='DOA confidence needed to turn; below it the demo is camera only',
        ),

        ExecuteProcess(
            cmd=['python3', '-u', LaunchConfiguration('camera_script')],
            name='go2_camera',
            output='screen',
            respawn=True,
            respawn_delay=2.0,
            condition=IfCondition(PythonExpression([
                "'", camera, "'.lower() == 'true' and '", use_mock, "'.lower() == 'false'",
            ])),
        ),
        Node(
            package='come_here_audio',
            executable='audio_node',
            name='audio_node',
            parameters=[config, mock, {
                'respeaker_profile': LaunchConfiguration('respeaker_profile'),
                'adaptive_gate': ParameterValue(
                    LaunchConfiguration('adaptive_gate'), value_type=bool),
                'respeaker_frame_offset_deg': ParameterValue(
                    LaunchConfiguration('doa_offset_deg'), value_type=float),
                'doa_mirror': ParameterValue(
                    LaunchConfiguration('doa_mirror'), value_type=bool),
            }],
            output='screen',
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package='come_here_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[config, mock],
            output='screen',
            respawn=True,
            respawn_delay=2.0,
        ),
        # behavior_node and go2_bridge_node are never respawned: a restart
        # would forget a latched e-stop.
        Node(
            package='come_here_behavior',
            executable='behavior_node',
            name='behavior_node',
            parameters=[config, {
                'max_walk_distance_m': ParameterValue(
                    LaunchConfiguration('max_walk_distance_m'), value_type=float),
                'trial_log_dir': LaunchConfiguration('trial_log_dir'),
                'skip_turn_to_sound': ParameterValue(
                    LaunchConfiguration('skip_turn_to_sound'), value_type=bool),
                'direction_confidence_threshold': ParameterValue(
                    LaunchConfiguration('direction_confidence_threshold'), value_type=float),
            }],
            output='screen',
        ),
        Node(
            package='come_here_behavior',
            executable='go2_bridge_node',
            name='go2_bridge_node',
            parameters=[config, {
                'dry_run': ParameterValue(dry_run, value_type=bool),
                'wav_dir': sounds,
            }],
            output='screen',
            condition=UnlessCondition(use_mock),
        ),
    ])
