"""COME HERE ANY (experimental): "come here" -> DOA turn -> DOA-gated caller -> pursuit
through the GO2's native obstacle avoidance -> verified facing -> sit.

    ros2 launch come_here_bringup come_here_any.launch.py                  # dry run
    ros2 launch come_here_bringup come_here_any.launch.py dry_run:=false \
        native_live_motion_cleared:=true                                    # live motion

This is NOT the class demo. The legacy, hardware-demonstrated demo is
professor_demo.launch.py and is untouched by this file. Same camera, audio,
perception, face and viewer processes; the behavior and bridge nodes are the
ANY executables (come_here_any_behavior_node, native_avoid_bridge_node) under
the usual node names, so estop_console and trial_report work unchanged. Only
one bridge (motion owner) runs: never start this beside professor_demo.

dry_run defaults to true (requests go to /come_here/dry_run/*, native replies
are simulated). dry_run:=false still moves nothing unless
native_live_motion_cleared:=true, which is only for after the Stage A-D ladder
in docs/come_here_any.md has passed.
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
        get_package_share_directory('come_here_bringup'), 'config', 'come_here_any.yaml'
    )
    sounds = os.path.join(get_package_share_directory('come_here_audio'), 'sounds')

    dry_run = LaunchConfiguration('dry_run')
    use_mock = LaunchConfiguration('use_mock')
    camera = LaunchConfiguration('camera')
    mock = {'use_mock': ParameterValue(use_mock, value_type=bool)}
    # Symlink install: the launch file resolves to the repo, next to scripts/.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    view_script = os.path.join(repo, 'scripts', 'demo_view.py')
    if not os.path.isfile(view_script):
        view_script = os.path.expanduser('~/come-here-demo/scripts/demo_view.py')

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
            'max_walk_distance_m', default_value='6.0',
            description='commanded walking budget: an abort backstop in ANY, never an arrival',
        ),
        DeclareLaunchArgument(
            'native_avoid_backend', default_value='sport_freeavoid',
            description='sport_freeavoid (FreeAvoid 2048 + Sport Move) or obstacles_avoid',
        ),
        DeclareLaunchArgument(
            'native_live_motion_cleared', default_value='false',
            description='true only after the Stage A-D ladder passed; live motion is refused otherwise',
        ),
        DeclareLaunchArgument(
            'trial_log_dir', default_value='~/come_here_trials',
        ),
        DeclareLaunchArgument(
            'skip_turn_to_sound', default_value='false',
            description='true: no turn toward the voice, the caller must start in camera view',
        ),
        DeclareLaunchArgument(
            'doa_calibration_path', default_value='~/come_here_trials/doa_calibration.json',
            description='scripts/calibrate_doa.py output; required: no file = no turn',
        ),
        DeclareLaunchArgument(
            'view', default_value='true',
            description='serve the operator view (subscribe-only) on port 8088',
        ),
        DeclareLaunchArgument('view_script', default_value=view_script),
        DeclareLaunchArgument(
            'doa_offset_deg', default_value='-90.0',
            description='software DOA mount offset, from scripts/doa_probe.py (caller ahead)',
        ),
        DeclareLaunchArgument(
            'doa_mirror', default_value='false',
            description='true if the probe reports left callers as right',
        ),
        DeclareLaunchArgument(
            'direction_confidence_threshold', default_value='0.4',
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
                'doa_calibration_path': LaunchConfiguration('doa_calibration_path'),
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
        Node(
            package='come_here_perception',
            executable='face_detector_node',
            name='face_detector_node',
            parameters=[config, mock],
            output='screen',
            respawn=True,
            respawn_delay=2.0,
        ),
        ExecuteProcess(
            cmd=['nice', '-n', '19', 'python3', '-u', LaunchConfiguration('view_script'),
                 '--port', '8088'],
            name='demo_view',
            output='screen',
            respawn=True,
            respawn_delay=3.0,
            condition=IfCondition(LaunchConfiguration('view')),
        ),
        # behavior_node and go2_bridge_node are never respawned: a restart
        # would forget a latched e-stop.
        Node(
            package='come_here_behavior',
            executable='come_here_any_behavior_node',
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
            executable='native_avoid_bridge_node',
            name='go2_bridge_node',
            parameters=[config, {
                'dry_run': ParameterValue(dry_run, value_type=bool),
                'wav_dir': sounds,
                'native_avoid_backend': LaunchConfiguration('native_avoid_backend'),
                'native_live_motion_cleared': ParameterValue(
                    LaunchConfiguration('native_live_motion_cleared'), value_type=bool),
            }],
            output='screen',
            condition=UnlessCondition(use_mock),
        ),
    ])
