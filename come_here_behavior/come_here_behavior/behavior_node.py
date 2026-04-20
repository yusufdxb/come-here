"""State machine for come-here behavior.

States:
  IDLE               - waiting, no command received
  LISTENING          - wake phrase detected, collecting audio direction
  TURN_TO_SOUND      - rotating toward estimated sound source
  SEARCH_FOR_PERSON  - scanning for a person visually
  APPROACH_PERSON    - moving toward detected person (visual servoing)
  SIT_AND_IDENTIFY   - sit, run face detection, speak, stand, return to IDLE

Subscribes:
  /come_here/wake_phrase       (std_msgs/String)
  /come_here/audio_direction   (std_msgs/Float64MultiArray) [azimuth, confidence]
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]
  /come_here/face_detection    (come_here_msgs/FaceDetection)

Publishes:
  /come_here/cmd_rotate             (std_msgs/Float64) - target rotation in radians
  /come_here/cmd_move               (std_msgs/Float64) - forward velocity command
  /come_here/cmd_sit                (std_msgs/Bool)
  /come_here/cmd_stand              (std_msgs/Bool)
  /come_here/cmd_say                (std_msgs/String)
  /come_here/face_detect_request    (std_msgs/Bool)
  /come_here/state                  (std_msgs/String)
"""

from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

from come_here_msgs.msg import FaceDetection


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TURN_TO_SOUND = auto()
    SEARCH_FOR_PERSON = auto()
    APPROACH_PERSON = auto()
    SIT_AND_IDENTIFY = auto()


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # -- Parameters --
        self.declare_parameter('tick_rate_hz', 10.0)
        self.declare_parameter('direction_confidence_threshold', 0.5)
        self.declare_parameter('person_confidence_threshold', 0.5)
        self.declare_parameter('approach_stop_distance_m', 0.8)
        self.declare_parameter('search_timeout_s', 10.0)
        self.declare_parameter('approach_speed', 0.3)
        self.declare_parameter('lost_timeout_s', 1.0)
        self.declare_parameter('sit_settle_s', 1.0)
        self.declare_parameter('face_timeout_s', 1.5)
        self.declare_parameter('speak_hold_s', 5.0)
        self.declare_parameter('stand_settle_s', 0.5)
        self.declare_parameter('speak_text', 'I am here')
        self.declare_parameter('wake_speak_text', 'I am coming')
        self.declare_parameter('approach_align_threshold_rad', 0.26)
        self.declare_parameter('approach_ccw_yaw', 1.0)
        self.declare_parameter('approach_cw_yaw', 1.2)

        self._dir_threshold = self.get_parameter('direction_confidence_threshold').value
        self._person_threshold = self.get_parameter('person_confidence_threshold').value
        self._stop_distance = self.get_parameter('approach_stop_distance_m').value
        self._search_timeout = self.get_parameter('search_timeout_s').value
        self._approach_speed = self.get_parameter('approach_speed').value
        self._lost_timeout = self.get_parameter('lost_timeout_s').value
        self._sit_settle_s = self.get_parameter('sit_settle_s').value
        self._face_timeout_s = self.get_parameter('face_timeout_s').value
        self._speak_hold_s = self.get_parameter('speak_hold_s').value
        self._stand_settle_s = self.get_parameter('stand_settle_s').value
        self._speak_text = self.get_parameter('speak_text').value
        self._wake_speak_text = self.get_parameter('wake_speak_text').value
        self._approach_align_threshold = self.get_parameter(
            'approach_align_threshold_rad'
        ).value
        self._approach_ccw_yaw = self.get_parameter('approach_ccw_yaw').value
        self._approach_cw_yaw = self.get_parameter('approach_cw_yaw').value

        # -- Runtime state --
        self._state = State.IDLE
        self._last_azimuth = 0.0
        self._last_dir_confidence = 0.0
        self._person_detected = False
        self._person_bearing = 0.0
        self._person_distance = 0.0
        self._person_confidence = 0.0
        self._person_last_seen = None
        self._search_start_time = None

        # APPROACH_PERSON sub-phase state (commit-phase, no per-tick switching).
        # Phase is 'ALIGN' (rotate in place) or 'WALK' (forward only).
        # min_phase_s is the minimum time we stay in a phase before re-evaluating.
        self._approach_phase = 'ALIGN'
        self._approach_phase_start = None
        self._approach_min_align_s = 0.4
        self._approach_min_walk_s = 1.5

        # SIT_AND_IDENTIFY sub-sequence state
        self._sit_substep = 0
        self._sit_step_time = None
        self._face_received_in_sequence = False
        self._last_face_result = None

        # -- Subscribers --
        self.create_subscription(
            String, '/come_here/wake_phrase', self._wake_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/come_here/audio_direction', self._direction_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/come_here/person_detection', self._person_cb, 10
        )
        self.create_subscription(
            FaceDetection, '/come_here/face_detection', self._face_cb, 10
        )

        # -- Publishers --
        self._rotate_pub = self.create_publisher(Float64, '/come_here/cmd_rotate', 10)
        self._move_pub = self.create_publisher(Float64, '/come_here/cmd_move', 10)
        self._velocity_pub = self.create_publisher(
            Float64MultiArray, '/come_here/cmd_velocity', 10
        )
        self._sit_pub = self.create_publisher(Bool, '/come_here/cmd_sit', 10)
        self._stand_pub = self.create_publisher(Bool, '/come_here/cmd_stand', 10)
        self._say_pub = self.create_publisher(String, '/come_here/cmd_say', 10)
        self._face_req_pub = self.create_publisher(
            Bool, '/come_here/face_detect_request', 10
        )
        self._state_pub = self.create_publisher(String, '/come_here/state', 10)

        rate = self.get_parameter('tick_rate_hz').value
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'Behavior node started in {self._state.name}')

    # -- Callbacks --

    def _wake_cb(self, msg: String):
        if self._state == State.IDLE:
            self.get_logger().info(f'Wake phrase received: "{msg.data}"')
            if self._wake_speak_text:
                say = String()
                say.data = self._wake_speak_text
                self._say_pub.publish(say)
                self.get_logger().info(f'WAKE: saying "{self._wake_speak_text}"')
            self._transition(State.LISTENING)

    def _direction_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            self._last_azimuth = msg.data[0]
            self._last_dir_confidence = msg.data[1]

    def _person_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self._person_bearing = msg.data[0]
            self._person_distance = msg.data[1]
            self._person_confidence = msg.data[2]
            detected = bool(msg.data[3])
            self._person_detected = detected
            if detected:
                self._person_last_seen = self.get_clock().now()

    def _face_cb(self, msg: FaceDetection):
        self._last_face_result = msg
        if self._state == State.SIT_AND_IDENTIFY and self._sit_substep == 1:
            self._face_received_in_sequence = True

    # -- State machine tick --

    def _tick(self):
        # Publish current state name
        state_msg = String()
        state_msg.data = self._state.name
        self._state_pub.publish(state_msg)

        if self._state == State.IDLE:
            return

        if self._state == State.LISTENING:
            if self._last_dir_confidence >= self._dir_threshold:
                self._transition(State.TURN_TO_SOUND)
            return

        if self._state == State.TURN_TO_SOUND:
            rotate_msg = Float64()
            rotate_msg.data = self._last_azimuth
            self._rotate_pub.publish(rotate_msg)
            self.get_logger().info(
                f'Rotating toward sound: {self._last_azimuth:.2f} rad'
            )
            self._transition(State.SEARCH_FOR_PERSON)
            self._search_start_time = self.get_clock().now()
            self._person_last_seen = None
            return

        if self._state == State.SEARCH_FOR_PERSON:
            elapsed = self._seconds_since(self._search_start_time)
            if self._person_detected and self._person_confidence >= self._person_threshold:
                self._transition(State.APPROACH_PERSON)
            elif elapsed > self._search_timeout:
                self.get_logger().warn('Search timed out, returning to IDLE')
                self._transition(State.IDLE)
            return

        if self._state == State.APPROACH_PERSON:
            self._tick_approach()
            return

        if self._state == State.SIT_AND_IDENTIFY:
            self._tick_sit_sequence()
            return

    # -- APPROACH_PERSON controller --

    def _tick_approach(self):
        # Lost-timeout check: go back to search if we haven't seen a person recently.
        if self._person_last_seen is None:
            return
        elapsed_since_seen = self._seconds_since(self._person_last_seen)
        if elapsed_since_seen > self._lost_timeout:
            self.get_logger().warn(
                f'Lost person for {elapsed_since_seen:.1f}s, re-searching'
            )
            self._stop_motion()
            self._search_start_time = self.get_clock().now()
            self._transition(State.SEARCH_FOR_PERSON)
            return

        # Close-enough check: trigger sit sequence
        if self._person_distance > 0 and self._person_distance <= self._stop_distance:
            self._stop_motion()
            self._enter_sit_sequence()
            return

        # Commit-phase controller: mcf gait can't cleanly combine vx+yaw and
        # needs a stable setpoint for ~1s+ to engage a clean trot. Publishing
        # [vx, yaw] with vx flipping each tick (10 Hz) caused aggressive shake
        # rather than locomotion. So: hold one axis per phase, for a minimum
        # duration, before re-evaluating.
        if self._approach_phase_start is None:
            self._approach_phase_start = self.get_clock().now()
        phase_elapsed = self._seconds_since(self._approach_phase_start)
        bearing = self._person_bearing
        vel_msg = Float64MultiArray()

        if self._approach_phase == 'ALIGN':
            if (abs(bearing) < self._approach_align_threshold
                    and phase_elapsed >= self._approach_min_align_s):
                self._approach_phase = 'WALK'
                self._approach_phase_start = self.get_clock().now()
                vel_msg.data = [self._approach_speed, 0.0]
            else:
                yaw_rate = self._approach_ccw_yaw if bearing > 0 else -self._approach_cw_yaw
                vel_msg.data = [0.0, yaw_rate]
        else:  # WALK
            # Stay committed to WALK for at least min_walk_s, regardless of
            # bearing jitter. Only re-align if we've walked the minimum and
            # bearing is well outside the deadband.
            if (phase_elapsed >= self._approach_min_walk_s
                    and abs(bearing) > 2.0 * self._approach_align_threshold):
                self._approach_phase = 'ALIGN'
                self._approach_phase_start = self.get_clock().now()
                yaw_rate = self._approach_ccw_yaw if bearing > 0 else -self._approach_cw_yaw
                vel_msg.data = [0.0, yaw_rate]
            else:
                vel_msg.data = [self._approach_speed, 0.0]

        self._velocity_pub.publish(vel_msg)

    # -- SIT_AND_IDENTIFY sequence --

    def _enter_sit_sequence(self):
        self._sit_substep = 0
        self._sit_step_time = self.get_clock().now()
        self._face_received_in_sequence = False
        self._last_face_result = None

        # Immediate: publish sit
        sit_msg = Bool()
        sit_msg.data = True
        self._sit_pub.publish(sit_msg)
        self.get_logger().info('SIT_AND_IDENTIFY: sitting')
        self._transition(State.SIT_AND_IDENTIFY)

    def _tick_sit_sequence(self):
        elapsed = self._seconds_since(self._sit_step_time)

        if self._sit_substep == 0:
            # Waiting for sit to settle, then fire face detect request
            if elapsed >= self._sit_settle_s:
                req = Bool()
                req.data = True
                self._face_req_pub.publish(req)
                self.get_logger().info('SIT_AND_IDENTIFY: face detect requested')
                self._sit_substep = 1
                self._sit_step_time = self.get_clock().now()

        elif self._sit_substep == 1:
            # Waiting for face detection result OR timeout, then speak
            if self._face_received_in_sequence or elapsed >= self._face_timeout_s:
                self._log_face_result()
                say = String()
                say.data = self._speak_text
                self._say_pub.publish(say)
                self.get_logger().info(f'SIT_AND_IDENTIFY: saying "{self._speak_text}"')
                self._sit_substep = 2
                self._sit_step_time = self.get_clock().now()

        elif self._sit_substep == 2:
            # Waiting for speak to finish, then stand
            if elapsed >= self._speak_hold_s:
                stand = Bool()
                stand.data = True
                self._stand_pub.publish(stand)
                self.get_logger().info('SIT_AND_IDENTIFY: standing')
                self._sit_substep = 3
                self._sit_step_time = self.get_clock().now()

        elif self._sit_substep == 3:
            # Waiting for stand to settle, then back to IDLE
            if elapsed >= self._stand_settle_s:
                self.get_logger().info('SIT_AND_IDENTIFY: sequence complete')
                self._transition(State.IDLE)

    def _log_face_result(self):
        if self._last_face_result is None:
            self.get_logger().info('Face result: none received (timeout)')
            return
        r = self._last_face_result
        self.get_logger().info(
            f'Face result: present={r.face_present} count={r.face_count} '
            f'conf={r.max_confidence:.2f}'
        )

    # -- Helpers --

    def _transition(self, new_state: State):
        old = self._state.name
        self._state = new_state
        self.get_logger().info(f'State: {old} -> {new_state.name}')
        if new_state == State.APPROACH_PERSON:
            self._approach_phase = 'ALIGN'
            self._approach_phase_start = None

    def _stop_motion(self):
        move_msg = Float64()
        move_msg.data = 0.0
        self._move_pub.publish(move_msg)

    def _seconds_since(self, start_time) -> float:
        if start_time is None:
            return 0.0
        return (self.get_clock().now() - start_time).nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # A second SIGINT can land during teardown or during interpreter
        # shutdown (e.g. threading._shutdown). Ignore it for the rest of the
        # process so shutdown stays quiet.
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
