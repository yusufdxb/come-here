"""State machine for come-here behavior.

States:
  IDLE             - waiting, no command received
  LISTENING        - wake phrase detected, collecting audio direction
  TURN_TO_SOUND    - rotating toward estimated sound source
  SEARCH_FOR_PERSON - scanning for a person visually
  APPROACH_PERSON  - moving toward detected person
  STOP             - arrived or aborted, waiting for reset

Subscribes:
  /come_here/wake_phrase       (std_msgs/String)
  /come_here/audio_direction   (std_msgs/Float64MultiArray) [azimuth, confidence]
  /come_here/person_detection  (std_msgs/Float64MultiArray) [bearing, distance, confidence, detected]

Publishes:
  /come_here/cmd_rotate        (std_msgs/Float64) - target rotation in radians
  /come_here/cmd_move          (std_msgs/Float64) - forward velocity command
  /come_here/state             (std_msgs/String) - current state name
"""

from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, String


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TURN_TO_SOUND = auto()
    SEARCH_FOR_PERSON = auto()
    APPROACH_PERSON = auto()
    STOP = auto()


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        # Parameters
        self.declare_parameter('tick_rate_hz', 10.0)
        self.declare_parameter('direction_confidence_threshold', 0.5)
        self.declare_parameter('person_confidence_threshold', 0.5)
        self.declare_parameter('approach_stop_distance_m', 0.8)
        self.declare_parameter('search_timeout_s', 10.0)
        self.declare_parameter('approach_speed', 0.3)

        self._dir_threshold = self.get_parameter('direction_confidence_threshold').value
        self._person_threshold = self.get_parameter('person_confidence_threshold').value
        self._stop_distance = self.get_parameter('approach_stop_distance_m').value
        self._search_timeout = self.get_parameter('search_timeout_s').value
        self._approach_speed = self.get_parameter('approach_speed').value

        # State
        self._state = State.IDLE
        self._last_azimuth = 0.0
        self._last_dir_confidence = 0.0
        self._person_detected = False
        self._person_bearing = 0.0
        self._person_distance = 0.0
        self._person_confidence = 0.0
        self._search_start_time = None

        # Subscribers
        self.create_subscription(
            String, '/come_here/wake_phrase', self._wake_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/come_here/audio_direction', self._direction_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/come_here/person_detection', self._person_cb, 10
        )

        # Publishers
        self._rotate_pub = self.create_publisher(Float64, '/come_here/cmd_rotate', 10)
        self._move_pub = self.create_publisher(Float64, '/come_here/cmd_move', 10)
        self._state_pub = self.create_publisher(String, '/come_here/state', 10)

        # Tick timer
        rate = self.get_parameter('tick_rate_hz').value
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'Behavior node started in {self._state.name}')

    # -- Callbacks --

    def _wake_cb(self, msg: String):
        if self._state == State.IDLE:
            self.get_logger().info(f'Wake phrase received: "{msg.data}"')
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
            self._person_detected = bool(msg.data[3])

    # -- State machine --

    def _tick(self):
        # Publish current state
        state_msg = String()
        state_msg.data = self._state.name
        self._state_pub.publish(state_msg)

        if self._state == State.IDLE:
            pass  # waiting for wake phrase

        elif self._state == State.LISTENING:
            # Wait for a confident direction estimate
            if self._last_dir_confidence >= self._dir_threshold:
                self._transition(State.TURN_TO_SOUND)

        elif self._state == State.TURN_TO_SOUND:
            # Command rotation toward sound source
            # TODO: Replace with real motion command to GO2 SDK or cmd_vel
            rotate_msg = Float64()
            rotate_msg.data = self._last_azimuth
            self._rotate_pub.publish(rotate_msg)
            self.get_logger().info(
                f'Rotating toward sound: {self._last_azimuth:.2f} rad'
            )
            # After issuing rotation, transition to search
            self._transition(State.SEARCH_FOR_PERSON)
            self._search_start_time = self.get_clock().now()

        elif self._state == State.SEARCH_FOR_PERSON:
            elapsed = (self.get_clock().now() - self._search_start_time).nanoseconds / 1e9
            if self._person_detected and self._person_confidence >= self._person_threshold:
                self._transition(State.APPROACH_PERSON)
            elif elapsed > self._search_timeout:
                self.get_logger().warn('Search timed out, returning to IDLE')
                self._transition(State.IDLE)

        elif self._state == State.APPROACH_PERSON:
            if not self._person_detected:
                # Lost the person, go back to searching
                self._transition(State.SEARCH_FOR_PERSON)
                self._search_start_time = self.get_clock().now()
                return

            if self._person_distance > 0 and self._person_distance <= self._stop_distance:
                # Close enough, stop
                self._stop_motion()
                self._transition(State.STOP)
                return

            # TODO: Replace with real GO2 locomotion commands
            # Steer toward person and move forward
            rotate_msg = Float64()
            rotate_msg.data = self._person_bearing
            self._rotate_pub.publish(rotate_msg)

            move_msg = Float64()
            move_msg.data = self._approach_speed
            self._move_pub.publish(move_msg)

        elif self._state == State.STOP:
            self._stop_motion()
            self.get_logger().info('Arrived. Returning to IDLE.')
            self._transition(State.IDLE)

    def _transition(self, new_state: State):
        old = self._state.name
        self._state = new_state
        self.get_logger().info(f'State: {old} -> {new_state.name}')

    def _stop_motion(self):
        move_msg = Float64()
        move_msg.data = 0.0
        self._move_pub.publish(move_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
