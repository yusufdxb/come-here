# shellcheck shell=bash
# Environment for the come-here class demo. Source it in EVERY terminal you use
# (launch, e-stop console, checks), so all of them share the same DDS settings:
#
#   source ~/come-here/scripts/demo_env.sh
#
# Separate shells with different DDS settings do not see each other's topics,
# which looks exactly like a dead node.

_ch_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ch_note() { printf '[demo_env] %s\n' "$*"; }

if [ ! -f /opt/ros/humble/setup.bash ]; then
  _ch_note "ERROR: /opt/ros/humble/setup.bash not found"
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

# unitree_api (Sport API messages) and the robot-side workspaces, when present.
for _ch_ws in "$HOME/unitree_ros2/cyclonedds_ws/install/setup.bash" \
              "$HOME/go2_ws/install/setup.bash"; do
  if [ -f "$_ch_ws" ]; then
    # shellcheck disable=SC1090
    source "$_ch_ws"
    _ch_note "sourced $_ch_ws"
  fi
done

if [ -f "$_ch_root/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$_ch_root/install/setup.bash"
else
  _ch_note "ERROR: $_ch_root/install/setup.bash missing; run: cd $_ch_root && colcon build --symlink-install"
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f "$HOME/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml" ]; then
  export CYCLONEDDS_URI="file://$HOME/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml"
fi
# The lab network has no internet: never let a model library try to download.
export HF_HUB_OFFLINE=1 YOLO_OFFLINE=true ULTRALYTICS_OFFLINE=true PYTHONUNBUFFERED=1

# A CycloneDDS interface name that does not exist on this host leaves every GO2
# topic advertised but carrying no data.
if [ -n "${CYCLONEDDS_URI:-}" ]; then
  _ch_xml="${CYCLONEDDS_URI#file://}"
  _ch_iface="$(grep -o 'NetworkInterface [^>]*name="[^"]*"' "$_ch_xml" 2>/dev/null | head -1 | sed 's/.*name="\([^"]*\)".*/\1/')"
  if [ -z "$_ch_iface" ]; then
    _ch_iface="$(grep -o '<NetworkInterfaceAddress>[^<]*' "$_ch_xml" 2>/dev/null | head -1 | sed 's/.*>//')"
  fi
  if [ -n "$_ch_iface" ] && [ ! -e "/sys/class/net/$_ch_iface" ]; then
    _ch_note "WARNING: CycloneDDS interface '$_ch_iface' does not exist on this host"
  fi
fi

_ch_note "repo=$_ch_root commit=$(git -C "$_ch_root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
_ch_note "RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID CYCLONEDDS_URI=${CYCLONEDDS_URI:-unset}"
unset _ch_ws _ch_xml _ch_iface
