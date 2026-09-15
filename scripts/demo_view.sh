#!/usr/bin/env bash
# Operator laptop: ONE command opens the live GO2 view (camera, person boxes,
# selected caller, gate, state). The Jetson draws it (scripts/demo_view.py,
# started by the launch); this only opens a low-latency player on the stream.
#
#   ./scripts/demo_view.sh              # Jetson on lab wifi, 192.168.0.70
#   ./scripts/demo_view.sh 192.168.123.18
set -u
HOST="${1:-${JETSON_IP:-192.168.0.70}}"
PORT="${DEMO_VIEW_PORT:-8088}"
URL="http://${HOST}:${PORT}/camera.mjpg"
for _ in $(seq 1 60); do
  curl -s -m 2 -o /dev/null "http://${HOST}:${PORT}/status.json" && break
  echo "waiting for ${URL} (is the demo launch running with view:=true?)"
  sleep 2
done
echo "opening ${URL}"
if command -v ffplay >/dev/null 2>&1; then
  exec env -u LD_LIBRARY_PATH ffplay -hide_banner -loglevel error \
    -fflags nobuffer -flags low_delay -framedrop -probesize 32 -analyzeduration 0 \
    -window_title "GO2 come-here" "${URL}"
elif command -v vlc >/dev/null 2>&1 || [ -x /snap/bin/vlc ]; then
  exec env -u LD_LIBRARY_PATH "$(command -v vlc || echo /snap/bin/vlc)" \
    --network-caching=50 --no-video-title-show "${URL}"
else
  xdg-open "http://${HOST}:${PORT}/"
fi
