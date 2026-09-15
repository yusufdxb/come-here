#!/usr/bin/env bash
# Install (or refresh) the come-here boot service on the robot computer.
# Needs sudo: it writes /etc/systemd/system/come-here.service.
#
#   ./scripts/install_come_here_service.sh            # install, do not enable
#   ./scripts/install_come_here_service.sh --enable   # install and enable at boot
#
# The service starts in DRY RUN unless the live flag file exists:
#   touch <repo>/.come_here_live     # motion enabled at boot
#   rm    <repo>/.come_here_live     # dry run at boot
#
# Disable again:  sudo systemctl disable --now come-here
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$ROOT/systemd/come-here.service.in"
UNIT_DST=/etc/systemd/system/come-here.service
[ -f "$UNIT_SRC" ] || { echo "missing $UNIT_SRC"; exit 1; }
[ -x "$ROOT/scripts/come_here_boot.sh" ] || { echo "$ROOT/scripts/come_here_boot.sh is not executable"; exit 1; }

# Run under sudo, `id -un` is root and $HOME is /root: the service would then
# run as root and log into /root. The account that owns the checkout is the one
# that must run the stack, so resolve it from SUDO_USER.
TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || { echo "cannot resolve the home directory of $TARGET_USER"; exit 1; }
if [ "$TARGET_USER" = root ] && [ "${COME_HERE_ALLOW_ROOT:-0}" != 1 ]; then
  echo "refusing to install a service that runs as root."
  echo "run this from the robot account: sudo $0 ${*:-}"
  exit 1
fi
echo "service will run as $TARGET_USER (home $TARGET_HOME)"

sed -e "s|@ROOT@|$ROOT|g" -e "s|@USER@|$TARGET_USER|g" -e "s|@HOME@|$TARGET_HOME|g" "$UNIT_SRC" \
  | sudo tee "$UNIT_DST" >/dev/null
sudo systemctl daemon-reload
echo "installed $UNIT_DST"

if [ "${1:-}" = "--enable" ]; then
  sudo systemctl enable come-here.service
  echo "enabled at boot. Start now: sudo systemctl start come-here"
else
  echo "not enabled. Enable at boot: sudo systemctl enable --now come-here"
fi
if [ -f "$ROOT/.come_here_live" ]; then
  echo "live flag present: the service WILL command motion on a wake phrase"
else
  echo "no live flag: the service starts in dry run (no motion)"
fi
echo "logs: journalctl -u come-here -f   and   $TARGET_HOME/come_here_trials/boot_service.log"
