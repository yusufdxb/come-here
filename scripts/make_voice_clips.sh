#!/usr/bin/env bash
# Regenerate the GO2 speech clips for every phrase in professor_demo.yaml.
# Voice: edge-tts en-US-AriaNeural (the voice heard on the GO2 speaker in April
# and September 2026), converted to 16 kHz mono s16 for the audiohub stream.
# Needs internet once; the clips stay in come_here_audio/scripts (not in git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/come_here_audio/scripts"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
python3 - "$ROOT/come_here_bringup/config/professor_demo.yaml" <<'PY' > "$TMP/phrases"
import sys, yaml
b = yaml.safe_load(open(sys.argv[1]))['behavior_node']['ros__parameters']
seen = set()
for key in ('wake_speak_text', 'direction_speak_text', 'acquired_speak_text', 'speak_text'):
    for p in str(b.get(key) or '').split('|'):
        p = p.strip()
        if p and p not in seen:
            seen.add(p); print(p)
PY
while IFS= read -r phrase; do
  slug="$(python3 -c "import re,sys; print(re.sub(r'[^a-z0-9]+','_',sys.argv[1].lower()).strip('_'))" "$phrase")"
  edge-tts --voice en-US-AriaNeural --text "$phrase" --write-media "$TMP/$slug.mp3" >/dev/null
  ffmpeg -loglevel error -y -i "$TMP/$slug.mp3" -ar 16000 -ac 1 -c:a pcm_s16le "$OUT/$slug.wav"
  echo "wrote $OUT/$slug.wav  ($phrase)"
done < "$TMP/phrases"
