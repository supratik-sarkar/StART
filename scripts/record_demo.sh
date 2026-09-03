#!/usr/bin/env bash
#
# Record the StART demonstration flight for the README.
#
# Produces docs/media/start-demo.mp4 plus a poster frame. GitHub will not play
# an <video> tag from a README, so the README links a poster image to the asset
# instead — one click, plays in a new tab. See README "Watch the demo".
#
#   ./scripts/record_demo.sh                        # no keys: deterministic run
#   ./scripts/record_demo.sh --provider openai --model gpt-4.1-mini
#
# Requirements (macOS):
#   brew install asciinema agg ffmpeg
#
# asciinema records the terminal as text, agg renders it to GIF, ffmpeg converts
# to MP4. Recording text rather than pixels matters: the capture is a fraction
# of the size, stays sharp at any resolution, and — the part people forget — can
# be diffed. You can prove the recording matches what the code actually printed.
set -euo pipefail
cd "$(dirname "$0")/.."

MEDIA_DIR="docs/media"
CAST="${MEDIA_DIR}/start-demo.cast"
GIF="${MEDIA_DIR}/start-demo.gif"
MP4="${MEDIA_DIR}/start-demo.mp4"
POSTER="${MEDIA_DIR}/start-demo-poster.png"

mkdir -p "${MEDIA_DIR}"

missing=()
for tool in asciinema agg ffmpeg; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing: ${missing[*]}"
  echo "Install with:  brew install ${missing[*]}"
  echo
  echo "Or record with QuickTime (File > New Screen Recording), save to ${MP4},"
  echo "and skip this script entirely — the README markup is the same either way."
  exit 2
fi

echo "Recording. Use a 100x32 terminal for the cleanest result."
rm -f "${CAST}"

asciinema rec "${CAST}" \
  --cols 100 --rows 32 --idle-time-limit 2 --overwrite \
  --command "python scripts/demo_flight.py $*"

echo "Rendering GIF..."
agg --theme monokai --font-size 15 --speed 1.0 "${CAST}" "${GIF}"

echo "Converting to MP4..."
ffmpeg -y -loglevel error -i "${GIF}" \
  -movflags faststart -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "${MP4}"

echo "Extracting poster frame..."
ffmpeg -y -loglevel error -ss 00:00:02 -i "${MP4}" -frames:v 1 "${POSTER}"

echo
echo "Done:"
ls -lh "${CAST}" "${GIF}" "${MP4}" "${POSTER}" | awk '{print "  " $9 "  " $5}'
echo
echo "Commit these, then push. The README already points at them."
