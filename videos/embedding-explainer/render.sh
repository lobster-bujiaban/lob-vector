#!/bin/sh
set -eu

DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$DIR/../.." && pwd)"

cd "$ROOT"

edge-tts \
  --voice zh-CN-XiaoyiNeural \
  --rate=+70% \
  --file "$DIR/narration.txt" \
  --write-media "$DIR/narration.mp3"
ffmpeg -y \
  -f lavfi -i "color=c=0x101412:s=1080x1920:r=30:d=56" \
  -i "$DIR/narration.mp3" \
  -filter_complex_script "$DIR/video-filter.txt" \
  -map '[v]' -map 1:a \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -shortest -movflags +faststart \
  "$DIR/embedding-explainer.mp4"
