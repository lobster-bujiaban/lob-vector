#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

edge-tts \
  --voice zh-CN-XiaoyiNeural \
  --rate=+70% \
  --file videos/narration.txt \
  --write-media videos/narration.mp3
ffmpeg -y \
  -f lavfi -i "color=c=0x101412:s=1080x1920:r=30:d=56" \
  -i videos/narration.mp3 \
  -filter_complex_script videos/video-filter.txt \
  -map '[v]' -map 1:a \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -shortest -movflags +faststart \
  videos/embedding-explainer.mp4
