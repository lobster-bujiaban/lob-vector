#!/bin/sh
set -eu

cd "$(dirname "$0")"

ffmpeg -y \
  -f lavfi -i "color=c=0x101412:s=1080x1920:r=30:d=53" \
  -i narration.mp3 \
  -filter_complex_script video-filter.txt \
  -map '[v]' -map 1:a \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -shortest -movflags +faststart \
  keep-scrolling.mp4
