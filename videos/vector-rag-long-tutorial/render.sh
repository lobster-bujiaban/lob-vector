#!/bin/sh
set -eu

DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$DIR/../.." && pwd)"
VOICE="${VOICE:-zh-CN-XiaoyiNeural}"
RATE="${RATE:-+25%}"

cd "$ROOT"

if [ "${SKIP_TTS:-0}" != "1" ]; then
  edge-tts \
    --voice "$VOICE" \
    --rate="$RATE" \
    --file "$DIR/narration.txt" \
    --write-media "$DIR/narration.mp3"
fi

DURATION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$DIR/narration.mp3")"

ffmpeg -y \
  -f lavfi -i "color=c=0x101412:s=1080x1920:r=30:d=$DURATION" \
  -i "$DIR/narration.mp3" \
  -filter_complex_script "$DIR/video-filter.txt" \
  -map '[v]' -map 1:a \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -shortest -movflags +faststart \
  "$DIR/vector-rag-long-tutorial.mp4"

ffprobe -v error \
  -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate \
  -show_entries format=duration,size \
  -of json "$DIR/vector-rag-long-tutorial.mp4"
