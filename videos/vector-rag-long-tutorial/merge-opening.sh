#!/bin/sh
set -eu

DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
OPENING="${1:-$DIR/seedance/opening.mp4}"
TUTORIAL="$DIR/vector-rag-long-tutorial.mp4"
OUTPUT="$DIR/vector-rag-with-seedance-opening.mp4"
TUTORIAL_START_SECONDS="${TUTORIAL_START_SECONDS:-13.393}"

test -f "$OPENING" || { echo "缺少 Seedance 开场：$OPENING" >&2; exit 1; }
test -f "$TUTORIAL" || { echo "缺少教程视频：$TUTORIAL" >&2; exit 1; }

ffmpeg -y \
  -i "$OPENING" -i "$TUTORIAL" \
  -filter_complex "[0:v]setpts=PTS-STARTPTS,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v0];[0:a]asetpts=PTS-STARTPTS,aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono,afade=t=out:st=9.75:d=0.25[a0];[1:v]trim=start=${TUTORIAL_START_SECONDS},setpts=PTS-STARTPTS,scale=1080:1920,setsar=1,fps=30,format=yuv420p[v1];[1:a]atrim=start=${TUTORIAL_START_SECONDS},asetpts=PTS-STARTPTS,aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono,afade=t=in:st=0:d=0.15[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" \
  -map '[v]' -map '[a]' \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -shortest -movflags +faststart \
  "$OUTPUT"

ffprobe -v error -show_entries format=duration,size -of json "$OUTPUT"
