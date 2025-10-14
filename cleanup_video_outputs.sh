#!/bin/bash

set -euo pipefail

OUTPUT_DIR="${1:-ai_video_editor_output}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output directory not found (nothing to clean): $OUTPUT_DIR"
  exit 0
fi

echo "Cleaning video artifacts under $OUTPUT_DIR"

# Remove directories that hold generated video clips.
for dir in "chunks" "new" "edited"; do
  target="$OUTPUT_DIR/$dir"
  if [[ -d "$target" ]]; then
    rm -rf "$target"
    echo "Removed directory: $target"
  fi
done

# Remove final rendered videos within the output tree.
find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.mp4" -print -delete

echo "Cleanup complete."
