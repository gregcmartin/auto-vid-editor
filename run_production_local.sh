#!/bin/bash

# AI Video Editor - Production Run with Local Models
# This script processes a real video with full production settings

echo "=========================================="
echo "AI Video Editor - Production Local Run"
echo "=========================================="
echo ""

# Check if video file is provided
if [ -z "$1" ]; then
    echo "ERROR: No video file specified!"
    echo ""
    echo "Usage: ./run_production_local.sh <video_file> [options]"
    echo ""
    echo "Example:"
    echo "  ./run_production_local.sh my_video.mp4"
    echo "  ./run_production_local.sh my_video.mp4 --parts 4"
    echo "  ./run_production_local.sh my_video.mp4 --device cuda --torch-dtype bfloat16"
    echo ""
    exit 1
fi

VIDEO_FILE="$1"
shift  # Remove first argument, rest are options

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "ERROR: Video file not found: $VIDEO_FILE"
    exit 1
fi

echo "Video file: $VIDEO_FILE"
echo ""
# Reminder about initial downloads
echo "⚠️  FIRST RUN NOTICE:"
echo "If this is your first time running with local models,"
echo "the analyser + planner will download 60GB+ of weights from HuggingFace." 
echo "This is a ONE-TIME download (depends on your connection)."
echo "Ensure you have sufficient disk space and allow extra time for the first run."
echo "Subsequent runs will be much faster!"
echo ""

# Activate virtual environment
source venv/bin/activate

# Get video filename without extension for output directory
BASENAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')
OUTPUT_DIR="${BASENAME}_ai_edited"

echo ""
echo "=========================================="
echo "Starting Production Processing"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Additional options: $@"
echo ""

# Run with production settings
ai-video-editor-local "$VIDEO_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  --torch-dtype auto \
  --log-level INFO \
  "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Processing Complete!"
    echo "=========================================="
    echo ""
    echo "Output location: $OUTPUT_DIR/"
    echo ""
    echo "Generated files:"
    echo "  📁 chunks/        - Video segments"
    echo "  📁 analysis/      - AI analysis reports"
    echo "  📁 plan/          - Editing plan (JSON + Markdown)"
    echo "  📁 new/           - Edited segments"
    echo "  🎬 final_video.mp4 - Your finished video!"
    echo ""
    echo "To view the final video:"
    echo "  open $OUTPUT_DIR/final_video.mp4"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Processing Failed"
    echo "=========================================="
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Model download interrupted (press Ctrl+C to cancel, then retry)"
    echo "  - Out of memory (try --torch-dtype float16 or smaller models)"
    echo "  - Disk space (need 25GB+ for models)"
    echo ""
fi

exit $EXIT_CODE
