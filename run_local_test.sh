#!/bin/bash

# AI Video Editor - Local Model Test Script
# This script will download models (~30GB) on first run

echo "==================================="
echo "AI Video Editor - Local Model Test"
echo "==================================="
echo ""
echo "This will:"
echo "1. Download Qwen2-VL-7B-Instruct (~15GB) - for video analysis"
echo "2. Download Qwen2.5-7B-Instruct (~15GB) - for planning"
echo "3. Process test_video.mp4 with 1 chunk"
echo ""
echo "First run will take 10-30 minutes depending on internet speed."
echo "Subsequent runs will be much faster (models are cached)."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run with local models
echo ""
echo "Starting AI Video Editor with local models..."
echo "Press Ctrl+C to cancel model download if needed"
echo ""

ai-video-editor-local test_video.mp4 \
  --output-dir ./local_test_output \
  --parts 1 \
  --device auto \
  --torch-dtype auto \
  --log-level INFO

echo ""
echo "==================================="
echo "Processing complete!"
echo "Check ./local_test_output/ for results"
echo "==================================="
