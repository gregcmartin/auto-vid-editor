# AI Video Editor - Setup Complete! ✅

## Installation Summary

The AI Video Editor is now fully installed and operational on your system.

### What Was Installed

1. **Python Virtual Environment** (Python 3.11.12)
   - Located at: `./venv/`
   - Activated with: `source venv/bin/activate`

2. **Dependencies Installed**
   - `dashscope>=1.14.0` - For Qwen model API access
   - `qwen-agent>=0.0.10` - Agent framework
   - All required dependencies (aiohttp, requests, pydantic, etc.)

3. **System Requirements Verified**
   - ✅ ffmpeg 7.1.1 (installed and working)
   - ✅ ffprobe 7.1.1 (installed and working)
   - ✅ Python 3.11.12 (installed and working)

4. **Test Run Completed**
   - Created test video (30 seconds, 1280x720)
   - Successfully ran dry-run mode
   - Generated complete output structure

## Quick Start Guide

### 1. Activate the Virtual Environment

```bash
source venv/bin/activate
```

### 2. Set Your API Key

You need a DashScope API key to use the Qwen models:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

Or pass it directly with `--api-key` flag.

### 3. Basic Usage

**Process a video with dry-run (no API calls):**
```bash
ai-video-editor your_video.mp4 --dry-run
```

**Process a video with AI analysis (requires API key):**
```bash
ai-video-editor your_video.mp4 \
  --output-dir ./output \
  --parts 4 \
  --analysis-model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --planner-model Qwen3-30B-A3B
```

**Use the file picker to select a video:**
```bash
ai-video-editor --use-file-picker
```

### 4. Key Options

- `--parts N` - Split video into N chunks (default: 4)
- `--output-dir PATH` - Where to save outputs (default: beside video)
- `--music-dir PATH` - Directory with background music (default: ./music)
- `--dry-run` - Test without API calls or ffmpeg execution
- `--skip-analysis` - Reuse existing analysis
- `--skip-planning` - Reuse existing plan
- `--skip-editing` - Skip ffmpeg editing
- `--log-level DEBUG` - Increase verbosity

### 5. Output Structure

When you run the tool, it creates:

```
output_directory/
├── chunks/           # Split video segments
│   ├── chunk_01.mp4
│   ├── chunk_02.mp4
│   └── ...
├── analysis/         # AI analysis reports
│   ├── chunk_01.md
│   ├── chunk_02.md
│   └── people.md
├── plan/            # Editing plan
│   ├── plan.json
│   └── plan.md
├── new/             # Edited segments
│   ├── 01_Hook.mp4
│   └── ...
└── final_video.mp4  # Final assembled video
```

### 6. Music Library

The tool automatically discovers music files in the `./music/` directory:
- Supported formats: MP3, WAV, M4A
- Current tracks available:
  - ambient-background-loop-chill.mp3
  - ambient-background-music-uplifting.mp3
  - fun-easy-islandy-loop-925bpm.mp3
  - hardcore-techno-pumping-loop.mp3
  - loop-black-box-exciting-bass-loop-130bpm-13888.mp3
  - mellow-edm-trap-beat-loop.mp3
  - nostalgia-melody-loop.mp3

### 7. Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Set API key
export DASHSCOPE_API_KEY="your-key"

# 3. Process a video
ai-video-editor my_video.mp4 \
  --output-dir ./my_edit \
  --parts 6 \
  --music-dir ./music

# 4. Review the plan
cat ./my_edit/plan/plan.md

# 5. If you want to re-edit with different settings, skip analysis/planning
ai-video-editor my_video.mp4 \
  --output-dir ./my_edit \
  --skip-analysis \
  --skip-planning
```

### 8. Testing Without API Key

You can test the entire workflow without an API key using `--dry-run`:

```bash
ai-video-editor test_video.mp4 --dry-run --parts 2
```

This generates placeholder data and shows you the complete pipeline.

## What the Tool Does

1. **Chunking** - Splits your video into equal segments
2. **Analysis** - Uses Qwen3-VL vision model to analyze each chunk:
   - Timeline of events
   - Dull sections to trim
   - People detection and tracking
3. **Planning** - Uses Qwen3 text model to create editing plan:
   - Trim points
   - Speed adjustments
   - Text overlays
   - Subtitles
   - Music cues
4. **Editing** - Applies plan with ffmpeg:
   - Crops, speed changes
   - Text overlays and subtitles
   - Background music mixing
5. **Assembly** - Concatenates edited chunks into final video

## Troubleshooting

**If you get "command not found: ai-video-editor":**
```bash
source venv/bin/activate
```

**If you get API errors:**
- Check your DASHSCOPE_API_KEY is set correctly
- Verify you have API credits
- Try with `--dry-run` first to test the pipeline

**If ffmpeg fails:**
- Check ffmpeg is in your PATH: `which ffmpeg`
- Verify video file is valid: `ffprobe your_video.mp4`

## Next Steps

1. Get a DashScope API key from: https://dashscope.aliyun.com/
2. Try processing a real video
3. Experiment with different `--parts` values
4. Add your own music to the `./music/` directory
5. Review and customize the generated plans

## Documentation

- README.md - Project overview
- Source code in `src/ai_video_editor/`
- Tests in `tests/`

Enjoy automated video editing! 🎬
