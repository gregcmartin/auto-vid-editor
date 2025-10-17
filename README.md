# AI Video Editor

This project aims to take a single long-form source video and automatically craft an edited cut using ffmpeg driven by large language models.


## Ingredients

- ### MLX Models (Apple optimized)
- **Video analysis:** `mlx-community/Qwen3-VL-8B-Thinking-8bit`
- **Planning:** `Qwen/Qwen3-30B-A3B-MLX-8bit`

## Features

The CLI currently automates the entire post-production loop:

1. **Chunking** – Splits a long source video into evenly sized segments via `ffmpeg`.
2. **Chunk analysis** – Samples multiple frames per detected shot and routes them (plus transcripts/audio cues) through a JSON-locked video analyzer to describe every segment, highlight dull moments, and capture grounded people metadata into markdown.
3. **Director plan** – Summarises all chunk reports with Planning agent to produce a structured editing script (JSON + Markdown).
4. **ffmpeg execution** – Applies trims, optional crops, speed ramps, overlays, subtitles, and background music cues (mixed from the local royalty-free library) to produce polished chunk edits.
5. **Final assembly** – Concatenates the edited chunks into one final deliverable.
6. **Quality review** – Runs automatically after editing to flag pacing issues and capture a model-backed verdict (disable with `--no-quality-review`).

## Quick start

### This is only for Mac / OSX platform

Install dependencies (Python 3.11+ with `ffmpeg`/`ffprobe` on `PATH`):

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install the package (CLI + transcription extras)
pip install -e '.[transcription]'
```

Process a video end-to-end:

```bash
./run_production_local.sh /path/to/video.mp4 --parts 16
```

Or use the CLI directly:

```bash
ai-video-editor-local /path/to/video.mp4 \
  --output-dir ./workspace \
  --parts 16
```

Clean up between runs (optional):

```bash
./cleanup_artifacts.sh [--output-dir ./workspace]
```

### Quality review

The post-edit quality review runs by default: it analyses the rendered segments and final export, and can optionally consult a language-model “judge” for narrative feedback. By default the CLI reuses the planning backend; you can override it with `--quality-model Qwen/Qwen2.5-7B-Instruct` (or an MLX model such as `mlx-community/Qwen3-4B-Instruct-MLX`). Disable the review with `--no-quality-review`. The report is persisted as both JSON and Markdown in `quality/` alongside the final video.


## Development

- Run the unit tests (requires `pytest`): `pytest`
- Linting/formatting is left to your preference; the project stays black/ruff-compatible.
- `--dry-run` in the CLI writes placeholder files so you can validate the orchestration without real media or API calls.

## Troubleshooting

### Memory Issues on Mac
If you encounter memory errors:
1. Increase `--parts` to create smaller chunks (try 16 or 32)
2. Close other applications to free up RAM
3. Ensure you have at least 64GB RAM for the 30B model

### Model Download
First run will download models:
- Models are cached in `~/.cache/huggingface/`
- Download time: 30m minutes depending on connection
- Subsequent runs use cached models

### Analyzer keeps falling back
- Inspect `analysis/chunk_*_attempt*_raw.json` for the model output that failed validation.
- The pipeline now synthesises a minimal analysis if both analyzers disagree, so planning can continue, but tuning prompts or regenerating those chunks may improve quality.
