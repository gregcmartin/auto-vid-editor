# Local Models Guide - AI Video Editor

This guide explains how to run the AI Video Editor with **local models from HuggingFace** instead of using the DashScope API.

## Why Local Models?

- ✅ **No API costs** - Run everything on your own hardware
- ✅ **Privacy** - Your videos never leave your machine
- ✅ **Offline capability** - No internet required after downloading models
- ✅ **Full control** - Choose any compatible model from HuggingFace

## Installation

The local model dependencies are already installed! They include:

- **PyTorch 2.8.0** - Deep learning framework
- **Transformers 4.57.0** - HuggingFace model library
- **Accelerate 1.10.1** - Efficient model loading
- **qwen-vl-utils** - Video processing utilities

## Hardware Requirements

### Minimum (CPU-only)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for models
- **Speed**: Slow but functional

### Recommended (GPU)
- **GPU**: Apple Silicon (M1/M2/M3), NVIDIA GPU with 8GB+ VRAM, or AMD GPU
- **RAM**: 16GB+
- **Storage**: 20GB+ for models
- **Speed**: Much faster inference

## Quick Start

### 1. Basic Usage

Use the `ai-video-editor-local` command instead of `ai-video-editor`:

```bash
source venv/bin/activate
ai-video-editor-local your_video.mp4
```

### 2. First Run - Model Download

The first time you run it, the models will be downloaded from HuggingFace:

- **Qwen2-VL-7B-Instruct** (~15GB) - For video analysis
- **Qwen2.5-7B-Instruct** (~15GB) - For planning

This is a one-time download. Models are cached in `~/.cache/huggingface/`

### 3. Choose Your Device

**Automatic (recommended):**
```bash
ai-video-editor-local your_video.mp4 --device auto
```

**Force CPU:**
```bash
ai-video-editor-local your_video.mp4 --device cpu
```

**Force GPU (CUDA):**
```bash
ai-video-editor-local your_video.mp4 --device cuda
```

**Apple Silicon (MPS):**
```bash
ai-video-editor-local your_video.mp4 --device mps
```

### 4. Optimize Memory Usage

**Use float16 (saves memory, slight quality loss):**
```bash
ai-video-editor-local your_video.mp4 --torch-dtype float16
```

**Use bfloat16 (best for modern GPUs):**
```bash
ai-video-editor-local your_video.mp4 --torch-dtype bfloat16
```

**Use float32 (highest quality, most memory):**
```bash
ai-video-editor-local your_video.mp4 --torch-dtype float32
```

## Available Models

### Video Analysis Models

**⚠️ IMPORTANT: This project ONLY uses `Qwen/Qwen3-VL-30B-A3B-Instruct` for video analysis. No exceptions!**

Default: `Qwen/Qwen3-VL-30B-A3B-Instruct` (~30GB)

This is the ONLY supported model for video analysis. Do not use other models.

```bash
# This is the default - no need to specify
ai-video-editor-local your_video.mp4
```

### Planning Models

**⚠️ IMPORTANT: This project ONLY uses `Qwen/Qwen3-30B-A3B` for planning. No exceptions!**

Default: `Qwen/Qwen3-30B-A3B` (~30GB)

This is the ONLY supported model for planning. Do not use other models.

```bash
# This is the default - no need to specify
ai-video-editor-local your_video.mp4
```

## Complete Example

```bash
# Activate environment
source venv/bin/activate

# Process video with local models
ai-video-editor-local my_video.mp4 \
  --output-dir ./my_edit \
  --parts 4 \
  --analysis-model Qwen/Qwen2-VL-7B-Instruct \
  --planner-model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --torch-dtype auto \
  --music-dir ./music \
  --log-level INFO
```

## Performance Tips

### 1. Reduce Video Parts
Fewer chunks = fewer model calls = faster processing:
```bash
ai-video-editor-local video.mp4 --parts 2
```

### 2. Use Smaller Models
Trade quality for speed:
```bash
ai-video-editor-local video.mp4 \
  --analysis-model Qwen/Qwen2-VL-2B-Instruct \
  --planner-model Qwen/Qwen2.5-3B-Instruct
```

### 3. Skip Steps
Reuse previous analysis/planning:
```bash
# First run
ai-video-editor-local video.mp4 --skip-editing

# Review the plan, then run editing
ai-video-editor-local video.mp4 --skip-analysis --skip-planning
```

### 4. Optimize for Your Hardware

**For Apple Silicon (M1/M2/M3):**
```bash
ai-video-editor-local video.mp4 \
  --device mps \
  --torch-dtype float16
```

**For NVIDIA GPU:**
```bash
ai-video-editor-local video.mp4 \
  --device cuda \
  --torch-dtype bfloat16
```

**For CPU-only:**
```bash
ai-video-editor-local video.mp4 \
  --device cpu \
  --torch-dtype float32 \
  --parts 2
```

## Command Reference

```
ai-video-editor-local [VIDEO] [OPTIONS]

Required:
  VIDEO                 Path to input video file

Model Options:
  --analysis-model      HuggingFace model for video analysis
                        (default: Qwen/Qwen2-VL-7B-Instruct)
  --planner-model       HuggingFace model for planning
                        (default: Qwen/Qwen2.5-7B-Instruct)
  --device              Device: auto, cuda, cpu, mps
                        (default: auto)
  --torch-dtype         Precision: auto, float32, float16, bfloat16
                        (default: auto)

Processing Options:
  --output-dir DIR      Output directory
  --parts N             Number of chunks (default: 4)
  --music-dir DIR       Music library directory
  --font-path PATH      Font for text overlays

Control Options:
  --skip-analysis       Reuse existing analysis
  --skip-planning       Reuse existing plan
  --skip-editing        Skip ffmpeg editing
  --use-file-picker     Open file picker dialog
  --log-level LEVEL     DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Troubleshooting

### Out of Memory Errors

**Solution 1: Use smaller models**
```bash
ai-video-editor-local video.mp4 \
  --analysis-model Qwen/Qwen2-VL-2B-Instruct \
  --planner-model Qwen/Qwen2.5-3B-Instruct
```

**Solution 2: Use lower precision**
```bash
ai-video-editor-local video.mp4 --torch-dtype float16
```

**Solution 3: Force CPU**
```bash
ai-video-editor-local video.mp4 --device cpu
```

**Solution 4: Process fewer chunks**
```bash
ai-video-editor-local video.mp4 --parts 2
```

### Slow Performance

1. **Check if GPU is being used:**
   - Look for "Loading local model" messages in logs
   - Should say "cuda" or "mps" if using GPU

2. **Use smaller models** (see above)

3. **Reduce video parts** (see above)

### Model Download Issues

**Check your internet connection** - Models are large (15GB each)

**Manually download models:**
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# This will download and cache the models
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
```

**Check cache location:**
```bash
ls ~/.cache/huggingface/hub/
```

## Comparison: Local vs API

| Feature | Local Models | DashScope API |
|---------|-------------|---------------|
| **Cost** | Free (after hardware) | Pay per use |
| **Privacy** | Complete | Data sent to cloud |
| **Speed** | Depends on hardware | Fast (cloud GPUs) |
| **Setup** | Download models once | Just API key |
| **Offline** | Yes | No |
| **Model Choice** | Any HF model | Limited options |

## Advanced: Custom Models

You can use any compatible HuggingFace model:

```bash
ai-video-editor-local video.mp4 \
  --analysis-model "your-username/your-vision-model" \
  --planner-model "your-username/your-text-model"
```

Requirements:
- Vision model must support video input
- Text model must support chat format
- Both must be compatible with transformers library

## Next Steps

1. **Try it out** with a short test video
2. **Experiment** with different models and settings
3. **Monitor** memory usage and adjust accordingly
4. **Optimize** for your specific hardware

For more help, see:
- Main README.md
- SETUP_COMPLETE.md
- HuggingFace model cards for specific models
