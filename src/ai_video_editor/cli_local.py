from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .analysis.nexa_qwen_vl import NexaQwenVideoAnalyzer
from .analysis.pipeline import AnalysisPipeline
from .config import AppConfig
from .editor import FFmpegEditor, VideoAssembler, FFmpegEditorError
from .editing_plan import EditingPlan
from .local_planner import LocalDirectorPlanner
from .video_splitter import VideoSplitter, VideoSplitterError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-video-editor-local",
        description="Automate video editing with local AI models from HuggingFace.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for generated assets. Defaults beside the video file.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=4,
        help="Number of equal video segments to produce (default: 4).",
    )
    parser.add_argument(
        "--analysis-model",
        type=str,
        default="NexaAI/qwen3vl-30B-A3B-mlx",
        help="NexaAI MLX model for video analysis (default: NexaAI/qwen3vl-30B-A3B-mlx).",
    )
    parser.add_argument(
        "--planner-model",
        default="Qwen/Qwen3-30B-A3B-MLX-8bit",
        help="HuggingFace model for planning (default: Qwen/Qwen3-30B-A3B-MLX-8bit).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model inference: 'auto', 'cuda', 'cpu', etc. (default: auto).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for models (default: auto - bfloat16 on GPU, float32 on CPU).",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=None,
        help="Location for the final assembled video (default: <output>/final_video.mp4).",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Font file to use for text overlays (defaults to macOS Arial).",
    )
    parser.add_argument(
        "--music-dir",
        type=Path,
        default=None,
        help="Directory containing royalty-free music tracks for background cues.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--use-file-picker",
        action="store_true",
        help="Open a file picker dialog to select the video path.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Reuse existing chunk analysis files if they already exist.",
    )
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="Skip the planning step and reuse an existing plan.json if present.",
    )
    parser.add_argument(
        "--skip-editing",
        action="store_true",
        help="Skip ffmpeg editing and final assembly (useful for analysis-only runs).",
    )
    return parser


def resolve_video_path(arg_path: Optional[str], use_picker: bool) -> Path:
    if use_picker:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("File picker unavailable: tkinter not present.") from exc

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select source video",
            filetypes=[("Video files", ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.*"))],
        )
        root.destroy()
        if not file_path:
            raise RuntimeError("No video selected in file picker.")
        return Path(file_path).expanduser().resolve()

    if not arg_path:
        raise ValueError("Video path is required when not using the file picker.")
    return Path(arg_path).expanduser().resolve()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_analysis(pipeline: AnalysisPipeline, chunks: list[Path], skip: bool) -> None:
    if skip:
        logging.info("Skipping analysis as requested.")
        return
    pipeline.run(chunks)


def ensure_plan(
    planner: LocalDirectorPlanner,
    config: AppConfig,
    chunks: list[Path],
    music_library: List[str],
    skip: bool,
) -> EditingPlan:
    if skip and config.plan_json.exists():
        logging.info("Skipping planning and loading existing plan: %s", config.plan_json)
        data = json.loads(config.plan_json.read_text())
        return EditingPlan.from_dict(data)

    plan = planner.plan(config.analysis_dir, chunks, music_library)
    config.plan_dir.mkdir(parents=True, exist_ok=True)
    config.plan_json.write_text(json.dumps(plan.to_dict(), indent=2))
    config.plan_markdown.write_text(plan.to_markdown())
    logging.info("Saved plan to %s", config.plan_json)
    return plan


def ensure_editing(
    editor: FFmpegEditor,
    assembler: VideoAssembler,
    plan: EditingPlan,
    chunks: list[Path],
    config: AppConfig,
    skip: bool,
) -> None:
    if skip:
        logging.info("Skipping editing/assembly as requested.")
        return

    chunk_lookup = {chunk.name: chunk for chunk in chunks}
    rendered = editor.render(plan, chunk_lookup, config.edited_dir)
    assembler.assemble(rendered, config.final_output)
    logging.info("Final video created at %s", config.final_output)


def discover_music_tracks(music_dir: Optional[Path]) -> List[str]:
    if not music_dir:
        return []
    music_dir = music_dir.expanduser()
    if not music_dir.exists():
        logging.warning("Music directory not found: %s", music_dir)
        return []
    tracks = sorted(
        path.name
        for path in music_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a"}
    )
    if not tracks:
        logging.warning("No music files discovered in %s", music_dir)
    else:
        logging.debug("Discovered music tracks: %s", ", ".join(tracks))
    return tracks


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        video_path = resolve_video_path(args.video, args.use_file_picker)
    except Exception as exc:
        logging.error("Failed to determine video path: %s", exc)
        return 1

    if not video_path.exists():
        logging.error("Video path not found: %s", video_path)
        return 1

    root_output = args.output_dir or video_path.parent / "ai_video_editor_output"
    default_music_dir = args.music_dir or Path(__file__).resolve().parent.parent.parent / "music"
    default_music_dir = default_music_dir.expanduser()
    config = AppConfig.from_paths(
        video_path=video_path,
        root_output=root_output,
        font_path=args.font_path,
        music_dir=default_music_dir,
    )

    if args.final_output:
        config.final_output = args.final_output

    logging.info("Output directory: %s", config.root_output)
    logging.info("Using local models - Analysis: %s, Planner: %s", 
                 args.analysis_model, args.planner_model)

    splitter = VideoSplitter(ffmpeg=config.ffmpeg, ffprobe=config.ffprobe)
    try:
        chunks = splitter.split(video_path, config.chunks_dir, parts=args.parts)
    except (VideoSplitterError, ValueError, FileNotFoundError) as exc:
        logging.error("Video splitting failed: %s", exc)
        return 1

    analyzer = NexaQwenVideoAnalyzer(
        model_name=args.analysis_model,
    )
    pipeline = AnalysisPipeline(
        analyzer=analyzer,
        analysis_dir=config.analysis_dir,
        people_report_path=config.people_report,
    )

    try:
        ensure_analysis(pipeline, chunks, args.skip_analysis)
    except Exception as exc:
        logging.error("Analysis pipeline failed: %s", exc)
        return 1

    music_library = discover_music_tracks(config.music_dir)
    planner = LocalDirectorPlanner(
        model_name=args.planner_model,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    try:
        plan = ensure_plan(planner, config, chunks, music_library, args.skip_planning)
    except Exception as exc:
        logging.error("Planning failed: %s", exc)
        return 1

    editor = FFmpegEditor(
        ffmpeg=config.ffmpeg,
        font_path=config.font_path,
        music_dir=config.music_dir,
        ffprobe=config.ffprobe,
        dry_run=False,
    )
    assembler = VideoAssembler(ffmpeg=config.ffmpeg, dry_run=False)

    try:
        ensure_editing(editor, assembler, plan, chunks, config, args.skip_editing)
    except (FFmpegEditorError, Exception) as exc:
        logging.error("Editing/assembly failed: %s", exc)
        return 1

    logging.info("Processing completed. Output located in %s", config.root_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
