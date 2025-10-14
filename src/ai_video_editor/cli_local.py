from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .analysis.base import VideoAnalyzer
from .analysis.local_qwen_vl import LocalQwenVideoAnalyzer
from .analysis.mlx_qwen_vl import MLXQwenVideoAnalyzer
from .analysis.pipeline import AnalysisPipeline
from .config import AppConfig
from .editor import FFmpegEditor, VideoAssembler, FFmpegEditorError
from .editing_plan import EditingPlan
from .ffmpeg_utils import (
    ensure_encoder_available,
    probe_ffmpeg_encoders,
    select_video_encoder,
)
from .local_planner import LocalDirectorPlanner
from .mlx_planner import MLXDirectorPlanner
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
        "--scene-threshold",
        type=float,
        default=0.4,
        help="Scene detection threshold for analysis (default: 0.4).",
    )
    parser.add_argument(
        "--disable-scene-detection",
        action="store_true",
        help="Skip ffmpeg-based scene detection and analyse whole chunks.",
    )
    parser.add_argument(
        "--analysis-model",
        type=str,
        default="mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit",
        help="Model for video analysis (default: mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit).",
    )
    parser.add_argument(
        "--planner-model",
        default="Qwen/Qwen3-30B-A3B-MLX-8bit",
        help="Model for planning (default: Qwen/Qwen3-30B-A3B-MLX-8bit).",
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
        "--analysis-backend",
        default="auto",
        choices=["auto", "mlx", "torch"],
        help="Implementation to load the analysis model (default: auto).",
    )
    parser.add_argument(
        "--analysis-retries",
        type=int,
        default=1,
        help="Number of times to retry analysis if validation fails (default: 1).",
    )
    parser.add_argument(
        "--planner-backend",
        default="auto",
        choices=["auto", "mlx", "torch"],
        help="Implementation to load the planner model (default: auto).",
    )
    parser.add_argument(
        "--transcription-model",
        dest="transcription_model",
        default="small",
        help="Lightning Whisper MLX model size to use for transcription (default: small).",
    )
    parser.add_argument(
        "--whisper-model",
        dest="transcription_model",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-transcription",
        dest="enable_transcription",
        action="store_false",
        help="Disable audio transcription during analysis.",
    )
    parser.add_argument(
        "--enable-transcription",
        dest="enable_transcription",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(enable_transcription=True)
    parser.add_argument(
        "--disable-audio-analysis",
        action="store_true",
        help="Disable audio cue analysis during analysis.",
    )
    parser.add_argument(
        "--video-encoder",
        default="auto",
        choices=["auto", "libx264", "h264_videotoolbox", "hevc_videotoolbox"],
        help=(
            "Video encoder passed to ffmpeg (default auto: selects VideoToolbox on macOS if "
            "available, otherwise libx264)."
        ),
    )
    parser.add_argument(
        "--video-bitrate",
        default=None,
        help="Target video bitrate (e.g. 8M) for hardware encoders that do not support CRF.",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=None,
        help="Constant quality value (1-100) for VideoToolbox encoders (higher = better).",
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
    planner: Any,
    config: AppConfig,
    chunks: list[Path],
    music_library: List[str],
    skip: bool,
    fallback_factory: Optional[Callable[[], Any]] = None,
    fallback_label: Optional[str] = None,
) -> EditingPlan:
    if skip and config.plan_json.exists():
        logging.info("Skipping planning and loading existing plan: %s", config.plan_json)
        data = json.loads(config.plan_json.read_text())
        return EditingPlan.from_dict(data)
    try:
        plan = planner.plan(config.analysis_dir, chunks, music_library)
    except Exception as exc:
        if not fallback_factory:
            raise
        logging.warning(
            "Primary planner failed (%s). Retrying with %s backend.",
            exc,
            fallback_label or "fallback",
        )
        try:
            fallback_planner = fallback_factory()
        except Exception as fallback_exc:  # pragma: no cover - fallback init failure
            logging.error(
                "Failed to initialise fallback planner (%s).", fallback_exc, exc_info=True
            )
            raise
        plan = fallback_planner.plan(config.analysis_dir, chunks, music_library)
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
    missing_chunks = [name for name, path in chunk_lookup.items() if not path.exists()]
    if missing_chunks:
        raise FFmpegEditorError(f"Missing chunk files for editing: {', '.join(sorted(missing_chunks))}")

    if config.music_dir:
        missing_tracks = {
            cue.track
            for instruction in plan.instructions
            for cue in instruction.music_cues
            if cue.track and not (config.music_dir / cue.track).exists()
        }
        if missing_tracks:
            logging.warning(
                "Music tracks referenced in the plan were not found: %s",
                ", ".join(sorted(missing_tracks)),
            )

    progress_manifest = config.edited_dir / "render_manifest.json"
    rendered = editor.render(
        plan,
        chunk_lookup,
        config.edited_dir,
        progress_path=progress_manifest,
    )
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


def build_analyzer(args: argparse.Namespace) -> VideoAnalyzer:
    """Select the appropriate analysis backend based on CLI options."""
    backend = args.analysis_backend.lower()
    prefer_mlx = backend == "mlx"
    prefer_torch = backend == "torch"

    if backend == "auto":
        prefer_mlx = args.analysis_model.startswith("mlx-") or args.analysis_model.startswith(
            "mlx-community/"
        )
        if not prefer_mlx:
            # Default to MLX on Apple Silicon when torch MPS is likely available.
            import platform

            prefer_mlx = platform.system() == "Darwin"

    if prefer_mlx and not prefer_torch:
        try:
            logging.info("Using MLX backend for analysis model.")
            analyzer = MLXQwenVideoAnalyzer(model_name=args.analysis_model)
            analyzer.ensure_model_ready()
            return analyzer
        except ImportError as exc:
            logging.warning("Failed to initialise MLX backend (%s); falling back to torch.", exc)
        except Exception as exc:
            if backend == "mlx":
                logging.error("MLX backend initialisation failed: %s", exc, exc_info=True)
                raise
            logging.warning(
                "MLX backend initialisation failed (%s); falling back to torch.", exc, exc_info=True
            )

    torch_model_name = args.analysis_model
    if torch_model_name.startswith("mlx-community/"):
        original = torch_model_name.split("/", 1)[1]
        mlx_to_torch_map = {
            "Qwen2-VL-7B-Instruct-4bit": "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen3-VL-30B-A3B-Thinking-4bit": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        }
        torch_model_name = mlx_to_torch_map.get(original)
        if torch_model_name is None:
            torch_model_name = original
            if torch_model_name.endswith("-4bit"):
                torch_model_name = torch_model_name[:-5]
            torch_model_name = f"Qwen/{torch_model_name}"
        logging.info(
            "Requested model %s is MLX-specific; switching to torch-compatible %s",
            args.analysis_model,
            torch_model_name,
        )

    logging.info("Using torch backend for analysis model.")
    return LocalQwenVideoAnalyzer(
        model_name=torch_model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )


def build_planner(args: argparse.Namespace):
    backend = args.planner_backend.lower()
    prefer_mlx = backend == "mlx"
    prefer_torch = backend == "torch"

    def resolve_torch_model(name: str) -> str:
        if name.startswith("mlx-community/"):
            original = name.split("/", 1)[1]
            mlx_to_torch_map = {
                "Qwen3-4B-Instruct-2507-4bit-DWQ-2510": "Qwen/Qwen3-4B-Instruct",
                "Qwen3-14B-MLX-4bit": "Qwen/Qwen3-14B-Instruct",
                "GLM-4.5-Air-4bit": "THUDM/glm-4-9b-chat",
                "Qwen3-Next-80B-A3B-Instruct-4bit": "Qwen/Qwen3-72B-A3B-Instruct",
                "Qwen3-30B-A3B-MLX-8bit": "Qwen/Qwen3-30B-A3B-Instruct",
            }
            mapped = mlx_to_torch_map.get(original)
            if mapped:
                logging.info(
                    "Planner model %s is MLX-specific; switching fallback to torch-compatible %s",
                    name,
                    mapped,
                )
                return mapped
            name = original
            if name.endswith("-4bit"):
                name = name[:-5]
            return f"Qwen/{name}"
        return name

    if backend == "auto":
        prefer_mlx = "MLX" in args.planner_model.upper()

    if prefer_mlx and not prefer_torch:
        try:
            logging.info("Using MLX backend for planner model.")
            planner = MLXDirectorPlanner(model_name=args.planner_model)
            fallback_name = resolve_torch_model(args.planner_model)

            def fallback_factory() -> LocalDirectorPlanner:
                return LocalDirectorPlanner(
                    model_name=fallback_name,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                )

            return planner, "mlx", fallback_factory, "torch"
        except ImportError as exc:
            logging.warning(
                "Failed to initialise MLX planner backend (%s); falling back to torch.",
                exc,
            )
        except Exception as exc:
            if backend == "mlx":
                logging.error(
                    "MLX planner initialisation failed: %s", exc, exc_info=True
                )
                raise
            logging.warning(
                "MLX planner initialisation failed (%s); falling back to torch.",
                exc,
                exc_info=True,
            )

    planner_model_name = resolve_torch_model(args.planner_model)

    logging.info("Using torch backend for planner model.")
    planner = LocalDirectorPlanner(
        model_name=planner_model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    return planner, "torch", None, None


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    configure_logging(args.log_level)
    start_time = time.perf_counter()
    try:
        if args.video_quality is not None and not (1 <= args.video_quality <= 100):
            logging.error("Video quality must be between 1 and 100.")
            return 1

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

        available_encoders = probe_ffmpeg_encoders(config.ffmpeg)
        video_encoder = select_video_encoder(args.video_encoder, available_encoders)
        resolved_encoder = ensure_encoder_available(video_encoder, available_encoders)
        if args.video_encoder != "auto" and resolved_encoder != args.video_encoder:
            logging.warning(
                "Requested video encoder '%s' unavailable. Falling back to %s.",
                args.video_encoder,
                resolved_encoder,
            )
        video_encoder = resolved_encoder

        video_quality = args.video_quality
        video_bitrate = args.video_bitrate
        if video_quality is not None and video_bitrate is not None:
            logging.warning(
                "Both video quality and bitrate specified; using quality and ignoring bitrate."
            )
            video_bitrate = None
        if video_quality is not None and video_encoder not in {"h264_videotoolbox", "hevc_videotoolbox"}:
            logging.warning(
                "Selected encoder %s does not support constant quality; ignoring --video-quality.",
                video_encoder,
            )
            video_quality = None

        logging.info("Output directory: %s", config.root_output)
        logging.info(
            "Using local models - Analysis: %s, Planner: %s",
            args.analysis_model,
            args.planner_model,
        )
        logging.info("Video encoder: %s", video_encoder)

        splitter = VideoSplitter(ffmpeg=config.ffmpeg, ffprobe=config.ffprobe)
        try:
            chunks = splitter.split(video_path, config.chunks_dir, parts=args.parts)
        except (VideoSplitterError, ValueError, FileNotFoundError) as exc:
            logging.error("Video splitting failed: %s", exc)
            return 1

        try:
            analyzer = build_analyzer(args)
            backend_name = "mlx" if isinstance(analyzer, MLXQwenVideoAnalyzer) else "torch"
            logging.info("Analysis backend selected: %s", backend_name)
        except Exception as exc:
            logging.error("Failed to initialise analysis model: %s", exc)
            return 1
        fallback_analyzer = None
        if isinstance(analyzer, MLXQwenVideoAnalyzer):
            fallback_analyzer = LocalQwenVideoAnalyzer(
                model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                device="cpu",
                torch_dtype="float32",
            )
        pipeline = AnalysisPipeline(
            analyzer=analyzer,
            analysis_dir=config.analysis_dir,
            people_report_path=config.people_report,
            enable_scene_detection=not args.disable_scene_detection,
            scene_threshold=args.scene_threshold,
            enable_transcription=args.enable_transcription,
            whisper_model=args.transcription_model,
            enable_audio_analysis=not args.disable_audio_analysis,
            analysis_retries=max(0, args.analysis_retries),
            fallback_analyzer=fallback_analyzer,
        )

        try:
            ensure_analysis(pipeline, chunks, args.skip_analysis)
        except Exception as exc:
            logging.error("Analysis pipeline failed: %s", exc)
            return 1

        music_library = discover_music_tracks(config.music_dir)
        planner, planner_backend, planner_fallback_factory, planner_fallback_label = build_planner(args)
        logging.info("Planner backend selected: %s", planner_backend)

        try:
            plan = ensure_plan(
                planner,
                config,
                chunks,
                music_library,
                args.skip_planning,
                fallback_factory=planner_fallback_factory,
                fallback_label=planner_fallback_label,
            )
        except Exception as exc:
            logging.error("Planning failed: %s", exc)
            return 1

        editor = FFmpegEditor(
            ffmpeg=config.ffmpeg,
            font_path=config.font_path,
            music_dir=config.music_dir,
            ffprobe=config.ffprobe,
            dry_run=False,
            video_encoder=video_encoder,
            video_bitrate=video_bitrate,
            video_quality=video_quality,
        )
        assembler = VideoAssembler(ffmpeg=config.ffmpeg, dry_run=False)

        try:
            ensure_editing(editor, assembler, plan, chunks, config, args.skip_editing)
        except (FFmpegEditorError, Exception) as exc:
            logging.error("Editing/assembly failed: %s", exc)
            return 1

        logging.info("Processing completed. Output located in %s", config.root_output)
        return 0
    finally:
        elapsed = time.perf_counter() - start_time
        logging.info("Total execution time: %.2f seconds", elapsed)


if __name__ == "__main__":
    sys.exit(main())
