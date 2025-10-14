from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .editing_plan import (
    EditInstruction,
    EditingPlan,
    MusicCue,
    SubtitleLine,
    TextOverlay,
)

logger = logging.getLogger(__name__)


class FFmpegEditorError(RuntimeError):
    pass


class FFmpegEditor:
    """Convert a planning script into ffmpeg commands."""

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        font_path: Optional[Path] = None,
        music_dir: Optional[Path] = None,
        ffprobe: str = "ffprobe",
        dry_run: bool = False,
        max_retries: int = 2,
        video_encoder: str = "libx264",
        video_bitrate: Optional[str] = None,
        video_quality: Optional[int] = None,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.font_path = font_path
        self.music_dir = music_dir
        self.ffprobe = ffprobe
        self.dry_run = dry_run
        self.max_retries = max(1, max_retries)
        self.video_encoder = video_encoder
        self.video_bitrate = video_bitrate
        self.video_quality = video_quality
        self._x264_preset = "veryfast"
        self._x264_crf = 18

    def render(
        self,
        plan: EditingPlan,
        chunk_lookup: dict[str, Path],
        output_dir: Path,
        *,
        progress_path: Optional[Path] = None,
    ) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered_paths: List[Path] = []

        progress: dict[str, dict[str, object]] = {}
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text())
                    if not isinstance(progress, dict):
                        progress = {}
                except Exception:
                    progress = {}

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            for index, instruction in enumerate(plan.instructions, start=1):
                logger.info("Rendering instruction %d/%d", index, len(plan.instructions))
                output_path = output_dir / self._build_output_name(index, instruction)
                progress_entry = progress.get(output_path.name) if progress_path else None
                if (
                    progress_entry
                    and progress_entry.get("completed")
                    and output_path.exists()
                ):
                    logger.info(
                        "Skipping already-rendered segment %s (progress manifest)",
                        output_path.name,
                    )
                    rendered_paths.append(output_path)
                    continue

                duration = instruction.in_end - instruction.in_start
                if duration <= 0:
                    raise FFmpegEditorError(
                        f"Instruction {index} has non-positive duration ({duration})."
                    )

                needs_music = bool(instruction.music_cues)
                intermediate_path = (
                    temp_dir / f"{output_path.stem}_base.mp4" if needs_music else output_path
                )

                subtitle_path = (
                    self._write_subtitles(temp_dir, index, instruction.subtitles)
                    if instruction.subtitles
                    else None
                )
                cmd = self._build_command(
                    instruction=instruction,
                    chunk_lookup=chunk_lookup,
                    output_path=intermediate_path,
                    subtitle_path=subtitle_path,
                )
                rendered_paths.append(output_path)
                self._run_ffmpeg(cmd, intermediate_path, purpose=f"segment {index} render")

                if needs_music:
                    self._mix_music(
                        base_video=intermediate_path,
                        final_output=output_path,
                        cues=instruction.music_cues,
                        include_source_audio=instruction.keep_audio,
                        segment_duration=duration,
                    )
                    if not self.dry_run and intermediate_path.exists():
                        intermediate_path.unlink()

                if progress_path is not None:
                    progress[output_path.name] = {
                        "chunk": instruction.chunk,
                        "completed": True,
                        "timestamp": time.time(),
                    }
                    progress_path.write_text(json.dumps(progress, indent=2))

        return rendered_paths

    def _build_output_name(self, index: int, instruction: EditInstruction) -> str:
        label = instruction.label or instruction.chunk.replace(".", "_")
        safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("_", "-"))
        return f"{index:02d}_{safe_label}.mp4"

    def _write_subtitles(
        self,
        temp_dir: Path,
        index: int,
        subtitles: Sequence[SubtitleLine],
    ) -> Path:
        path = temp_dir / f"segment_{index:02d}.srt"
        lines = []
        for idx, line in enumerate(subtitles, start=1):
            lines.append(str(idx))
            lines.append(
                f"{self._format_ts(line.start)} --> {self._format_ts(line.end)}"
            )
            lines.append(line.text)
            lines.append("")
        path.write_text("\n".join(lines).strip() + "\n")
        return path

    @staticmethod
    def _format_ts(seconds: float) -> str:
        millis = int(round(seconds * 1000))
        hours, remainder = divmod(millis, 3600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def _build_command(
        self,
        instruction: EditInstruction,
        chunk_lookup: dict[str, Path],
        output_path: Path,
        subtitle_path: Optional[Path],
    ) -> List[str]:
        if instruction.chunk not in chunk_lookup:
            raise FFmpegEditorError(f"Chunk not found for instruction: {instruction.chunk}")

        chunk_path = chunk_lookup[instruction.chunk]
        video_filters = self._build_video_filters(instruction, subtitle_path)
        audio_options = self._build_audio_options(instruction)

        duration = instruction.in_end - instruction.in_start
        if duration <= 0:
            raise FFmpegEditorError(
                f"Invalid trim window for {instruction.chunk}: start={instruction.in_start}, end={instruction.in_end}"
            )

        cmd: List[str] = [
            self.ffmpeg,
            "-y",
            "-i",
            str(chunk_path),
            "-ss",
            f"{instruction.in_start:.3f}",
            "-t",
            f"{duration:.3f}",
        ]

        if video_filters:
            cmd += ["-vf", video_filters]

        if instruction.keep_audio:
            if audio_options:
                cmd += audio_options
        else:
            cmd += ["-an"]

        encoder = self.video_encoder or "libx264"
        cmd += ["-c:v", encoder]
        if encoder == "libx264":
            cmd += [
                "-preset",
                self._x264_preset,
                "-crf",
                str(self._x264_crf),
            ]
        else:
            if self.video_quality is not None:
                cmd += ["-q:v", str(self.video_quality)]
            if self.video_bitrate:
                cmd += ["-b:v", self.video_bitrate]
            elif self.video_quality is None:
                # Sensible default bitrate to avoid under-encoding when hardware acceleration is used.
                cmd += ["-b:v", "8M"]

        if instruction.keep_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]

        cmd.append(str(output_path))
        return cmd

    def _build_video_filters(
        self, instruction: EditInstruction, subtitle_path: Optional[Path]
    ) -> str:
        filters: List[str] = []

        if instruction.crop:
            crop = instruction.crop
            filters.append(
                f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}"
            )

        if instruction.speed != 1.0:
            speed = instruction.speed
            if speed <= 0:
                raise FFmpegEditorError("Speed must be positive.")
            filters.append(f"setpts=PTS/{speed}")

        for overlay in instruction.text_overlays:
            filters.append(self._drawtext_filter(overlay))

        if subtitle_path:
            filters.append(f"subtitles={self._escape_path(subtitle_path)}")

        if not filters:
            return ""
        filters.append("format=yuv420p")
        return ",".join(filters)

    def _build_audio_options(self, instruction: EditInstruction) -> List[str]:
        if instruction.speed == 1.0:
            return []

        speed = instruction.speed
        filters: List[str] = []
        remaining = speed
        # atempo supports 0.5-2.0; chain if outside range.
        while remaining < 0.5 or remaining > 2.0:
            part = 2.0 if remaining > 2.0 else 0.5
            filters.append(f"atempo={part}")
            remaining /= part
        filters.append(f"atempo={remaining:.3f}")
        return ["-af", ",".join(filters)]

    def _drawtext_filter(self, overlay: TextOverlay) -> str:
        font = self.font_path or Path("/System/Library/Fonts/Supplemental/Arial.ttf")
        position = self._position_expr(overlay.position)
        enable = f"between(t,{overlay.start},{overlay.end})"
        sanitized_text = overlay.text.replace(":", r"\:").replace("'", r"\\'")
        return (
            "drawtext="
            f"fontfile={self._escape_path(font)}:"
            f"text='{sanitized_text}':"
            f"x={position[0]}:y={position[1]}:"
            "fontcolor=white:bordercolor=black:borderw=2:"
            "fontsize=36:"
            f"enable='{enable}'"
        )

    @staticmethod
    def _escape_path(path: Path) -> str:
        return shlex.quote(str(path))

    @staticmethod
    def _position_expr(position: str) -> tuple[str, str]:
        position = (position or "center").lower()
        mapping = {
            "center": ("(w-text_w)/2", "(h-text_h)/2"),
            "top_left": ("10", "10"),
            "top_right": ("w-text_w-10", "10"),
            "bottom_left": ("10", "h-text_h-10"),
            "bottom_right": ("w-text_w-10", "h-text_h-10"),
        }
        return mapping.get(position, mapping["center"])

    def _run_ffmpeg(self, cmd: Sequence[str], output_path: Path, *, purpose: str) -> None:
        self._execute_ffmpeg(cmd, output_path, purpose=purpose)

    def _mix_music(
        self,
        base_video: Path,
        final_output: Path,
        cues: Sequence[MusicCue],
        include_source_audio: bool,
        segment_duration: float,
    ) -> None:
        if not cues:
            self._copy_video(base_video, final_output)
            return
        if self.music_dir is None:
            raise FFmpegEditorError("Music directory not configured for music cues.")

        if self.dry_run:
            logger.info("Dry-run music mix for %s", final_output.name)
            final_output.write_text("dry-run music mix\n")
            return

        include_source_audio = include_source_audio and self._has_audio_stream(base_video)

        cmd: List[str] = [self.ffmpeg, "-y", "-i", str(base_video)]

        filter_cmds: List[str] = []
        mix_inputs: List[str] = []

        if include_source_audio:
            filter_cmds.append("[0:a]asetpts=N/SR/TB[a0]")
            mix_inputs.append("[a0]")

        for idx, cue in enumerate(cues, start=1):
            try:
                music_path = self._resolve_music_track(cue.track)
            except FFmpegEditorError as exc:
                logger.warning("Skipping music cue %s: %s", cue.track, exc)
                continue
            cmd += ["-i", str(music_path)]

            start = max(0.0, cue.start)
            end = min(segment_duration, cue.end)
            if end <= start:
                continue
            duration = end - start
            label_out = f"[m{idx}]"
            chain = f"[{idx}:a]atrim=0:{duration:.3f},asetpts=N/SR/TB"
            if cue.volume and cue.volume != 1.0:
                chain += f",volume={cue.volume:.3f}"
            delay_ms = max(0, int(round(start * 1000)))
            chain += f",adelay={delay_ms}|{delay_ms}{label_out}"
            filter_cmds.append(chain)
            mix_inputs.append(label_out)

        if not mix_inputs:
            logger.warning("No valid music cues applied; copying base video.")
            self._copy_video(base_video, final_output)
            return

        mix_chain = "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0[aout]"
        filter_cmds.append(mix_chain)
        filter_cmds = [cmd for cmd in filter_cmds if cmd and cmd.strip()]
        filter_complex = ";".join(filter_cmds)
        if not filter_complex.strip():
            logger.warning("Constructed empty filter graph; copying base video.")
            self._copy_video(base_video, final_output)
            return

        logger.debug("Music mix filter graph: %s", filter_complex)

        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(final_output),
        ]

        self._execute_ffmpeg(cmd, final_output, purpose="music mix")

    def _resolve_music_track(self, track: str) -> Path:
        if self.music_dir is None:
            raise FFmpegEditorError("Music directory not configured.")
        path = (self.music_dir / track).expanduser()
        if not path.exists():
            raise FFmpegEditorError(
                f"Music track '{track}' not found in {self.music_dir}"
            )
        return path

    def _copy_video(self, source: Path, destination: Path) -> None:
        if self.dry_run:
            destination.write_text("dry-run copy\n")
            return
        shutil.copy2(source, destination)

    def _has_audio_stream(self, video_path: Path) -> bool:
        if self.dry_run:
            return True
        cmd = [
            self.ffprobe,
            "-loglevel",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    def _execute_ffmpeg(self, cmd: Sequence[str], output_path: Path, *, purpose: str) -> None:
        joined = " ".join(cmd)
        if self.dry_run:
            logger.info("Dry-run (%s): %s", purpose, joined)
            output_path.write_text(f"dry-run {purpose}\n")
            return

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            logger.debug("Running ffmpeg (%s attempt %d/%d): %s", purpose, attempt, self.max_retries, joined)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode == 0:
                return
            last_error = result.stderr.decode().strip()
            if attempt < self.max_retries:
                logger.warning(
                    "ffmpeg %s failed (attempt %d/%d). Retrying: %s",
                    purpose,
                    attempt,
                    self.max_retries,
                    last_error,
                )
                time.sleep(1)
            else:
                break

        raise FFmpegEditorError(
            f"ffmpeg {purpose} failed after {self.max_retries} attempts: {last_error}"
        )


class VideoAssembler:
    """Combine rendered segments into the final output."""

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        dry_run: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.dry_run = dry_run
        self.max_retries = max(1, max_retries)

    def assemble(self, segments: Iterable[Path], output_path: Path) -> None:
        segments = list(segments)
        if not segments:
            raise FFmpegEditorError("No segments supplied for final assembly.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            list_path = Path(tmp) / "concat.txt"
            list_path.write_text(
                "".join(f"file {shlex.quote(str(path.resolve()))}\n" for path in segments)
            )
            cmd = [
                self.ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ]
            self._run_ffmpeg(cmd, output_path, purpose="final assembly")

    def _run_ffmpeg(self, cmd: Sequence[str], output_path: Path, *, purpose: str) -> None:
        joined = " ".join(cmd)
        if self.dry_run:
            logger.info("Dry-run %s: %s", purpose, joined)
            output_path.write_text(f"dry-run {purpose}\n")
            return

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode == 0:
                return
            last_error = result.stderr.decode().strip()
            if attempt < self.max_retries:
                logger.warning(
                    "ffmpeg %s failed (attempt %d/%d). Retrying: %s",
                    purpose,
                    attempt,
                    self.max_retries,
                    last_error,
                )
                time.sleep(1)
            else:
                break

        raise FFmpegEditorError(
            f"ffmpeg {purpose} failed after {self.max_retries} attempts: {last_error}"
        )
