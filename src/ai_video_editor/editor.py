from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import tempfile
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
    ) -> None:
        self.ffmpeg = ffmpeg
        self.font_path = font_path
        self.music_dir = music_dir
        self.ffprobe = ffprobe
        self.dry_run = dry_run

    def render(
        self,
        plan: EditingPlan,
        chunk_lookup: dict[str, Path],
        output_dir: Path,
    ) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered_paths: List[Path] = []

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            for index, instruction in enumerate(plan.instructions, start=1):
                logger.info("Rendering instruction %d/%d", index, len(plan.instructions))
                output_path = output_dir / self._build_output_name(index, instruction)
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
                self._run_ffmpeg(cmd, intermediate_path)

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

        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
        ]
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

    def _run_ffmpeg(self, cmd: Sequence[str], output_path: Path) -> None:
        if self.dry_run:
            logger.info("Dry-run: %s", " ".join(cmd))
            output_path.write_text("dry-run placeholder\n")
            return

        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise FFmpegEditorError(
                f"ffmpeg failed ({result.returncode}): {result.stderr.decode().strip()}"
            )

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
            chain_parts = [
                f"[{idx}:a]",
                f"atrim=0:{duration:.3f}",
                "asetpts=N/SR/TB",
            ]
            if cue.volume and cue.volume != 1.0:
                chain_parts.append(f"volume={cue.volume:.3f}")
            delay_ms = max(0, int(round(start * 1000)))
            chain_parts.append(f"adelay={delay_ms}|{delay_ms}")
            filter_cmds.append(",".join(chain_parts) + label_out)
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

        logger.debug("Running music mix ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise FFmpegEditorError(
                f"ffmpeg music mix failed ({result.returncode}): {result.stderr.decode().strip()}"
            )

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


class VideoAssembler:
    """Combine rendered segments into the final output."""

    def __init__(self, ffmpeg: str = "ffmpeg", dry_run: bool = False) -> None:
        self.ffmpeg = ffmpeg
        self.dry_run = dry_run

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
            self._run_ffmpeg(cmd, output_path)

    def _run_ffmpeg(self, cmd: Sequence[str], output_path: Path) -> None:
        if self.dry_run:
            logger.info("Dry-run assembly: %s", " ".join(cmd))
            output_path.write_text("dry-run final placeholder\n")
            return

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise FFmpegEditorError(
                f"ffmpeg concat failed ({result.returncode}): {result.stderr.decode().strip()}"
            )
