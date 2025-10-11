from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class VideoSplitterError(RuntimeError):
    """Raised when ffmpeg/ffprobe reports an error."""


class VideoSplitter:
    """Split a source video into evenly sized temporal chunks."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def split(
        self,
        video_path: Path,
        output_dir: Path,
        parts: int = 4,
        overwrite: bool = True,
    ) -> List[Path]:
        if parts <= 0:
            raise ValueError("parts must be > 0")
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        duration = self._probe_duration(video_path)
        logger.info("Video duration: %.2fs", duration)
        segment_length = duration / parts

        chunk_paths: List[Path] = []
        for index in range(parts):
            start_time = segment_length * index
            start_ts = self._format_timestamp(start_time)
            chunk_path = output_dir / f"chunk_{index+1:02d}{video_path.suffix}"
            chunk_paths.append(chunk_path)

            ffmpeg_cmd = [
                self.ffmpeg,
                "-y" if overwrite else "-n",
                "-ss",
                start_ts,
                "-i",
                str(video_path),
            ]
            if index < parts - 1:
                ffmpeg_cmd += ["-t", self._format_timestamp(segment_length)]
            ffmpeg_cmd += ["-c", "copy", str(chunk_path)]

            logger.debug("Running ffmpeg: %s", " ".join(ffmpeg_cmd))
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise VideoSplitterError(
                    f"ffmpeg failed for chunk {index+1}: {result.stderr.decode().strip()}"
                )

        return chunk_paths

    def _probe_duration(self, video_path: Path) -> float:
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        logger.debug("Running ffprobe: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise VideoSplitterError(
                f"ffprobe failed: {result.stderr.decode().strip()}"
            )
        try:
            return float(result.stdout.decode().strip())
        except ValueError as exc:
            raise VideoSplitterError(
                f"Unable to parse ffprobe duration: {result.stdout!r}"
            ) from exc

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds - (hours * 3600 + minutes * 60)
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
