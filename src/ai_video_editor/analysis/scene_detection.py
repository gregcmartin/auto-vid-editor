from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

SHOWINFO_PATTERN = re.compile(r"pts_time:(?P<time>[0-9]+\.?[0-9]*)")


def detect_scenes(video_path: Path, threshold: float = 0.4) -> List[Tuple[float, float]]:
    """Return a list of (start, end) timestamps for detected scenes."""
    duration = _probe_duration(video_path)
    if duration <= 0:
        return [(0.0, 0.0)]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        logger.warning("ffmpeg not found for scene detection: %s", exc)
        return [(0.0, duration)]

    scene_times: List[float] = [0.0]
    for line in result.stderr.splitlines():
        match = SHOWINFO_PATTERN.search(line)
        if match:
            try:
                time = float(match.group("time"))
                if 0.0 < time < duration:
                    scene_times.append(time)
            except ValueError:
                continue

    scene_times.append(duration)
    scene_times = sorted(set(scene_times))

    segments: List[Tuple[float, float]] = []
    for start, end in zip(scene_times, scene_times[1:]):
        if end - start <= 0.05:  # Ignore extremely short shots
            continue
        segments.append((start, end))

    if not segments:
        segments = [(0.0, duration)]

    return segments


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        logger.warning("Failed to probe duration for %s", video_path)
        return 0.0
