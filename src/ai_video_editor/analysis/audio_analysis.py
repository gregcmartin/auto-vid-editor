from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .base import AudioSummary

logger = logging.getLogger(__name__)


VOL_REGEX = re.compile(r"(max_volume|mean_volume):\s*(-?[0-9]+\.?[0-9]*)\s*dB")


def analyze_audio(video_path: Path) -> AudioSummary:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.debug("ffmpeg not available for audio analysis: %s", exc)
        return AudioSummary()

    mean_volume: Optional[float] = None
    max_volume: Optional[float] = None
    for match in VOL_REGEX.finditer(result.stderr):
        value = float(match.group(2))
        if match.group(1) == "mean_volume":
            mean_volume = value
        elif match.group(1) == "max_volume":
            max_volume = value

    events = []
    if max_volume is not None and max_volume > -6.0:
        events.append("Significant audio peak detected")
    if mean_volume is not None and mean_volume < -45.0:
        events.append("Clip contains extended quiet sections")
    if mean_volume is not None and max_volume is not None:
        dynamic_range = max_volume - mean_volume
        if dynamic_range > 25.0:
            events.append("High dynamic range audio")

    return AudioSummary(mean_volume=mean_volume, max_volume=max_volume, events=events)
