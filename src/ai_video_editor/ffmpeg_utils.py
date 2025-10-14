from __future__ import annotations

import logging
import platform
import subprocess
from typing import Optional, Set


def probe_ffmpeg_encoders(ffmpeg_bin: str) -> Set[str]:
    """Return the set of video encoder names advertised by ffmpeg."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        logging.debug("Failed to probe ffmpeg encoders via %s: %s", ffmpeg_bin, exc)
        return set()

    if result.returncode != 0:
        logging.debug(
            "ffmpeg encoder probe returned %s: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return set()

    encoders: Set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("-----") or line.startswith("Encoders:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def select_video_encoder(requested: str, encoders: Optional[Set[str]] = None) -> str:
    """Return the encoder to use based on the user's request and availability."""
    if requested != "auto":
        return requested

    detected = encoders or set()
    if platform.system() == "Darwin" and "h264_videotoolbox" in detected:
        logging.info("Auto-selecting h264_videotoolbox encoder (VideoToolbox detected).")
        return "h264_videotoolbox"

    return "libx264"


def ensure_encoder_available(encoder: str, encoders: Set[str]) -> str:
    """Confirm the selected encoder is available; otherwise fall back to libx264."""
    if encoder == "libx264":
        return encoder
    if encoder in encoders:
        return encoder
    if "h264_videotoolbox" in encoders and encoder == "hevc_videotoolbox":
        logging.warning(
            "Requested encoder '%s' unavailable. Falling back to h264_videotoolbox.",
            encoder,
        )
        return "h264_videotoolbox"
    logging.warning(
        "Requested encoder '%s' not found in ffmpeg build. Falling back to libx264.",
        encoder,
    )
    return "libx264"
