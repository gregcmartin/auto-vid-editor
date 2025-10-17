from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_name: str = "large-v2") -> None:
        try:
            from lightning_whisper_mlx import LightningWhisperMLX  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "lightning-whisper-mlx is not installed. Install with `pip install lightning-whisper-mlx`"
            ) from exc

        self._LightningWhisperMLX = LightningWhisperMLX
        self.model_name = model_name
        logger.info("Loading Lightning Whisper MLX model: %s", model_name)
        self.model = self._load_model(model_name)

    @classmethod
    def try_create(
        cls, model_name: str = "large-v2", **kwargs
    ) -> Optional["WhisperTranscriber"]:
        try:
            override = kwargs.get("whisper_model")
            return cls(model_name=override or model_name)
        except Exception as exc:
            logger.warning("Transcription disabled: %s", exc)
            return None

    def _load_model(self, model_name: str):
        """Handle constructor signature differences between library versions."""
        try:
            return self._LightningWhisperMLX(model_name)
        except TypeError:
            try:
                return self._LightningWhisperMLX(model=model_name)
            except TypeError:
                return self._LightningWhisperMLX(model_name)

    def transcribe_segment(
        self,
        video_path: Path,
        start: float,
        end: float,
        language: Optional[str] = None,
        audio_source: Optional[Path] = None,
    ) -> str:
        duration = max(0.0, end - start)
        if duration <= 0.1:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            source = audio_source or video_path
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source),
            ]
            if audio_source is None:
                cmd.append("-vn")
            cmd += [
                "-y",
                "-ac",
                "1",
                "-ar",
                "16000",
                temp_audio.name,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(
                    "Failed to extract audio for transcription (exit %s). Command: %s\n%s",
                    result.returncode,
                    " ".join(cmd),
                    result.stderr.decode().strip(),
                )
                return ""

            try:
                result = self.model.transcribe(
                    temp_audio.name,
                    language=language,
                )
                if isinstance(result, dict):
                    return str(result.get("text", "")).strip()
                return str(result).strip()
            except Exception as exc:
                logger.warning("Lightning Whisper MLX failed to transcribe segment: %s", exc)
                return ""
