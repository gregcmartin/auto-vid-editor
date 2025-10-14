from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from .base import (
    AnalysisResult,
    AudioEvent,
    AudioSummary,
    PersonProfile,
    ShotNote,
    ShotSegment,
    TimelineMoment,
    VideoAnalyzer,
)

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError:
    mx = None
    load = None
    generate = None
    apply_chat_template = None
    load_config = None


class MLXQwenVideoAnalyzer(VideoAnalyzer):
    """MLX-based implementation for Qwen3-VL on Apple Silicon."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit",
        max_tokens: int = 1536,
        temperature: float = 0.1,
        **kwargs
    ) -> None:
        if mx is None:
            raise ImportError(
                "mlx-vlm is required for MLX inference. "
                "Install with: pip install mlx-vlm"
            )

        self.model_name = model_name
        self.model = None
        self.processor = None
        self.config = None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_frames_per_shot = 4
        self.max_total_frames = 48

    def ensure_model_ready(self) -> None:
        """Preload the MLX model so that compatibility issues are raised early."""
        try:
            self._load_model()
        except Exception:
            self.model = None
            self.processor = None
            self.config = None
            raise

    def _load_model(self) -> None:
        if self.model is None:
            logger.info("Loading MLX model: %s", self.model_name)
            self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
            logger.info("MLX model loaded successfully")

    def analyze(
        self,
        video_path: Path,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary] = None,
    ) -> AnalysisResult:
        logger.info("Analyzing chunk with MLX: %s", video_path)
        self._load_model()

        frame_paths = self._frame_inputs(shots)
        system_prompt = (
            "You are a meticulous assistant helping a video editor. "
            "Analyse the provided video clip and respond strictly in JSON. "
            "Use the provided shot breakdown, transcripts, and audio cues to drive your analysis. "
            "Identify meaningful narrative beats with estimated timestamps, "
            "flag sections with minimal action suitable for trimming, "
            "profile each unique person in frame, and capture notable audio events. "
            "Only describe people, objects, or events you clearly observe in the provided frames."
        )

        user_prompt = (
            "Return a JSON object with the following keys:\n"
            "timeline: list of objects with keys start, end, summary, notes (optional), actions (string list), confidence (0-1).\n"
            "dull_sections: list of objects with keys start, end, summary, notes(optional), actions(optional), confidence (0-1).\n"
            "people: list of objects with keys identifier, appearance, first_seen, last_seen, inferred_name(optional), supporting_evidence(optional).\n"
            "shot_notes: list of objects with keys start, end, transcript, summary, notable_objects (string list), emotions(optional), confidence (0-1).\n"
            "audio_events: list of objects with keys time, description, confidence (0-1).\n"
            "overall_summary: string providing a concise narrative of the clip.\n"
            "Timestamps must remain within the clip bounds (0 through clip duration) and align with the shot timings.\n"
            "If you are uncertain about any key, return an empty list or null rather than guessing.\n"
            "Do NOT invent placeholder people or events.\n"
            "Ensure the JSON is valid and parsable with double quotes."
        )

        shot_context = self._build_shot_context(shots, audio_summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": shot_context + "\n\n" + user_prompt},
        ]

        prompt = apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
        )

        logger.debug("Running MLX inference with %d attached frames", len(frame_paths))
        try:
            media_kwargs = {}
            if frame_paths:
                media_kwargs["image"] = frame_paths
            else:
                media_kwargs["video"] = [str(video_path.resolve())]
            generation = generate(
                self.model,
                self.processor,
                prompt,
                sampler=make_sampler(temp=self.temperature, top_p=1.0),
                max_tokens=self.max_tokens,
                **media_kwargs,
                verbose=False,
            )
            output_text = generation.text if hasattr(generation, "text") else str(generation)
            logger.debug("Model response: %s", output_text[:200])

            if not output_text or output_text.strip() == "":
                raise RuntimeError("MLX analyser returned empty response")

            return self._parse_response(output_text)

        except Exception as exc:
            logger.error("MLX inference failed: %s", exc)
            return AnalysisResult(
                timeline=[],
                dull_sections=[],
                people=[],
                raw_response=f"Error: {str(exc)}",
            )

    def _build_shot_context(
        self,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary],
    ) -> str:
        lines: List[str] = ["Shot breakdown:"]
        if not shots:
            lines.append("No shot segmentation available for this clip.")
        else:
            for idx, shot in enumerate(shots, start=1):
                lines.append(f"Shot {idx}: {shot.start:.2f}s → {shot.end:.2f}s")
                if shot.transcript:
                    lines.append(f"  Transcript: {shot.transcript.strip()}")
                if shot.audio_highlights:
                    lines.append("  Audio cues: " + ", ".join(shot.audio_highlights))
        if audio_summary:
            lines.append("\nChunk-level audio summary:")
            if audio_summary.mean_volume is not None:
                lines.append(f"  Mean volume: {audio_summary.mean_volume:.2f} dBFS")
            if audio_summary.max_volume is not None:
                lines.append(f"  Peak volume: {audio_summary.max_volume:.2f} dBFS")
            if audio_summary.events:
                lines.append("  Events: " + "; ".join(audio_summary.events))
        for idx, shot in enumerate(shots, start=1):
            if not shot.frames:
                continue
            lines.append(f"\nFrames for shot {idx}: {len(shot.frames)} captured")
        return "\n".join(lines)

    def _frame_inputs(self, shots: List[ShotSegment]) -> List[str]:
        paths: List[str] = []
        for shot in shots:
            if not shot.frames:
                continue
            for frame_path in shot.frames[: self.max_frames_per_shot]:
                paths.append(str(frame_path.resolve()))
                if len(paths) >= self.max_total_frames:
                    return paths
        return paths

    def _parse_response(self, response_text: str) -> AnalysisResult:
        cleaned = self._clean_json_response(response_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON response: %s", exc)
            return AnalysisResult(
                timeline=[],
                dull_sections=[],
                people=[],
                raw_response=response_text,
            )

        timeline = [
            TimelineMoment(
                start=float(entry["start"]),
                end=float(entry["end"]),
                summary=entry["summary"],
                notes=entry.get("notes"),
                actions=list(entry.get("actions", [])),
                confidence=_safe_float(entry.get("confidence")),
            )
            for entry in data.get("timeline", [])
        ]
        dull_sections = [
            TimelineMoment(
                start=float(entry["start"]),
                end=float(entry["end"]),
                summary=entry.get("summary", ""),
                notes=entry.get("notes"),
                actions=list(entry.get("actions", [])),
                confidence=_safe_float(entry.get("confidence")),
            )
            for entry in data.get("dull_sections", [])
        ]
        people = [
            PersonProfile(
                identifier=entry.get("identifier", "unknown"),
                appearance=entry.get("appearance", ""),
                first_seen=float(entry.get("first_seen", 0.0)),
                last_seen=float(entry.get("last_seen", 0.0)),
                inferred_name=entry.get("inferred_name"),
                supporting_evidence=entry.get("supporting_evidence"),
            )
            for entry in data.get("people", [])
        ]

        shot_notes = [
            ShotNote(
                start=float(entry.get("start", 0.0)),
                end=float(entry.get("end", 0.0)),
                transcript=entry.get("transcript"),
                summary=entry.get("summary", ""),
                notable_objects=list(entry.get("notable_objects", [])),
                emotions=entry.get("emotions"),
                confidence=_safe_float(entry.get("confidence")),
            )
            for entry in data.get("shot_notes", [])
        ]

        audio_events = [
            AudioEvent(
                time=float(entry.get("time", 0.0)),
                description=entry.get("description", ""),
                confidence=_safe_float(entry.get("confidence")),
            )
            for entry in data.get("audio_events", [])
        ]

        overall_summary = data.get("overall_summary")

        return AnalysisResult(
            timeline=timeline,
            dull_sections=dull_sections,
            people=people,
            shot_notes=shot_notes,
            audio_events=audio_events,
            overall_summary=overall_summary,
            raw_response=response_text,
        )

    @staticmethod
    def _clean_json_response(text: str) -> str:
        stripped = text.strip()
        if "</think>" in stripped:
            stripped = stripped.split("</think>", 1)[1]
        elif stripped.startswith("<think>"):
            stripped = stripped[len("<think>") :]
        stripped = stripped.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            newline = stripped.find("\n")
            if newline != -1:
                stripped = stripped[newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        stripped = stripped.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and start < end:
            return stripped[start : end + 1].strip()
        return stripped


def _safe_float(value: Optional[Union[str, float, int]]) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
