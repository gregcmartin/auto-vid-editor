from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

from .base import (
    AnalysisResult,
    AudioEvent,
    AudioSummary,
    AnalysisParseError,
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
        model_name: str = "mlx-community/Qwen3-VL-8B-Thinking-8bit",
        max_tokens: int = 768,
        temperature: float = 0.0,
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
        self._retry_hint: Optional[str] = None
        self._last_raw_response: Optional[str] = None

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
            "You are a meticulous assistant helping a video editor.\n"
            "You must respond with exactly one valid JSON object adhering to the requested schema.\n"
            "Do not include commentary, markdown, or code fences—return JSON only.\n"
            "If information is unavailable, use empty lists [] or null."
        )

        example_json = json.dumps(
            {
                "timeline": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "summary": "Example summary",
                        "notes": None,
                        "actions": [],
                        "confidence": 0.9,
                    }
                ],
                "dull_sections": [],
                "people": [],
                "shot_notes": [],
                "audio_events": [],
                "overall_summary": "Example overview",
            },
            indent=2,
        )

        user_prompt = (
            "Return a JSON object with the following keys:\n"
            "timeline: list of objects with keys start, end, summary, notes(optional), actions(list[str]), confidence(optional 0-1).\n"
            "dull_sections: list of objects with keys start, end, summary, notes(optional), actions(optional), confidence(optional 0-1).\n"
            "people: list of objects with keys identifier, appearance, first_seen, last_seen, inferred_name(optional), supporting_evidence(optional).\n"
            "shot_notes: list of objects with keys start, end, transcript(optional), summary, notable_objects(list[str]), emotions(optional), confidence(optional 0-1).\n"
            "audio_events: list of objects with keys time, description, confidence(optional 0-1).\n"
            "overall_summary: string describing the clip.\n"
            "Keep all timestamps within the clip bounds and aligned to provided shots.\n"
            "Do NOT invent placeholder people or ungrounded events.\n"
            "If the clip feels uneventful, still include one timeline entry covering the full duration explaining what is visible.\n"
            "The timeline array must contain at least one item and overall_summary must be non-empty.\n"
            "Respond with JSON ONLY, beginning with '{' and ending with '}'.\n"
            "Example format:\n"
            f"{example_json}"
        )
        if self._retry_hint:
            user_prompt += f"\nAdditional guidance: {self._retry_hint.strip()}"

        shot_context = self._build_shot_context(shots, audio_summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": shot_context + "\n\n" + user_prompt},
        ]

        num_images = len(frame_paths)
        template_kwargs = {}
        if num_images > 0:
            template_kwargs["num_images"] = num_images
        else:
            template_kwargs["video"] = str(video_path.resolve())

        prompt = apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True,
            **template_kwargs,
        )
        logger.debug(
            "MLX prompt prepared (type=%s, len=%s, images=%d)",
            type(prompt).__name__,
            len(prompt) if isinstance(prompt, str) else "n/a",
            num_images,
        )

        logger.debug(
            "Running MLX inference with %d frames. Prompt preview: %s",
            len(frame_paths),
            prompt[:400] if isinstance(prompt, str) else str(prompt)[:400],
        )
        try:
            media_kwargs = {}
            if frame_paths:
                media_kwargs["images"] = frame_paths
            else:
                media_kwargs["video"] = [str(video_path.resolve())]
            generation_kwargs = {
                "max_tokens": self.max_tokens,
                "verbose": False,
                **media_kwargs,
            }
            if self.temperature and self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            generation = generate(
                self.model,
                self.processor,
                prompt,
                **generation_kwargs,
            )
            output_text = generation.text if hasattr(generation, "text") else str(generation)
            logger.debug("Model response: %s", output_text[:200])

            if not output_text or output_text.strip() == "":
                raise RuntimeError("MLX analyser returned empty response")

            self._last_raw_response = output_text
            try:
                result = self._parse_response(output_text)
            except AnalysisParseError as parse_exc:
                if not parse_exc.raw_response:
                    parse_exc.raw_response = output_text
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise AnalysisParseError("Failed to parse model response", raw_response=output_text) from exc
            return result

        except AnalysisParseError:
            raise
        except Exception as exc:
            logger.error("MLX inference failed: %s", exc)
            return AnalysisResult(
                timeline=[],
                dull_sections=[],
                people=[],
                raw_response=f"Error: {str(exc)}",
            )

    # Retry hint plumbing -------------------------------------------------

    def set_retry_hint(self, hint: Optional[str]) -> None:
        self._retry_hint = hint.strip() if hint else None

    def clear_retry_hint(self) -> None:
        self._retry_hint = None

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
        if cleaned is None:
            raise AnalysisParseError("Model response did not contain JSON object.", raw_response=response_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse JSON response: %s. Raw snippet: %s",
                exc,
                response_text[:500],
            )
            raise AnalysisParseError(str(exc), raw_response=response_text)

        timeline = [
            TimelineMoment(
                start=_coerce_required_float(entry.get("start"), "timeline.start"),
                end=_coerce_required_float(entry.get("end"), "timeline.end"),
                summary=entry["summary"],
                notes=entry.get("notes"),
                actions=list(entry.get("actions", [])),
                confidence=_safe_float(entry.get("confidence")),
            )
            for entry in data.get("timeline", [])
        ]
        dull_sections = [
            TimelineMoment(
                start=_coerce_required_float(entry.get("start"), "dull_sections.start"),
                end=_coerce_required_float(entry.get("end"), "dull_sections.end"),
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
                first_seen=_coerce_required_float(entry.get("first_seen"), "people.first_seen"),
                last_seen=_coerce_required_float(entry.get("last_seen"), "people.last_seen"),
                inferred_name=entry.get("inferred_name"),
                supporting_evidence=entry.get("supporting_evidence"),
            )
            for entry in data.get("people", [])
        ]

        shot_notes = [
            ShotNote(
                start=_coerce_required_float(entry.get("start"), "shot_notes.start"),
                end=_coerce_required_float(entry.get("end"), "shot_notes.end"),
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
                time=_coerce_required_float(entry.get("time"), "audio_events.time"),
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
    def _clean_json_response(text: str) -> Optional[str]:
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
        return _extract_first_json_object(stripped)


def _safe_float(value: Optional[Union[str, float, int]]) -> Optional[float]:
    parsed = _parse_float(value)
    return parsed


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_float(value: Optional[Union[str, float, int]]) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER_PATTERN.search(value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _coerce_required_float(value: Optional[Union[str, float, int]], field: str) -> float:
    parsed = _parse_float(value)
    if parsed is None:
        logger.debug("Field %s missing or invalid (%r); defaulting to 0.0", field, value)
        return 0.0
    return parsed


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None
