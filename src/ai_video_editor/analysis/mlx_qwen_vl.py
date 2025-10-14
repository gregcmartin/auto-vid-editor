from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base import AnalysisResult, PersonProfile, TimelineMoment, VideoAnalyzer

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

    def ensure_model_ready(self) -> None:
        """
        Preload the MLX model so that compatibility issues are raised early.
        """
        try:
            self._load_model()
        except Exception:
            # Reset partial state so a fallback path can retry cleanly.
            self.model = None
            self.processor = None
            self.config = None
            raise
        
    def _load_model(self):
        """Lazy load the MLX model on first use."""
        if self.model is None:
            logger.info("Loading MLX model: %s", self.model_name)
            self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
            logger.info("MLX model loaded successfully")

    def analyze(self, video_path: Path) -> AnalysisResult:
        logger.info("Analyzing chunk with MLX: %s", video_path)
        self._load_model()
        
        system_prompt = (
            "You are a meticulous assistant helping a video editor. "
            "Analyse the provided video clip and respond strictly in JSON. "
            "Identify meaningful narrative beats with estimated timestamps, "
            "flag sections with minimal action suitable for trimming, "
            "and profile each unique person in frame."
        )
        
        user_prompt = (
            "Return a JSON object with the following keys:\n"
            "timeline: list of objects with keys start, end, summary, notes (optional), actions (string list).\n"
            "dull_sections: list of objects with keys start, end, summary, notes(optional), actions(optional).\n"
            "people: list of objects with keys identifier, appearance, first_seen, last_seen, inferred_name(optional), supporting_evidence(optional).\n"
            "Timestamps should be seconds relative to the start of this clip.\n"
            "If unsure about a value, make your best estimate but mark the uncertainty in notes.\n"
            "Ensure the JSON is valid and parsable with double quotes."
        )
        
        # Prepare the prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = apply_chat_template(
            self.processor,
            self.config,
            messages,
            add_generation_prompt=True
        )
        
        # Run inference with video
        logger.debug("Running MLX inference on video")
        video_source = str(video_path.resolve())
        try:
            generation = generate(
                self.model,
                self.processor,
                prompt,
                video=[video_source],
                max_tokens=2048,
                temperature=0.7,
                verbose=False,
            )
            output_text = generation.text if hasattr(generation, "text") else str(generation)
            logger.debug("Model response: %s", output_text[:200])
            return self._parse_response(output_text)
            
        except Exception as e:
            logger.error("MLX inference failed: %s", e)
            # Return empty result on error
            return AnalysisResult(
                timeline=[],
                dull_sections=[],
                people=[],
                raw_response=f"Error: {str(e)}",
            )

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

        return AnalysisResult(
            timeline=timeline,
            dull_sections=dull_sections,
            people=people,
            raw_response=response_text,
        )

    @staticmethod
    def _clean_json_response(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            newline = stripped.find("\n")
            if newline != -1:
                stripped = stripped[newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()
