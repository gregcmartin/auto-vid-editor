from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base import AnalysisResult, PersonProfile, TimelineMoment, VideoAnalyzer

logger = logging.getLogger(__name__)

try:
    from nexa.gguf import NexaVLMInference
except ImportError:
    NexaVLMInference = None


class NexaQwenVideoAnalyzer(VideoAnalyzer):
    """MLX-based implementation using NexaSDK for Qwen3-VL on Apple Silicon."""

    def __init__(
        self,
        model_name: str = "NexaAI/qwen3vl-30B-A3B-mlx",
        **kwargs
    ) -> None:
        if NexaVLMInference is None:
            raise ImportError(
                "nexaai is required for MLX inference. "
                "Install with: pip install nexaai"
            )
        
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load the MLX model on first use."""
        if self.model is None:
            logger.info("Loading MLX model via NexaSDK: %s", self.model_name)
            self.model = NexaVLMInference(
                model_path=self.model_name,
                local_path=None,
                stop_words=[],
                temperature=0.7,
                max_new_tokens=2048,
                top_k=50,
                top_p=1.0,
            )
            logger.info("MLX model loaded successfully")

    def analyze(self, video_path: Path) -> AnalysisResult:
        logger.info("Analyzing chunk with Nexa MLX: %s", video_path)
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
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Run inference with video
        logger.debug("Running MLX inference on video")
        try:
            # NexaSDK expects video path as string
            response = self.model.inference(
                prompt=full_prompt,
                image_path=str(video_path.resolve())
            )
            
            # Extract the generated text
            if isinstance(response, dict) and "output" in response:
                output_text = response["output"]
            elif isinstance(response, str):
                output_text = response
            else:
                output_text = str(response)
            
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
        try:
            data = json.loads(response_text)
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
