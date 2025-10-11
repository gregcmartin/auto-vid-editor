from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import AnalysisResult, PersonProfile, TimelineMoment, VideoAnalyzer

logger = logging.getLogger(__name__)

try:
    from dashscope import MultiModalConversation
except ImportError:  # pragma: no cover - optional dependency
    MultiModalConversation = None


class QwenVideoAnalyzer(VideoAnalyzer):
    """Wrapper around Qwen3-VL-30B-A3B-Instruct to describe video chunks."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
        self.model = model
        self.dry_run = dry_run
        if not dry_run and MultiModalConversation is None:
            raise ImportError(
                "dashscope is not installed. Install it to invoke Qwen models."
            )

    def analyze(self, video_path: Path) -> AnalysisResult:
        logger.info("Analyzing chunk with Qwen: %s", video_path)
        if self.dry_run:
            return self._mock_analysis(video_path)

        request = self._build_request_messages(video_path)
        logger.debug("Sending %s request to Qwen", self.model)
        response = MultiModalConversation.call(
            model=self.model,
            messages=request,
            result_format="json",
        )

        if response.status_code != 200:
            raise RuntimeError(f"Qwen request failed: {response.message}")

        payload = response.output.get("text") or response.output_text
        if not payload:
            raise RuntimeError("Qwen response did not include text output.")

        return self._parse_response(payload)

    def _build_request_messages(self, video_path: Path) -> List[Dict[str, Any]]:
        file_url = f"file://{video_path.resolve()}"
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
        return [
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"video": file_url},
                    {"text": user_prompt},
                ],
            },
        ]

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

    def _mock_analysis(self, video_path: Path) -> AnalysisResult:
        logger.debug("Using mock analysis for %s", video_path)
        placeholder = (
            f'{{"timeline":[{{"start":0,"end":30,"summary":"Overview of {video_path.name}","actions":["mock"]}}],'
            '"dull_sections":[],"people":[]}}'
        )
        return self._parse_response(placeholder)

