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
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
except ImportError:
    torch = None
    AutoProcessor = None
    AutoModelForImageTextToText = None
    process_vision_info = None


class LocalQwenVideoAnalyzer(VideoAnalyzer):
    """Local implementation using HuggingFace transformers for Qwen3-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1536,
        temperature: float = 0.1,
    ) -> None:
        if AutoModelForImageTextToText is None or torch is None:
            raise ImportError(
                "transformers, torch, and qwen-vl-utils are required for local inference. "
                "Install with: pip install transformers>=4.43.0 qwen-vl-utils torch"
            )

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_frames_per_shot = 4
        self.max_total_frames = 48
        
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            logger.info("Loading local model: %s", self.model_name)
            
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "auto":
                dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
            else:
                dtype = torch.float32

            if self.device == "auto":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            else:
                device = torch.device(self.device)

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map={"": device.type},
                trust_remote_code=True,
            )

            if device.type == "cpu":
                self.model = self.model.to(device)

            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            logger.info("Model loaded successfully")

    def analyze(
        self,
        video_path: Path,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary] = None,
    ) -> AnalysisResult:
        logger.info("Analyzing chunk with local Qwen: %s", video_path)
        self._load_model()
        shot_context = self._build_shot_context(shots, audio_summary)
        frame_inputs = self._frame_inputs(shots)

        system_prompt = (
            "You are a meticulous assistant helping a video editor. "
            "Analyse the provided video clip and respond strictly in JSON. "
            "Use the provided shot breakdown, transcripts, and audio cues to drive your analysis. "
            "Identify meaningful narrative beats with estimated timestamps, "
            "flag sections with minimal action suitable for trimming, "
            "profile each unique person in frame, and capture notable audio events. "
            "Only describe people, objects, or events that are clearly visible in the provided frames."
        )

        user_prompt = (
            "Return a JSON object with the following keys:\n"
            "timeline: list of objects with keys start, end, summary, notes (optional), actions (string list), confidence (0-1).\n"
            "dull_sections: list of objects with keys start, end, summary, notes(optional), actions(optional), confidence (0-1).\n"
            "people: list of objects with keys identifier, appearance, first_seen, last_seen, inferred_name(optional), supporting_evidence(optional).\n"
            "shot_notes: list of objects with keys start, end, transcript, summary, notable_objects (string list), emotions(optional), confidence (0-1).\n"
            "audio_events: list of objects with keys time, description, confidence (0-1).\n"
            "overall_summary: string providing a concise narrative of the clip.\n"
            "Timestamps must fall within the clip bounds (0 through clip duration) and align with the provided shot timings.\n"
            "If you are uncertain about any field, return an empty list or null instead of inventing content.\n"
            "Do NOT create placeholder individuals (e.g. \"Person A\") or events that are not grounded in the supplied frames and transcripts.\n"
            "All strings must be plain text without markdown. Ensure the JSON is valid and parsable with double quotes."
        )

        user_content = [{"type": "text", "text": shot_context}]
        user_content.extend(frame_inputs)
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        
        # Process the messages
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        
        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        inputs = self.processor(**processor_kwargs)

        target_device = self.model.get_input_embeddings().weight.device
        inputs = self._move_to_device(inputs, target_device)
        
        # Generate response
        logger.debug("Generating response from local model")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                top_p=1.0,
            )
        
        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.debug("Model response: %s", output_text[:200])
        return self._parse_response(output_text)

    def _build_shot_context(
        self,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary],
    ) -> str:
        lines: List[str] = ["Shot breakdown:"]
        if not shots:
            lines.append("No shot segmentation available for this chunk.")
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

    def _frame_inputs(self, shots: List[ShotSegment]) -> List[dict]:
        payload: List[dict] = []
        total_frames = 0
        for shot in shots:
            if not shot.frames:
                continue
            frames_for_shot = shot.frames[: self.max_frames_per_shot]
            for frame_path in frames_for_shot:
                payload.append({"type": "image", "image": str(frame_path.resolve())})
                total_frames += 1
                if total_frames >= self.max_total_frames:
                    return payload
        return payload

    def _move_to_device(
        self,
        data: Union[torch.Tensor, dict, list, tuple],
        device: torch.device,
    ):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v, device) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_to_device(item, device) for item in data]
        if isinstance(data, tuple):
            return tuple(self._move_to_device(item, device) for item in data)
        return data

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
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
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
