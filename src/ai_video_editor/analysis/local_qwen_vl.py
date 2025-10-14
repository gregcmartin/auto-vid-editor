from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from .base import AnalysisResult, PersonProfile, TimelineMoment, VideoAnalyzer

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

    def analyze(self, video_path: Path) -> AnalysisResult:
        logger.info("Analyzing chunk with local Qwen: %s", video_path)
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
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path.resolve())},
                    {"type": "text", "text": user_prompt},
                ],
            }
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
                max_new_tokens=2048,
                temperature=0.7,
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

    def _move_to_device(self, data: Union[torch.Tensor, dict, list, tuple], device: torch.device):
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
            # Remove optional language hint like ```json
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()
