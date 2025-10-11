from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .base import AnalysisResult, PersonProfile, TimelineMoment, VideoAnalyzer

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from qwen_vl_utils import process_vision_info
except ImportError:
    torch = None
    AutoConfig = None
    AutoProcessor = None
    AutoModelForVision2Seq = None
    init_empty_weights = None
    load_checkpoint_and_dispatch = None
    process_vision_info = None


class LocalQwenVideoAnalyzer(VideoAnalyzer):
    """Local implementation using HuggingFace transformers for Qwen3-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        if AutoModelForVision2Seq is None or init_empty_weights is None:
            raise ImportError(
                "transformers, accelerate, and qwen-vl-utils are required for local inference. "
                "Install with: pip install transformers>=4.43.0 accelerate>=0.33.0 qwen-vl-utils torch"
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
            
            # Set MPS high watermark for better memory management on Apple Silicon
            import os
            from pathlib import Path
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            
            # Setup offload directory for weight sharding
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            offload_dir = hf_home / "ai_video_editor_offload"
            offload_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine torch dtype
            if self.torch_dtype == "auto":
                # Use float16 for MPS (Apple Silicon), bfloat16 for CUDA, float32 for CPU
                if torch.backends.mps.is_available():
                    dtype = torch.float16
                elif torch.cuda.is_available():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # Cap memory to force sharding across MPS+CPU instead of single huge buffer
            max_memory = {
                "mps": "46GiB",  # Leave headroom for video processing + Metal
                "cpu": "48GiB",  # Rest goes to CPU RAM
            }
            
            logger.info("Using dtype: %s, device: %s, offload_dir: %s", dtype, self.device, offload_dir)
            
            # Step 1: Load config only (no weights)
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Prefer SDPA attention (MPS-compatible) - set before model creation
            try:
                config.attn_implementation = "sdpa"
            except Exception:
                pass
            
            # Step 2: Build param-less module (no real tensors yet - avoids huge buffer allocation)
            logger.info("Initializing model with empty weights...")
            with init_empty_weights():
                self.model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
            
            # Step 3: Param-by-param dispatch from shards (streams weights directly to devices)
            # This prevents any single 40-50 GiB contiguous buffer allocation
            logger.info("Loading and dispatching model weights from checkpoint shards...")
            self.model = load_checkpoint_and_dispatch(
                self.model,
                self.model_name,  # HF repo ID or path
                device_map=self.device,  # Let Accelerate place MPS/CPU shards
                dtype=dtype,
                offload_folder=str(offload_dir),  # Push least-hot weights to SSD
                max_memory=max_memory,  # Cap memory per device
                no_split_module_classes=[  # Prevent bad mid-layer splits
                    "QwenDecoderLayer", "QwenMLP", "QwenAttention",
                    "VisionTransformerLayer", "QwenVisionBlock"
                ],
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
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
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        inputs = inputs.to(self.model.device)
        
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
