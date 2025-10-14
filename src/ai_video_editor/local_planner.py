from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .editing_plan import EditingPlan

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LocalDirectorPlanner:
    """Local implementation using HuggingFace transformers for Qwen text models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B-MLX-4bit",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_context_chars: int = 12000,
    ) -> None:
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers and torch are required for local inference. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_context_chars = max_context_chars
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            logger.info("Loading local planner model: %s", self.model_name)
            
            if self.device == "auto":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            else:
                device = torch.device(self.device)

            if self.torch_dtype == "auto":
                if device.type in ("mps", "cuda"):
                    dtype = torch.float16
                else:
                    dtype = torch.float32
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            logger.info("Using dtype: %s, device: %s", dtype, device)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                trust_remote_code=True,
            ).to(device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info("Planner model loaded successfully")

    def plan(
        self,
        analysis_dir: Path,
        source_chunks: List[Path],
        music_library: List[str],
    ) -> EditingPlan:
        if not analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
        if not source_chunks:
            raise ValueError("source_chunks must contain at least one chunk Path.")

        self._load_model()
        
        context = self._collect_markdown(analysis_dir, source_chunks)
        truncated = self._truncate_context(context)
        
        messages = self._build_messages(truncated, music_library)
        
        # Format messages for the model
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        logger.debug("Generating plan from local model")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
            )
        
        # Decode only the generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        cleaned_text = self._clean_json_response(output_text)

        logger.debug("Planner response: %s", cleaned_text[:200])
        data = self._safe_load_json(cleaned_text)
        return EditingPlan.from_dict(data)

    def _collect_markdown(
        self, analysis_dir: Path, source_chunks: List[Path]
    ) -> Dict[str, Dict[str, str]]:
        files = sorted(analysis_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(
                f"No markdown files found in analysis directory: {analysis_dir}"
            )
        chunk_lookup = {chunk.stem: chunk.name for chunk in source_chunks}
        context: Dict[str, Dict[str, str]] = {}
        for file in files:
            stem = file.stem
            chunk_name = chunk_lookup.get(stem, stem)
            context[file.name] = {
                "chunk": chunk_name,
                "content": file.read_text(),
            }
        return context

    def _truncate_context(
        self, chunks: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        total_length = sum(len(entry["content"]) for entry in chunks.values())
        if total_length <= self.max_context_chars:
            return chunks

        logger.warning(
            "Context exceeds max length (%d > %d). Truncating content.",
            total_length,
            self.max_context_chars,
        )
        truncated: Dict[str, Dict[str, str]] = {}
        quota = self.max_context_chars // max(len(chunks), 1)
        for name, info in chunks.items():
            truncated[name] = {
                "chunk": info["chunk"],
                "content": info["content"][:quota],
            }
        return truncated

    def _build_messages(
        self,
        context: Dict[str, Dict[str, str]],
        music_library: List[str],
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a seasoned video director creating an edit plan from chunk analyses. "
            "Respond strictly in JSON following the provided schema."
        )
        user_prompt = (
            "Schema:\n"
            "{\n"
            '  "title": string,\n'
            '  "summary": string,\n'
            '  "outro_text": string (optional),\n'
            '  "voiceover_notes": string (optional),\n'
            '  "warnings": [string],\n'
            '  "instructions": [\n'
            "    {\n"
            '      "chunk": string (filename),\n'
            '      "in_start": number,\n'
            '      "in_end": number,\n'
            '      "label": string (optional scene name),\n'
            '      "speed": number (1.0 normal),\n'
            '      "keep_audio": boolean,\n'
            '      "crop": {"width": int, "height": int, "x": int, "y": int} (optional),\n'
            '      "text_overlays": [{"text": string, "start": number, "end": number, "position": string, "style": string (optional)}],\n'
            '      "subtitles": [{"text": string, "start": number, "end": number}],\n'
            '      "music_cues": [{"track": string, "start": number, "end": number, "volume": number}],\n'
            '      "notes": string (optional)\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Use the clip summaries to design an engaging narrative. "
            "Combine or shorten dull sections, recommend speed-ups when useful, "
            "and align overlays/subtitles with people descriptions. Keep timestamps within the chunk bounds.\n\n"
            "Music library available for cues:\n"
        )
        if music_library:
            user_prompt += "\n".join(f"- {track}" for track in music_library)
        else:
            user_prompt += "- (No background music available)"
        
        context_blob = "\n\n".join(
            f"### {info['chunk']} (source: {name})\n{info['content']}"
            for name, info in context.items()
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\nAnalyses:\n" + context_blob},
        ]

    @staticmethod
    def _safe_load_json(payload: str) -> Dict[str, object]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.error("Planner returned invalid JSON: %s", exc)
            raise

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
