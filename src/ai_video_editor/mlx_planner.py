from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from .editing_plan import EditingPlan

logger = logging.getLogger(__name__)

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    mlx_generate = None
    mlx_load = None
    make_sampler = None


class MLXDirectorPlanner:
    """Planner implementation backed by mlx-lm models."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        max_context_chars: int = 60000,
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
    ) -> None:
        if mlx_load is None or mlx_generate is None or make_sampler is None:
            raise ImportError(
                "mlx-lm is required for MLX planner inference. "
                "Install with: pip install mlx-lm"
            )

        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None

    def _load_model(self) -> None:
        if self.model is None:
            logger.info("Loading MLX planner model: %s", self.model_name)
            self.model, self.tokenizer = mlx_load(self.model_name)
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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logger.debug("Generating plan from MLX model")
        sampler = make_sampler(temp=self.temperature)
        output_text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
        )

        cleaned_text = self._clean_json_response(output_text)
        logger.debug("Planner response: %s", cleaned_text[:200])
        data = self._safe_load_json(cleaned_text)
        return EditingPlan.from_dict(data)

    def _collect_markdown(
        self,
        analysis_dir: Path,
        source_chunks: List[Path],
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
        self,
        chunks: Dict[str, Dict[str, str]],
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
