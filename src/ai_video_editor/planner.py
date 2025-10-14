from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .editing_plan import EditingPlan

logger = logging.getLogger(__name__)

# DashScope integration removed


class DirectorPlanner:
    """Invokes a language model to turn analysis markdown into an editing plan."""

    def __init__(
        self,
        model: str = "Qwen3-30B-A3B",
        api_key: Optional[str] = None,
        dry_run: bool = False,
        max_context_chars: int = 12000,
    ) -> None:
        self.model = model
        self.dry_run = dry_run
        self.max_context_chars = max_context_chars

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

        context = self._collect_markdown(analysis_dir, source_chunks)
        truncated = self._truncate_context(context)

        if self.dry_run:
            logger.debug("Planner running in dry-run mode.")
            return self._mock_plan(truncated, music_library)

        messages = self._build_messages(truncated, music_library)
        logger.debug("Requesting plan from model %s", self.model)
        # DashScope integration removed - using local models only
        raise RuntimeError("DashScope integration removed. Use local models instead.")



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
    ) -> List[Dict[str, object]]:
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
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"text": user_prompt},
                    {"text": "Analyses:\n" + context_blob},
                ],
            },
        ]

    @staticmethod
    def _safe_load_json(payload: str) -> Dict[str, object]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.error("Planner returned invalid JSON: %s", exc)
            raise

    def _mock_plan(
        self,
        context: Dict[str, Dict[str, str]],
        music_library: List[str],
    ) -> EditingPlan:
        first_chunk = next(iter(context.values()))
        track = music_library[0] if music_library else None
        sample = {
            "title": "Sample Dry-Run Edit",
            "summary": "Mock summary created for testing purposes.",
            "warnings": [],
            "instructions": [
                {
                    "chunk": first_chunk["chunk"],
                    "in_start": 0,
                    "in_end": 30,
                    "label": "Hook",
                    "speed": 1.0,
                    "keep_audio": True,
                    "text_overlays": [
                        {
                            "text": "Dry-run overlay",
                            "start": 2,
                            "end": 6,
                            "position": "bottom_left",
                        }
                    ],
                    "subtitles": [
                        {"text": "This is a placeholder subtitle.", "start": 4, "end": 8}
                    ],
                    "notes": "Replace with real instructions when connected to the model.",
                }
            ],
        }
        if track:
            sample["instructions"][0]["music_cues"] = [
                {
                    "track": track,
                    "start": 0,
                    "end": 25,
                    "volume": 0.5,
                }
            ]
        return EditingPlan.from_dict(sample)
