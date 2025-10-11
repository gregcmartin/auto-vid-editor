from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CropSpec:
    """Simple crop configuration."""

    width: int
    height: int
    x: int
    y: int


@dataclass
class TextOverlay:
    """Describes a text overlay using ffmpeg drawtext semantics."""

    text: str
    start: float
    end: float
    position: str = "center"  # center, top_left, top_right, bottom_left, bottom_right
    style: Optional[str] = None  # free-form hints, e.g. "bold,24px"


@dataclass
class SubtitleLine:
    text: str
    start: float
    end: float


@dataclass
class MusicCue:
    """Represents a background music clip to blend into a segment."""

    track: str  # filename relative to the music library directory
    start: float  # seconds from the segment start
    end: float  # seconds from the segment start
    volume: float = 0.6  # linear scale (1.0 = unchanged)


@dataclass
class EditInstruction:
    """Represents one output clip derived from a source chunk."""

    chunk: str  # filename (without path) of the source chunk
    in_start: float
    in_end: float
    label: Optional[str] = None
    speed: float = 1.0
    keep_audio: bool = True
    crop: Optional[CropSpec] = None
    text_overlays: List[TextOverlay] = field(default_factory=list)
    subtitles: List[SubtitleLine] = field(default_factory=list)
    music_cues: List[MusicCue] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class EditingPlan:
    """Full editing plan produced by the director model."""

    title: str
    summary: str
    instructions: List[EditInstruction]
    outro_text: Optional[str] = None
    voiceover_notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "summary": self.summary,
            "outro_text": self.outro_text,
            "voiceover_notes": self.voiceover_notes,
            "warnings": self.warnings,
            "instructions": [self._instruction_to_dict(item) for item in self.instructions],
        }

    @staticmethod
    def _instruction_to_dict(item: EditInstruction) -> Dict[str, object]:
        return {
            "chunk": item.chunk,
            "in_start": item.in_start,
            "in_end": item.in_end,
            "label": item.label,
            "speed": item.speed,
            "keep_audio": item.keep_audio,
            "crop": vars(item.crop) if item.crop else None,
            "text_overlays": [vars(overlay) for overlay in item.text_overlays],
            "subtitles": [vars(line) for line in item.subtitles],
            "music_cues": [vars(cue) for cue in item.music_cues],
            "notes": item.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> EditingPlan:
        instructions_data = data.get("instructions", [])
        instructions: List[EditInstruction] = []
        for raw in instructions_data:
            crop_data = raw.get("crop")
            crop = CropSpec(**crop_data) if crop_data else None
            overlays = [TextOverlay(**overlay) for overlay in raw.get("text_overlays", [])]
            subtitles = [SubtitleLine(**line) for line in raw.get("subtitles", [])]
            music_cues = [MusicCue(**cue) for cue in raw.get("music_cues", [])]
            instructions.append(
                EditInstruction(
                    chunk=raw["chunk"],
                    in_start=float(raw["in_start"]),
                    in_end=float(raw["in_end"]),
                    label=raw.get("label"),
                    speed=float(raw.get("speed", 1.0)),
                    keep_audio=bool(raw.get("keep_audio", True)),
                    crop=crop,
                    text_overlays=overlays,
                    subtitles=subtitles,
                    music_cues=music_cues,
                    notes=raw.get("notes"),
                )
            )
        return cls(
            title=data.get("title", "Untitled Edit"),
            summary=data.get("summary", ""),
            instructions=instructions,
            outro_text=data.get("outro_text"),
            voiceover_notes=data.get("voiceover_notes"),
            warnings=list(data.get("warnings", [])),
        )

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.summary, ""]
        if self.warnings:
            lines.append("## Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        lines.append("## Instructions")
        for index, instruction in enumerate(self.instructions, start=1):
            heading = instruction.label or f"Segment {index}"
            lines.append(f"### {index}. {heading}")
            lines.append(f"- Source chunk: `{instruction.chunk}`")
            lines.append(
                f"- Trim: `{instruction.in_start:.2f}s` → `{instruction.in_end:.2f}s`"
            )
            if instruction.speed != 1.0:
                lines.append(f"- Playback speed: `{instruction.speed:0.2f}x`")
            if instruction.crop:
                crop = instruction.crop
                lines.append(
                    f"- Crop: `{crop.width}x{crop.height}` at ({crop.x}, {crop.y})"
                )
            if instruction.text_overlays:
                lines.append("- Text overlays:")
                for overlay in instruction.text_overlays:
                    lines.append(
                        f"  - `{overlay.text}` ({overlay.start:.2f}s → {overlay.end:.2f}s, pos={overlay.position})"
                    )
            if instruction.subtitles:
                lines.append(f"- Subtitle lines: {len(instruction.subtitles)}")
            if instruction.music_cues:
                lines.append("- Music cues:")
                for cue in instruction.music_cues:
                    lines.append(
                        f"  - `{cue.track}` from {cue.start:.2f}s → {cue.end:.2f}s @ volume {cue.volume:.2f}"
                    )
            if instruction.notes:
                lines.append(f"- Notes: {instruction.notes}")
            lines.append("")
        if self.outro_text:
            lines.append("## Outro")
            lines.append(self.outro_text)
            lines.append("")
        if self.voiceover_notes:
            lines.append("## Voiceover Notes")
            lines.append(self.voiceover_notes)
            lines.append("")
        return "\n".join(lines).strip() + "\n"
