from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ShotSegment:
    start: float
    end: float
    transcript: Optional[str] = None
    audio_highlights: List[str] = field(default_factory=list)
    frames: List[Path] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AudioSummary:
    mean_volume: Optional[float] = None
    max_volume: Optional[float] = None
    events: List[str] = field(default_factory=list)


@dataclass
class TimelineMoment:
    start: float
    end: float
    summary: str
    notes: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ShotNote:
    start: float
    end: float
    transcript: Optional[str]
    summary: str
    notable_objects: List[str] = field(default_factory=list)
    emotions: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class AudioEvent:
    time: float
    description: str
    confidence: Optional[float] = None


@dataclass
class PersonProfile:
    identifier: str
    appearance: str
    first_seen: float
    last_seen: float
    inferred_name: Optional[str] = None
    supporting_evidence: Optional[str] = None


@dataclass
class AnalysisResult:
    timeline: List[TimelineMoment]
    dull_sections: List[TimelineMoment]
    people: List[PersonProfile]
    shot_notes: List[ShotNote] = field(default_factory=list)
    audio_events: List[AudioEvent] = field(default_factory=list)
    overall_summary: Optional[str] = None
    raw_response: Optional[str] = None

    def to_markdown(self) -> str:
        lines = ["# Timeline"]
        for moment in self.timeline:
            lines.append(
                f"- `{moment.start:0.2f}s` → `{moment.end:0.2f}s`: {moment.summary}"
            )
            if moment.notes:
                lines.append(f"  - Notes: {moment.notes}")
            if moment.actions:
                lines.append(f"  - Actions: {', '.join(moment.actions)}")
            if moment.confidence is not None:
                lines.append(f"  - Confidence: {moment.confidence:.2f}")

        lines.append("")
        lines.append("# Low Activity / Trim Suggestions")
        if not self.dull_sections:
            lines.append("- None detected")
        else:
            for dull in self.dull_sections:
                lines.append(
                    f"- `{dull.start:0.2f}s` → `{dull.end:0.2f}s`: {dull.summary}"
                )
                if dull.notes:
                    lines.append(f"  - Notes: {dull.notes}")
                if dull.actions:
                    lines.append(f"  - Actions: {', '.join(dull.actions)}")
                if dull.confidence is not None:
                    lines.append(f"  - Confidence: {dull.confidence:.2f}")

        lines.append("")
        lines.append("# Shot Notes")
        if not self.shot_notes:
            lines.append("- Not captured")
        else:
            for note in self.shot_notes:
                lines.append(
                    f"- `{note.start:0.2f}s` → `{note.end:0.2f}s`: {note.summary}"
                )
                if note.transcript:
                    lines.append(f"  - Transcript: {note.transcript}")
                if note.notable_objects:
                    lines.append(
                        "  - Notable objects: " + ", ".join(note.notable_objects)
                    )
                if note.emotions:
                    lines.append(f"  - Emotions: {note.emotions}")
                if note.confidence is not None:
                    lines.append(f"  - Confidence: {note.confidence:.2f}")

        lines.append("")
        lines.append("# Audio Events")
        if not self.audio_events:
            lines.append("- None detected")
        else:
            for event in self.audio_events:
                lines.append(f"- `{event.time:0.2f}s`: {event.description}")
                if event.confidence is not None:
                    lines.append(f"  - Confidence: {event.confidence:.2f}")

        lines.append("")
        lines.append("# Overall Summary")
        if self.overall_summary:
            lines.append(self.overall_summary)
        else:
            lines.append("No overall summary provided.")

        lines.append("")
        lines.append("# Model Response")
        lines.append(self.raw_response or "_No raw response captured._")
        return "\n".join(lines)


class VideoAnalyzer:
    def analyze(
        self,
        video_path: Path,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary] = None,
    ) -> AnalysisResult:
        raise NotImplementedError

    def set_retry_hint(self, hint: Optional[str]) -> None:
        """Provide extra guidance for subsequent analyse calls."""
        _ = hint  # default noop

    def clear_retry_hint(self) -> None:
        """Clear any retry guidance."""
        self.set_retry_hint(None)

    @property
    def last_raw_response(self) -> Optional[str]:  # pragma: no cover - default noop
        return getattr(self, "_last_raw_response", None)


class AnalysisParseError(RuntimeError):
    """Raised when the model response cannot be parsed into structured analysis."""

    def __init__(self, message: str, *, raw_response: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
