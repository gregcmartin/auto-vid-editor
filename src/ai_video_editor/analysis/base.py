from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TimelineMoment:
    start: float
    end: float
    summary: str
    notes: Optional[str] = None
    actions: List[str] = field(default_factory=list)


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

        lines.append("")
        lines.append("# Low Activity / Trim Suggestions")
        if not self.dull_sections:
            lines.append("- None detected")
        else:
            for dull in self.dull_sections:
                lines.append(
                    f"- `{dull.start:0.2f}s` → `{dull.end:0.2f}s`: {dull.summary}"
                )

        lines.append("")
        lines.append("# Model Response")
        lines.append(self.raw_response or "_No raw response captured._")
        return "\n".join(lines)


class VideoAnalyzer:
    def analyze(self, video_path: Path) -> AnalysisResult:
        raise NotImplementedError
