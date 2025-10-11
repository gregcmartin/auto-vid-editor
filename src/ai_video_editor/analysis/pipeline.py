from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .base import AnalysisResult, PersonProfile, VideoAnalyzer

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Run chunk analysis and persist markdown artifacts."""

    def __init__(
        self,
        analyzer: VideoAnalyzer,
        analysis_dir: Path,
        people_report_path: Path,
    ) -> None:
        self.analyzer = analyzer
        self.analysis_dir = analysis_dir
        self.people_report_path = people_report_path

    def run(self, chunks: Sequence[Path]) -> None:
        logger.info("Starting analysis pipeline for %d chunks", len(chunks))
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        people_index: Dict[str, List[PersonProfile]] = {}

        for chunk in chunks:
            result = self.analyzer.analyze(chunk)
            chunk_report = self.analysis_dir / f"{chunk.stem}.md"
            logger.debug("Writing analysis: %s", chunk_report)
            self._write_chunk_report(chunk_report, chunk, result)

            if result.people:
                people_index[chunk.name] = result.people

        self._write_people_report(people_index)

    def _write_chunk_report(
        self,
        output_path: Path,
        chunk_path: Path,
        result: AnalysisResult,
    ) -> None:
        lines = [
            f"# Chunk Analysis — {chunk_path.name}",
            "",
            result.to_markdown(),
        ]
        output_path.write_text("\n".join(lines))

    def _write_people_report(
        self,
        people_index: Dict[str, List[PersonProfile]],
    ) -> None:
        lines = ["# People Detected"]
        if not people_index:
            lines.append("No people detected in analysed chunks.")
        else:
            for chunk_name, profiles in people_index.items():
                lines.append(f"## {chunk_name}")
                for profile in profiles:
                    lines.append(f"- **{profile.identifier}**")
                    lines.append(f"  - Appearance: {profile.appearance}")
                    lines.append(
                        f"  - Seen between `{profile.first_seen:0.2f}s` and `{profile.last_seen:0.2f}s`"
                    )
                    if profile.inferred_name:
                        lines.append(f"  - Inferred name: {profile.inferred_name}")
                    if profile.supporting_evidence:
                        lines.append(f"  - Evidence: {profile.supporting_evidence}")
                lines.append("")
        self.people_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.people_report_path.write_text("\n".join(lines).strip() + "\n")

