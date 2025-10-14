from __future__ import annotations

import re
from typing import List, Tuple

from .base import AnalysisResult


def validate_analysis(
    result: AnalysisResult,
    chunk_duration: float,
    min_timeline_coverage: float = 0.25,
) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    duration = max(chunk_duration, 0.01)

    if not result.timeline:
        errors.append("Timeline empty")

    if result.timeline:
        coverage = _coverage_ratio(result.timeline, duration)
        if coverage < min_timeline_coverage:
            errors.append(
                f"Timeline coverage only {coverage*100:.1f}% of chunk duration"
            )

    # Validate timeline and dull sections
    for collection_name, moments in (
        ("timeline", result.timeline),
        ("dull_sections", result.dull_sections),
    ):
        for idx, moment in enumerate(moments):
            context = f"{collection_name}[{idx}]"
            if moment.end <= moment.start:
                errors.append(
                    f"{context} has non-positive duration ({moment.start}-{moment.end})"
                )
            if moment.start < -0.05 or moment.end > duration + 0.05:
                errors.append(
                    f"{context} lies outside chunk bounds ({moment.start}-{moment.end} vs duration {duration:.2f})"
                )
            if not isinstance(moment.summary, str) or not moment.summary.strip():
                warnings.append(f"{context} summary is missing or not a string")
            if moment.notes is not None and not isinstance(moment.notes, str):
                warnings.append(f"{context} notes must be a string or null")
            if moment.actions and not all(isinstance(action, str) for action in moment.actions):
                warnings.append(f"{context} actions must be strings")

    # Detect suspiciously uniform timeline durations (hallucination indicator)
    durations = [moment.duration for moment in result.timeline if moment.duration > 0]
    if len(durations) >= 4:
        first = durations[0]
        if all(abs(d - first) <= 0.15 for d in durations):
            warnings.append("Timeline durations appear uniformly spaced; possible hallucination")
    if durations and len(result.timeline) > max(12, int(duration * 2)):
        warnings.append(
            f"Timeline contains {len(result.timeline)} entries for {duration:.1f}s chunk; review for over-segmentation"
        )

    # Validate people
    hallucinated_identifiers = 0
    for idx, person in enumerate(result.people):
        context = f"people[{idx}]"
        if person.first_seen < -0.05 or person.last_seen > duration + 0.05:
            errors.append(
                f"{context} timing outside bounds ({person.first_seen}-{person.last_seen})"
            )
        if person.last_seen < person.first_seen:
            errors.append(
                f"{context} last_seen earlier than first_seen ({person.first_seen}-{person.last_seen})"
            )
        if not isinstance(person.appearance, str) or not person.appearance.strip():
            warnings.append(f"{context} appearance must be a non-empty string")
        identifier_normalized = person.identifier.strip().lower()
        if re.fullmatch(r"person [a-z0-9]+", identifier_normalized):
            hallucinated_identifiers += 1
        if person.supporting_evidence is not None and not isinstance(person.supporting_evidence, str):
            warnings.append(f"{context} supporting_evidence should be a string when present")
    if hallucinated_identifiers >= 3:
        warnings.append(
            f"Detected {hallucinated_identifiers} placeholder identifiers (e.g., 'Person A'); likely hallucinated people"
        )

    # Validate shot notes
    for idx, note in enumerate(result.shot_notes):
        context = f"shot_notes[{idx}]"
        if note.end <= note.start:
            warnings.append(f"{context} has non-positive duration")
        if note.start < -0.05 or note.end > duration + 0.05:
            warnings.append(
                f"{context} lies outside chunk bounds ({note.start}-{note.end})"
            )
        if note.transcript is not None and not isinstance(note.transcript, str):
            warnings.append(f"{context} transcript must be a string or null")

    # Validate audio events
    for idx, event in enumerate(result.audio_events):
        if event.time < -0.05 or event.time > duration + 0.05:
            warnings.append(
                f"audio_events[{idx}] timestamp {event.time} outside chunk bounds"
            )

    if result.overall_summary in (None, ""):
        errors.append("Overall summary missing")

    messages = [f"ERROR: {msg}" for msg in errors] + warnings
    return len(errors) == 0, messages


def _coverage_ratio(timeline, duration: float) -> float:
    total = 0.0
    for moment in timeline:
        start = max(0.0, moment.start)
        end = max(start, moment.end)
        total += min(duration, end) - start
    return min(1.0, total / duration)
