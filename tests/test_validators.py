from __future__ import annotations

from ai_video_editor.analysis.base import (
    AnalysisResult,
    AudioEvent,
    PersonProfile,
    ShotNote,
    TimelineMoment,
)
from ai_video_editor.analysis.validators import validate_analysis


def test_validate_analysis_accepts_grounded_result() -> None:
    """Validator should accept a well-formed, grounded analysis result."""
    result = AnalysisResult(
        timeline=[
            TimelineMoment(
                start=0.0,
                end=5.0,
                summary="Runner jogs past the camera.",
                confidence=0.9,
                actions=["jogging"],
            )
        ],
        dull_sections=[],
        people=[
            PersonProfile(
                identifier="Runner",
                appearance="Adult in blue jacket jogging left to right.",
                first_seen=0.0,
                last_seen=5.0,
                inferred_name=None,
                supporting_evidence="Appears throughout the shot.",
            )
        ],
        shot_notes=[
            ShotNote(
                start=0.0,
                end=5.0,
                transcript="Keep the pace steady.",
                summary="Motivational cue during the run.",
            )
        ],
        audio_events=[
            AudioEvent(
                time=1.5,
                description="Footsteps grow louder near the camera.",
                confidence=0.8,
            )
        ],
        overall_summary="Runner passes the camera with steady pace.",
    )

    is_valid, messages = validate_analysis(result, chunk_duration=10.0)
    assert is_valid, messages
    assert not any(msg.startswith("ERROR") for msg in messages)


def test_validate_analysis_rejects_out_of_bounds_and_placeholder_people() -> None:
    """Validator should reject hallucinated segments and placeholder people."""
    result = AnalysisResult(
        timeline=[
            TimelineMoment(
                start=0.0,
                end=0.0,
                summary="Made up event.",
            )
        ],
        dull_sections=[],
        people=[
            PersonProfile(
                identifier="Person A",
                appearance="Standing",
                first_seen=-1.0,
                last_seen=12.0,
                inferred_name=None,
                supporting_evidence=["not a string"],  # type: ignore[list-item]
            )
        ],
        shot_notes=[],
        audio_events=[
            AudioEvent(
                time=20.0,
                description="Imaginary audio cue.",
            )
        ],
        overall_summary="",
    )

    is_valid, messages = validate_analysis(result, chunk_duration=5.0)
    assert not is_valid
    assert any(msg.startswith("ERROR") for msg in messages)
    assert any("placeholder" in msg.lower() for msg in messages)
