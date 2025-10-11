from __future__ import annotations

from ai_video_editor.editing_plan import (
    CropSpec,
    EditingPlan,
    EditInstruction,
    MusicCue,
    SubtitleLine,
    TextOverlay,
)


def sample_plan() -> EditingPlan:
    instruction = EditInstruction(
        chunk="chunk_01.mp4",
        in_start=0.0,
        in_end=10.0,
        label="Intro",
        speed=1.25,
        crop=CropSpec(width=1280, height=720, x=0, y=0),
        text_overlays=[
            TextOverlay(text="Hello", start=1.0, end=3.0, position="top_left")
        ],
        subtitles=[
            SubtitleLine(text="Mock line", start=2.0, end=4.0),
        ],
        music_cues=[
            MusicCue(track="ambient-background-loop-chill.mp3", start=0.0, end=8.0, volume=0.5)
        ],
        notes="Keep the energy high.",
    )
    return EditingPlan(
        title="Test Plan",
        summary="Summary here",
        instructions=[instruction],
        outro_text="Thanks for watching.",
        voiceover_notes="Record a punchy intro.",
        warnings=["Check audio quality"],
    )


def test_editing_plan_roundtrip():
    plan = sample_plan()
    data = plan.to_dict()
    restored = EditingPlan.from_dict(data)
    assert restored.title == plan.title
    assert restored.instructions[0].text_overlays[0].text == "Hello"
    assert restored.instructions[0].subtitles[0].text == "Mock line"
    assert restored.instructions[0].music_cues[0].track.endswith(".mp3")
    assert restored.outro_text == plan.outro_text


def test_editing_plan_markdown_contains_sections():
    plan = sample_plan()
    markdown = plan.to_markdown()
    assert "# Test Plan" in markdown
    assert "## Instructions" in markdown
    assert "Intro" in markdown
    assert "Subtitle lines" in markdown
    assert "Music cues" in markdown
