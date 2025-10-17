from __future__ import annotations

from pathlib import Path

from ai_video_editor.editing_plan import EditInstruction, EditingPlan
from ai_video_editor.quality_reviewer import QualityReviewer


def sample_plan() -> EditingPlan:
    instructions = [
        EditInstruction(chunk="chunk_01.mp4", in_start=0.0, in_end=5.0, label="Intro"),
        EditInstruction(chunk="chunk_02.mp4", in_start=5.0, in_end=10.0, label="Action"),
    ]
    return EditingPlan(title="Test", summary="Test summary", instructions=instructions)


def test_quality_reviewer_generates_report(tmp_path: Path) -> None:
    plan = sample_plan()

    seg1 = tmp_path / "01_Intro.mp4"
    seg2 = tmp_path / "02_Action.mp4"
    final_video = tmp_path / "final_video.mp4"
    for path in (seg1, seg2, final_video):
        path.write_bytes(b"mock")

    reviewer = QualityReviewer(ffprobe="ffprobe")
    duration_map = {
        seg1.name: 5.1,
        seg2.name: 4.9,
        final_video.name: 10.05,
    }
    reviewer._probe_duration = lambda path: duration_map.get(path.name, 0.0)  # type: ignore[attr-defined]

    review = reviewer.review(plan, [seg1, seg2], final_video)

    assert review.score > 0.9
    assert review.verdict == "Pass"
    assert not review.issues
    data = review.to_dict()
    assert "score" in data and "verdict" in data
