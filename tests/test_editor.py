from __future__ import annotations

from pathlib import Path

from ai_video_editor.editing_plan import EditingPlan, EditInstruction, MusicCue
from ai_video_editor.editor import FFmpegEditor, FFmpegEditorError, VideoAssembler


def build_plan(
    chunk_name: str,
    *,
    keep_audio: bool = False,
    music_cues: list[MusicCue] | None = None,
) -> EditingPlan:
    instruction = EditInstruction(
        chunk=chunk_name,
        in_start=0.0,
        in_end=5.0,
        label="Clip",
        speed=1.0,
        keep_audio=keep_audio,
        music_cues=music_cues or [],
    )
    return EditingPlan(
        title="Plan",
        summary="Test summary",
        instructions=[instruction],
    )


def test_editor_dry_run_creates_placeholder(tmp_path: Path):
    chunk = tmp_path / "chunk_01.mp4"
    chunk.write_bytes(b"placeholder")

    plan = build_plan(chunk.name)
    editor = FFmpegEditor(dry_run=True)
    outputs = editor.render(plan, {chunk.name: chunk}, tmp_path / "rendered")

    assert len(outputs) == 1
    output_path = outputs[0]
    assert output_path.exists()
    assert "dry-run" in output_path.read_text()

    assembler = VideoAssembler(dry_run=True)
    final_path = tmp_path / "final.mp4"
    assembler.assemble(outputs, final_path)
    assert final_path.exists()
    assert "dry-run final" in final_path.read_text()


def test_editor_raises_on_missing_chunk(tmp_path: Path):
    plan = build_plan("missing.mp4")
    editor = FFmpegEditor(dry_run=True)
    try:
        editor.render(plan, {}, tmp_path / "rendered")
        assert False, "Expected FFmpegEditorError"
    except FFmpegEditorError:
        pass


def test_editor_music_cue_dry_run(tmp_path: Path):
    chunk = tmp_path / "chunk_01.mp4"
    chunk.write_bytes(b"placeholder")

    music_dir = tmp_path / "music"
    music_dir.mkdir()
    track = music_dir / "loop.mp3"
    track.write_bytes(b"\x00" * 10)

    plan = build_plan(
        chunk.name,
        music_cues=[MusicCue(track=track.name, start=0.0, end=5.0, volume=0.7)],
    )
    editor = FFmpegEditor(dry_run=True, music_dir=music_dir)
    outputs = editor.render(plan, {chunk.name: chunk}, tmp_path / "rendered")
    assert outputs[0].read_text().strip() == "dry-run music mix"
