from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    video_path: Path
    root_output: Path
    chunks_dir: Path
    analysis_dir: Path
    people_report: Path
    plan_dir: Path
    plan_json: Path
    plan_markdown: Path
    edited_dir: Path
    final_output: Path
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    font_path: Path | None = None
    music_dir: Path | None = None

    @classmethod
    def from_paths(
        cls,
        video_path: Path,
        root_output: Path,
        font_path: Path | None = None,
        music_dir: Path | None = None,
    ) -> "AppConfig":
        return cls(
            video_path=video_path,
            root_output=root_output,
            chunks_dir=root_output / "chunks",
            analysis_dir=root_output / "analysis",
            people_report=root_output / "analysis" / "people.md",
            plan_dir=root_output / "plan",
            plan_json=root_output / "plan" / "plan.json",
            plan_markdown=root_output / "plan" / "plan.md",
            edited_dir=root_output / "new",
            final_output=root_output / "final_video.mp4",
            font_path=font_path,
            music_dir=music_dir,
        )
