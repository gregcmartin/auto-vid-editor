from __future__ import annotations

import logging
import json
import subprocess
import tempfile
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .audio_analysis import analyze_audio
from .base import (
    AnalysisResult,
    AudioSummary,
    PersonProfile,
    ShotSegment,
    VideoAnalyzer,
)
from .scene_detection import detect_scenes
from .transcription import WhisperTranscriber
from .validators import validate_analysis

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Run chunk analysis and persist markdown artifacts."""

    _FRAMES_PER_SHOT = 4
    _MAX_TOTAL_FRAMES = 64

    def __init__(
        self,
        analyzer: VideoAnalyzer,
        analysis_dir: Path,
        people_report_path: Path,
        *,
        enable_scene_detection: bool = True,
        scene_threshold: float = 0.4,
        enable_transcription: bool = False,
        whisper_model: str = "small",
        enable_audio_analysis: bool = True,
        analysis_retries: int = 0,
        fallback_analyzer: Optional[VideoAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer
        self.analysis_dir = analysis_dir
        self.people_report_path = people_report_path
        self.enable_scene_detection = enable_scene_detection
        self.scene_threshold = scene_threshold
        self.enable_audio_analysis = enable_audio_analysis
        self.transcriber: Optional[WhisperTranscriber] = None
        if enable_transcription:
            self.transcriber = WhisperTranscriber.try_create(whisper_model=whisper_model)
        self.analysis_retries = max(0, analysis_retries)
        self.fallback_analyzer = fallback_analyzer
        self._base_max_new_tokens = getattr(self.analyzer, "max_new_tokens", None)
        self._base_max_tokens = getattr(self.analyzer, "max_tokens", None)
        self._base_temperature = getattr(self.analyzer, "temperature", None)
        self._audio_tmpdir = tempfile.TemporaryDirectory() if self.transcriber else None
        self._chunk_audio_cache: Dict[Path, Optional[Path]] = {}

    def __del__(self) -> None:  # pragma: no cover - cleanup guard
        if self._audio_tmpdir is not None:
            try:
                self._audio_tmpdir.cleanup()
            except Exception:
                pass

    def run(self, chunks: Sequence[Path]) -> None:
        logger.info("Starting analysis pipeline for %d chunks", len(chunks))
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        people_index: Dict[str, List[PersonProfile]] = {}

        for chunk in chunks:
            duration = _probe_duration(chunk)
            shots = self._create_shots(chunk, duration_hint=duration)
            audio_summary = self._create_audio_summary(chunk)
            if audio_summary and audio_summary.events:
                for shot in shots:
                    shot.audio_highlights.extend(audio_summary.events)
                    shot.audio_highlights = list(dict.fromkeys(shot.audio_highlights))

            if hasattr(self.analyzer, "clear_retry_hint"):
                try:
                    self.analyzer.clear_retry_hint()
                except Exception:
                    pass
            result: Optional[AnalysisResult] = None
            warnings: List[str] = []
            for attempt in range(self.analysis_retries + 1):
                try:
                    result = self.analyzer.analyze(chunk, shots, audio_summary)
                except Exception as exc:
                    logger.error("Primary analyser failed for %s: %s", chunk.name, exc)
                    raw = getattr(self.analyzer, "last_raw_response", None)
                    if raw:
                        self._record_failed_attempt(chunk, attempt, raw_response=raw)
                        self._update_retry_hint(["non_json_response"])
                    result = None
                is_valid = False
                if result is not None:
                    is_valid, warnings = validate_analysis(result, duration)
                if not is_valid and result is not None:
                    self._record_failed_attempt(chunk, attempt, result)
                    self._update_retry_hint(warnings)
                if is_valid or attempt == self.analysis_retries:
                    if is_valid:
                        self._clear_retry_hint()
                    break
                logger.info(
                    "Analysis validation failed for %s (attempt %d/%d). Retrying with adjusted settings.",
                    chunk.name,
                    attempt + 1,
                    self.analysis_retries + 1,
                )
                self._prepare_retry(attempt)

            critical_warnings = [
                w
                for w in warnings
                if "Timeline" in w or "Overall summary" in w
            ]

            if (result is None or critical_warnings) and self.fallback_analyzer is not None:
                logger.info("Falling back to secondary analyser for %s", chunk.name)
                try:
                    result = self.fallback_analyzer.analyze(chunk, shots, audio_summary)
                    _, warnings = validate_analysis(result, duration)
                    self._clear_retry_hint()
                except Exception as exc:
                    logger.error("Fallback analyser failed for %s: %s", chunk.name, exc)
                    result = self._synthesise_default_analysis(
                        chunk,
                        shots,
                        audio_summary,
                        duration,
                        fallback_error=str(exc),
                    )
                    warnings = [f"Fallback analyser failed: {exc}"]

            if warnings:
                for message in warnings:
                    logger.warning("%s: %s", chunk.name, message)

            if result is None:
                logger.error("No analysis produced for %s; synthesising minimal result", chunk.name)
                result = self._synthesise_default_analysis(
                    chunk,
                    shots,
                    audio_summary,
                    duration,
                )
            chunk_report = self.analysis_dir / f"{chunk.stem}.md"
            logger.debug("Writing analysis: %s", chunk_report)
            self._write_chunk_report(chunk_report, chunk, result)
            self._write_shot_metadata(chunk, shots, audio_summary)

            if result.people:
                people_index[chunk.name] = result.people
            self._restore_analyzer_defaults()
            
            # Clear MPS cache after each chunk to prevent memory buildup
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS cache after chunk %s", chunk.name)
            except Exception:
                pass

        self._write_people_report(people_index)

    def _create_shots(self, chunk: Path, duration_hint: Optional[float] = None) -> List[ShotSegment]:
        if self.enable_scene_detection:
            segments = detect_scenes(chunk, threshold=self.scene_threshold)
        else:
            duration = duration_hint if duration_hint is not None else _probe_duration(chunk)
            if duration <= 0.0:
                segments = [(0.0, 0.0)]
            else:
                segments = [(0.0, duration)]
        frames_root = self.analysis_dir / "frames" / chunk.stem
        if frames_root.exists():
            shutil.rmtree(frames_root, ignore_errors=True)
        frames_root.mkdir(parents=True, exist_ok=True)

        audio_source = self._get_chunk_audio(chunk) if self.transcriber else None
        shots: List[ShotSegment] = []
        total_frames_budget = self._MAX_TOTAL_FRAMES
        for index, (start, end) in enumerate(segments, start=1):
            transcript = self._transcribe(chunk, start, end, audio_source)
            frames: List[Path] = []
            if total_frames_budget > 0:
                frames_needed = min(self._FRAMES_PER_SHOT, total_frames_budget)
                frames = self._extract_frames(
                    chunk,
                    frames_root,
                    index,
                    start,
                    end,
                    frames_needed if frames_needed > 0 else 1,
                )
                total_frames_budget = max(0, total_frames_budget - len(frames))
            shots.append(
                ShotSegment(
                    start=start,
                    end=end,
                    transcript=transcript,
                    frames=frames,
                )
            )
        return shots

    def _transcribe(
        self,
        chunk: Path,
        start: float,
        end: float,
        audio_source: Optional[Path],
    ) -> Optional[str]:
        if not self.transcriber:
            return None
        try:
            text = self.transcriber.transcribe_segment(
                chunk,
                start,
                end,
                audio_source=audio_source,
            )
            return text or None
        except Exception as exc:
            logger.warning("Transcription failed for %s [%0.2f-%0.2f]: %s", chunk, start, end, exc)
            return None

    def _create_audio_summary(self, chunk: Path) -> Optional[AudioSummary]:
        if not self.enable_audio_analysis:
            return None
        summary = analyze_audio(chunk)
        if summary.mean_volume is None and summary.max_volume is None and not summary.events:
            return None
        return summary

    def _get_chunk_audio(self, chunk: Path) -> Optional[Path]:
        if self._audio_tmpdir is None:
            return None
        cached = self._chunk_audio_cache.get(chunk)
        if cached is not None:
            return cached

        target = Path(self._audio_tmpdir.name) / f"{chunk.stem}_16k.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(chunk),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logger.error(
                "Failed to extract chunk audio (exit %s). Command: %s\n%s",
                result.returncode,
                " ".join(cmd),
                result.stderr.decode().strip(),
            )
            self._chunk_audio_cache[chunk] = None
            return None

        self._chunk_audio_cache[chunk] = target
        return target

    def _write_shot_metadata(
        self,
        chunk_path: Path,
        shots: List[ShotSegment],
        audio_summary: Optional[AudioSummary],
    ) -> None:
        metadata = {
            "shots": [
                {
                    "start": shot.start,
                    "end": shot.end,
                    "transcript": shot.transcript,
                    "audio_highlights": shot.audio_highlights,
                    "frames": [
                        self._relative_to_analysis_dir(frame)
                        for frame in shot.frames
                    ],
                }
                for shot in shots
            ],
            "audio_summary": asdict(audio_summary) if audio_summary else None,
        }
        path = self.analysis_dir / f"{chunk_path.stem}_shots.json"
        path.write_text(json.dumps(metadata, indent=2))

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

    def _record_failed_attempt(
        self,
        chunk_path: Path,
        attempt: int,
        result: Optional[AnalysisResult] = None,
        *,
        raw_response: Optional[str] = None,
    ) -> None:
        raw_payload: Optional[str] = raw_response
        if raw_payload is None and result is not None:
            raw_payload = result.raw_response
        if not raw_payload:
            return
        debug_path = self.analysis_dir / f"{chunk_path.stem}_attempt{attempt+1}_raw.json"
        try:
            debug_path.write_text(raw_payload)
        except Exception:
            logger.debug("Unable to write raw response for %s attempt %d", chunk_path.name, attempt + 1)

    def _update_retry_hint(self, warnings: List[str]) -> None:
        hint_segments: List[str] = []
        for message in warnings:
            lower = message.lower()
            if "timeline empty" in lower:
                hint_segments.append(
                    "Provide at least one timeline entry covering the clip, even if the action is minimal."
                )
            if "overall summary missing" in lower:
                hint_segments.append(
                    "Add a concise overall_summary describing the clip in plain text."
                )
            if "coverage" in lower:
                hint_segments.append(
                    "Ensure timeline coverage spans the full clip duration with realistic timestamps."
                )
            if "non_json" in lower or "json" in lower:
                hint_segments.append("Respond with valid JSON only; avoid commentary or explanations.")
        if not hint_segments:
            return
        hint = " ".join(dict.fromkeys(hint_segments))
        for analyzer in (self.analyzer, self.fallback_analyzer):
            if analyzer is None:
                continue
            try:
                analyzer.set_retry_hint(hint)
            except AttributeError:
                continue

    def _clear_retry_hint(self) -> None:
        for analyzer in (self.analyzer, self.fallback_analyzer):
            if analyzer is None:
                continue
            try:
                analyzer.clear_retry_hint()
            except AttributeError:
                continue

    def _synthesise_default_analysis(
        self,
        chunk: Path,
        shots: Sequence[ShotSegment],
        audio_summary: Optional[AudioSummary],
        duration: float,
        fallback_error: Optional[str] = None,
    ) -> AnalysisResult:
        def _summarise_text(text: Optional[str]) -> str:
            if not text:
                return "No transcript available."
            cleaned = " ".join(text.strip().split())
            return cleaned[:240] + ("…" if len(cleaned) > 240 else "")

        timeline: List[TimelineMoment] = []
        shot_notes: List[ShotNote] = []

        for shot in shots:
            summary = _summarise_text(shot.transcript)
            notes = None
            if shot.audio_highlights:
                notes = "; ".join(dict.fromkeys(shot.audio_highlights))
            timeline.append(
                TimelineMoment(
                    start=shot.start,
                    end=shot.end,
                    summary=summary,
                    notes=notes,
                    actions=[],
                    confidence=None,
                )
            )
            shot_notes.append(
                ShotNote(
                    start=shot.start,
                    end=shot.end,
                    transcript=shot.transcript,
                    summary=summary,
                    notable_objects=[],
                    emotions=None,
                    confidence=None,
                )
            )

        if not timeline:
            duration = max(duration, 0.0)
            timeline.append(
                TimelineMoment(
                    start=0.0,
                    end=duration if duration > 0 else 0.0,
                    summary="Clip segment with limited metadata.",
                    notes=None,
                    actions=[],
                    confidence=None,
                )
            )

        audio_events: List[AudioEvent] = []
        if audio_summary and audio_summary.events:
            reference_times: List[float] = []
            if shots:
                for shot in shots:
                    reference_times.append(max(0.0, min(duration, (shot.start + shot.end) / 2.0)))
            else:
                reference_times.append(max(0.0, min(duration, duration / 2 if duration else 0.0)))
            for idx, event in enumerate(audio_summary.events):
                time = reference_times[idx % len(reference_times)]
                audio_events.append(
                    AudioEvent(
                        time=time,
                        description=event,
                        confidence=None,
                    )
                )

        overall_summary = timeline[0].summary if timeline else "Clip segment"
        raw_response = None
        if fallback_error:
            raw_response = f"SYNTHESISED DUE TO ERROR: {fallback_error}"

        return AnalysisResult(
            timeline=timeline,
            dull_sections=[],
            people=[],
            shot_notes=shot_notes,
            audio_events=audio_events,
            overall_summary=overall_summary,
            raw_response=raw_response,
        )

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

    def _prepare_retry(self, attempt: int) -> None:
        growth = 1.5
        MAX_NEW_TOKENS = 3072
        MAX_TOKENS = 4096
        if hasattr(self.analyzer, "max_new_tokens"):
            try:
                current = getattr(self.analyzer, "max_new_tokens")
                new_value = min(MAX_NEW_TOKENS, max(int(current * growth), current + 128))
                setattr(self.analyzer, "max_new_tokens", new_value)
            except Exception:
                pass
        if hasattr(self.analyzer, "max_tokens"):
            try:
                current = getattr(self.analyzer, "max_tokens")
                new_value = min(MAX_TOKENS, max(int(current * growth), current + 128))
                setattr(self.analyzer, "max_tokens", new_value)
            except Exception:
                pass
        if hasattr(self.analyzer, "temperature"):
            try:
                current = getattr(self.analyzer, "temperature")
                new_value = max(0.05, current * 0.8)
                setattr(self.analyzer, "temperature", new_value)
            except Exception:
                pass

    def _restore_analyzer_defaults(self) -> None:
        if self._base_max_new_tokens is not None:
            try:
                setattr(self.analyzer, "max_new_tokens", self._base_max_new_tokens)
            except Exception:
                pass
        if self._base_max_tokens is not None:
            try:
                setattr(self.analyzer, "max_tokens", self._base_max_tokens)
            except Exception:
                pass
        if self._base_temperature is not None:
            try:
                setattr(self.analyzer, "temperature", self._base_temperature)
            except Exception:
                pass

    def _extract_frames(
        self,
        chunk: Path,
        frames_root: Path,
        shot_index: int,
        start: float,
        end: float,
        frame_targets: int,
    ) -> List[Path]:
        frame_targets = max(1, frame_targets)
        duration = max(0.0, end - start)
        offsets: List[float]
        if duration <= 0.0:
            offsets = [start]
        else:
            step = duration / (frame_targets + 1)
            offsets = [start + step * (i + 1) for i in range(frame_targets)]
        frame_paths: List[Path] = []
        for frame_idx, timestamp in enumerate(offsets, start=1):
            safe_ts = max(0.0, min(timestamp, end if duration > 0 else start))
            target = frames_root / f"{chunk.stem}_s{shot_index:02d}_f{frame_idx:02d}.jpg"
            if target.exists():
                frame_paths.append(target)
                continue
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{safe_ts:.3f}",
                "-i",
                str(chunk),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-1",
                "-y",
                str(target),
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.warning(
                    "Failed to sample frame for %s shot %d (ts=%0.3f): %s",
                    chunk.name,
                    shot_index,
                    safe_ts,
                    result.stderr.decode().strip(),
                )
                continue
            frame_paths.append(target)
        return frame_paths

    def _relative_to_analysis_dir(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.analysis_dir))
        except ValueError:
            return str(path)

def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
