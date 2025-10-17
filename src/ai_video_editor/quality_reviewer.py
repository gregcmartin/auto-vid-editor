from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .editing_plan import EditInstruction, EditingPlan

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    message: str
    severity: str = "warning"  # warning | error | info


@dataclass
class QualityReview:
    score: float
    verdict: str
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    raw_model_response: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "issues": [issue.__dict__ for issue in self.issues],
            "recommendations": self.recommendations,
            "raw_model_response": self.raw_model_response,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Quality Review",
            "",
            f"**Score:** {self.score:.2f}",
            f"**Verdict:** {self.verdict}",
            "",
        ]
        if self.issues:
            lines.append("## Issues")
            for issue in self.issues:
                lines.append(f"- ({issue.severity}) {issue.message}")
            lines.append("")
        if self.recommendations:
            lines.append("## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        if self.raw_model_response:
            lines.append("## Judge Notes")
            lines.append(self.raw_model_response.strip())
            lines.append("")
        return "\n".join(lines).strip() + "\n"


class QualityReviewer:
    """Evaluate the rendered clips and final output against the editing plan."""

    def __init__(
        self,
        *,
        ffprobe: str = "ffprobe",
        judge: Optional[Any] = None,
        judge_name: Optional[str] = None,
    ) -> None:
        self.ffprobe = ffprobe
        self.judge = judge
        self.judge_name = judge_name or (judge.__class__.__name__ if judge else None)

    def review(
        self,
        plan: EditingPlan,
        segment_paths: Sequence[Path],
        final_output: Path,
    ) -> QualityReview:
        metrics, issues = self._gather_metrics(plan.instructions, segment_paths, final_output)
        base_score, baseline_recommendations = self._score_metrics(metrics, issues)

        verdict = "Pass" if base_score >= 0.75 and not any(i.severity == "error" for i in issues) else "Needs attention"
        recommendations = baseline_recommendations
        raw_model_response = None
        score = base_score

        if self.judge is not None:
            try:
                judge_payload = self._build_judge_payload(plan, metrics, verdict, base_score)
                raw_model_response = self._run_model_judge(judge_payload)
                model_score, model_verdict, model_issues, model_recs = self._parse_model_judgement(raw_model_response)
                if model_score is not None:
                    score = model_score
                if model_verdict:
                    verdict = model_verdict
                if model_issues:
                    issues.extend(model_issues)
                if model_recs:
                    recommendations.extend(model_recs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Quality judge failed (%s); falling back to heuristic verdict.", exc, exc_info=True)

        return QualityReview(
            score=max(0.0, min(1.0, score)),
            verdict=verdict,
            issues=issues,
            recommendations=recommendations,
            raw_model_response=raw_model_response,
        )

    # ------------------------------------------------------------------ judge orchestration

    def _build_judge_payload(
        self,
        plan: EditingPlan,
        metrics: List[dict[str, Any]],
        heuristic_verdict: str,
        heuristic_score: float,
    ) -> str:
        payload = {
            "plan_title": plan.title,
            "plan_summary": plan.summary,
            "heuristic_score": round(heuristic_score, 3),
            "heuristic_verdict": heuristic_verdict,
            "segments": metrics,
        }
        return json.dumps(payload, indent=2)

    def _run_model_judge(self, payload: str) -> str:
        system_prompt = (
            "You are an experienced post-production supervisor. "
            "Review the provided plan summary and segment metrics. "
            "Return JSON with keys score (0-1), verdict (Pass/Needs attention/Fail), "
            "issues (list of strings), and recommendations (list of strings). "
            "Do not invent data that isn't supported by the metrics."
        )
        user_prompt = (
            "Analyse the editing outcome described below. "
            "Focus on pacing, continuity, and alignment with expected segment durations. "
            "Respond with strict JSON.\n\n"
            f"{payload}"
        )
        if hasattr(self.judge, "judge_quality"):
            return self.judge.judge_quality(system_prompt, user_prompt)
        if hasattr(self.judge, "generate_text"):  # fallback to generic text generator
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self.judge.generate_text(messages)
        raise RuntimeError("Attached judge does not expose a compatible interface.")

    def _parse_model_judgement(
        self,
        response: str,
    ) -> Tuple[Optional[float], Optional[str], List[QualityIssue], List[str]]:
        cleaned = _clean_json_block(response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Quality judge returned invalid JSON. Raw response begins: %s", response[:200])
            return None, None, [], []

        score = None
        verdict = None
        issues: List[QualityIssue] = []
        recommendations: List[str] = []

        raw_score = data.get("score")
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        raw_verdict = data.get("verdict")
        if isinstance(raw_verdict, str):
            verdict = raw_verdict.strip()

        for message in data.get("issues", []) or []:
            if isinstance(message, str) and message.strip():
                issues.append(QualityIssue(message=message.strip()))
        for message in data.get("recommendations", []) or []:
            if isinstance(message, str) and message.strip():
                recommendations.append(message.strip())

        return score, verdict, issues, recommendations

    # ------------------------------------------------------------------ heuristics

    def _gather_metrics(
        self,
        instructions: Sequence[EditInstruction],
        segment_paths: Sequence[Path],
        final_output: Path,
    ) -> Tuple[List[dict[str, Any]], List[QualityIssue]]:
        issues: List[QualityIssue] = []
        metrics: List[dict[str, Any]] = []
        path_lookup = {path.name: path for path in segment_paths}

        for index, instruction in enumerate(instructions, start=1):
            expected_duration = max(0.0, instruction.in_end - instruction.in_start)
            rendered_name = f"{index:02d}_{(instruction.label or instruction.chunk).replace('.', '_')}.mp4"
            rendered_path = path_lookup.get(rendered_name)
            actual_duration = self._probe_duration(rendered_path) if rendered_path else 0.0
            delta = abs(actual_duration - expected_duration)

            if rendered_path is None:
                issues.append(QualityIssue(f"Segment {index} missing output file ({rendered_name}).", severity="error"))
            elif delta > 0.35:
                issues.append(
                    QualityIssue(
                        f"Segment {index} duration off by {delta:.2f}s (expected {expected_duration:.2f}s, actual {actual_duration:.2f}s).",
                        severity="warning",
                    )
                )

            metrics.append(
                {
                    "segment": index,
                    "label": instruction.label or f"Segment {index}",
                    "expected_duration": round(expected_duration, 3),
                    "actual_duration": round(actual_duration, 3),
                    "delta": round(delta, 3),
                    "output_file": rendered_name,
                    "missing": rendered_path is None,
                }
            )

        final_duration = self._probe_duration(final_output)
        metrics.append(
            {
                "segment": "final",
                "label": "Final Assembly",
                "expected_duration": round(sum(m["actual_duration"] for m in metrics if not isinstance(m["segment"], str)), 3),
                "actual_duration": round(final_duration, 3),
                "delta": round(
                    abs(final_duration - sum(m["actual_duration"] for m in metrics if not isinstance(m["segment"], str))),
                    3,
                ),
                "output_file": final_output.name,
                "missing": not final_output.exists(),
            }
        )
        if not final_output.exists():
            issues.append(QualityIssue("Final video is missing.", severity="error"))

        return metrics, issues

    def _probe_duration(self, path: Optional[Path]) -> float:
        if path is None or not path.exists():
            return 0.0
        cmd = [
            self.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            return float(result.stdout.strip())
        except Exception:
            logger.debug("Failed to probe duration for %s", path)
            return 0.0

    def _score_metrics(
        self,
        metrics: Sequence[dict[str, Any]],
        issues: Sequence[QualityIssue],
    ) -> Tuple[float, List[str]]:
        if not metrics:
            return 0.0, ["No segments available for review."]

        deltas = []
        recommendations: List[str] = []
        for metric in metrics:
            if metric.get("missing"):
                deltas.append(1.0)
            else:
                expected = metric.get("expected_duration", 0.0) or 0.0
                delta = metric.get("delta", 0.0) or 0.0
                normalized = min(1.0, delta / (expected + 1e-3))
                deltas.append(normalized)
                if expected > 0 and delta > 0.5:
                    recommendations.append(
                        f"Review pacing for {metric.get('label')} (delta {delta:.2f}s against expected {expected:.2f}s)."
                    )

        penalty = sum(deltas) / len(deltas)
        base_score = max(0.0, 1.0 - penalty)

        if any(issue.severity == "error" for issue in issues):
            base_score = min(base_score, 0.4)
        return base_score, recommendations


def _clean_json_block(text: str) -> str:
    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1]
    elif stripped.startswith("<think>"):
        stripped = stripped[len("<think>") :]
    stripped = stripped.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        return stripped[start : end + 1].strip()
    return stripped
