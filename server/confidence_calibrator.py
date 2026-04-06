# server/confidence_calibrator.py
"""
Confidence Calibration Engine — v4.0

The key scientific question: Is the agent calibrated?
An agent is calibrated when its certainty level (inferred from behavior)
matches its likelihood of being correct.

Since agents don't expose probability distributions directly, we infer
confidence from behavioral proxies:
- How quickly did it commit to a hypothesis (read → write speed)?
- How much did it re-explore after writing (re-reads after write)?
- Did it verify (run_tests) before submitting?
- How many steps did it spend before the first write?

We then compare inferred confidence to actual accuracy (final_score).
Overconfident agents submit fast but score poorly.
Underconfident agents explore extensively but still score well.
Well-calibrated agents: confidence ∝ accuracy.

This is NOT measured by any existing benchmark or tracing tool.
"""
from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class CalibrationProfile(str, Enum):
    WELL_CALIBRATED = "WELL_CALIBRATED"    # Confidence ≈ accuracy
    OVERCONFIDENT = "OVERCONFIDENT"        # High confidence, low accuracy
    UNDERCONFIDENT = "UNDERCONFIDENT"      # Low confidence, high accuracy
    ERRATIC = "ERRATIC"                   # Confidence changes randomly


@dataclass
class ConfidenceSample:
    """Inferred confidence at one point in the trajectory."""
    step: int
    action_type: str
    inferred_confidence: float   # 0.0–1.0 based on behavioral proxy
    actual_accuracy: Optional[float]  # test_pass_rate at this step if known
    calibration_error: Optional[float]  # |confidence - accuracy| if both known


@dataclass
class CalibrationReport:
    """Full confidence calibration analysis."""
    episode_id: str
    task: str

    profile: CalibrationProfile
    calibration_score: float      # 1.0 = perfectly calibrated

    # Inferred overall confidence level (behavioral proxy)
    inferred_confidence: float    # 0.0–1.0
    actual_performance: float     # final_score

    # Decomposed signals
    commitment_speed: float      # How fast did agent commit? (0=slow/careful, 1=fast)
    re_exploration_rate: float   # Reads after first write / total reads
    verification_rate: float     # run_tests per write_file
    submit_speed: float          # Submit step / max_steps (early=overconfident)

    # Trajectory of inferred confidence
    confidence_trajectory: List[ConfidenceSample]

    # Calibration error
    expected_calibration_error: float  # Mean(|conf - acc|) where acc is known
    confidence_accuracy_correlation: float  # Should be high for good agents

    diagnosis: str
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "profile": self.profile.value,
            "calibration_score": round(self.calibration_score, 3),
            "inferred_confidence": round(self.inferred_confidence, 3),
            "actual_performance": round(self.actual_performance, 3),
            "signals": {
                "commitment_speed": round(self.commitment_speed, 3),
                "re_exploration_rate": round(self.re_exploration_rate, 3),
                "verification_rate": round(self.verification_rate, 3),
                "submit_speed": round(self.submit_speed, 3),
            },
            "expected_calibration_error": round(self.expected_calibration_error, 3),
            "confidence_accuracy_correlation": round(self.confidence_accuracy_correlation, 3),
            "confidence_trajectory": [
                {
                    "step": s.step,
                    "action": s.action_type,
                    "confidence": round(s.inferred_confidence, 3),
                    "accuracy": round(s.actual_accuracy, 3) if s.actual_accuracy is not None else None,
                    "error": round(s.calibration_error, 3) if s.calibration_error is not None else None,
                }
                for s in self.confidence_trajectory
            ],
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
        }


class ConfidenceCalibrator:
    """
    Infers behavioral confidence and compares to actual performance.

    Confidence proxy model:
    - Reading files = low confidence (still exploring)
    - Writing files = medium-high confidence (committed to hypothesis)
    - Running tests = verification (moderate, checking own hypothesis)
    - Submitting = maximum commitment (fully confident)

    Each action type has a confidence weight:
      read_file:   0.2  (exploring, uncertain)
      search_code: 0.3  (slightly more directed)
      run_tests:   0.6  (confident enough to test)
      write_file:  0.75 (committed to hypothesis)
      submit:      1.0  (maximum confidence)

    We track how this evolves over the trajectory.
    """

    ACTION_CONFIDENCE = {
        "read_file":   0.2,
        "search_code": 0.3,
        "run_tests":   0.6,
        "write_file":  0.75,
        "submit":      1.0,
    }

    def calibrate(
        self,
        episode_id: str,
        task: str,
        trajectory_steps: List[dict],
        final_score: float,
        max_steps: int = 20,
    ) -> CalibrationReport:
        """Compute the full calibration report for one episode."""

        if not trajectory_steps:
            return self._empty_report(episode_id, task, final_score)

        action_types = [s.get("action_type", "read_file") for s in trajectory_steps]
        total_steps = len(trajectory_steps)

        # ── Build confidence trajectory ───────────────────────────────────────
        confidence_traj: List[ConfidenceSample] = []
        running_conf = 0.0

        for s in trajectory_steps:
            atype = s.get("action_type", "read_file")
            base_conf = self.ACTION_CONFIDENCE.get(atype, 0.3)

            # Confidence grows as episode progresses
            step_n = s.get("step_number", 1)
            progress_bonus = (step_n / max(total_steps, 1)) * 0.1

            # Re-reads slightly lower confidence
            step_write_count = sum(
                1 for s2 in trajectory_steps
                if s2.get("action_type") == "write_file"
                and s2.get("step_number", 99) < step_n
            )
            step_reread = (
                s.get("action_type") == "read_file"
                and any(
                    s2.get("action_path") == s.get("action_path")
                    and s2.get("step_number", 0) < step_n
                    for s2 in trajectory_steps
                )
            )
            reread_penalty = -0.1 if step_reread else 0.0

            # After a write, confidence should be higher
            post_write_bonus = min(0.15, step_write_count * 0.05)

            inferred = min(1.0, max(0.0,
                base_conf + progress_bonus + post_write_bonus + reread_penalty
            ))

            # Actual accuracy at this step if test_pass_rate is known
            actual_acc = s.get("test_pass_rate")
            calib_err = abs(inferred - actual_acc) if actual_acc is not None else None

            confidence_traj.append(ConfidenceSample(
                step=step_n,
                action_type=atype,
                inferred_confidence=inferred,
                actual_accuracy=actual_acc,
                calibration_error=calib_err,
            ))

        # ── Behavioral signal computation ─────────────────────────────────────
        total = max(total_steps, 1)

        # Commitment speed: how many reads before first write?
        read_steps = [i for i, a in enumerate(action_types) if a == "read_file"]
        write_steps = [i for i, a in enumerate(action_types) if a == "write_file"]
        submit_step = next(
            (s.get("step_number", total) for s in trajectory_steps if s.get("action_type") == "submit"),
            total,
        )

        if write_steps:
            reads_before_first_write = len([r for r in read_steps if r < write_steps[0]])
            # Low reads before write = high commitment speed = overconfident
            commitment_speed = max(0.0, 1.0 - reads_before_first_write / max(total, 1))
        else:
            commitment_speed = 0.0  # Never wrote = very cautious

        # Re-exploration rate: reads after first write / total reads
        if write_steps and read_steps:
            reads_after_write = len([r for r in read_steps if r > write_steps[0]])
            re_exploration_rate = reads_after_write / len(read_steps)
        else:
            re_exploration_rate = 0.0

        # Verification rate: run_tests per write
        test_count = action_types.count("run_tests")
        write_count = action_types.count("write_file")
        verification_rate = test_count / max(write_count, 1)

        # Submit speed: earlier = more overconfident
        submit_speed = 1.0 - (submit_step / max(max_steps, 1))
        submit_speed = max(0.0, min(1.0, submit_speed))

        # ── Inferred overall confidence ───────────────────────────────────────
        # Weighted behavioral proxy
        inferred_confidence = (
            commitment_speed * 0.30 +
            (1.0 - re_exploration_rate) * 0.15 +
            verification_rate * 0.15 +
            submit_speed * 0.20 +
            (confidence_traj[-1].inferred_confidence if confidence_traj else 0.5) * 0.20
        )
        inferred_confidence = min(1.0, max(0.0, inferred_confidence))

        # ── Calibration error (where we have both conf + acc) ─────────────────
        calib_errors = [
            s.calibration_error for s in confidence_traj
            if s.calibration_error is not None
        ]
        ece = sum(calib_errors) / len(calib_errors) if calib_errors else abs(inferred_confidence - final_score)

        # ── Confidence-accuracy correlation ────────────────────────────────────
        paired = [
            (s.inferred_confidence, s.actual_accuracy)
            for s in confidence_traj
            if s.actual_accuracy is not None
        ]
        if len(paired) >= 2:
            corr = self._pearson_r([p[0] for p in paired], [p[1] for p in paired])
        else:
            # Fallback: use final point only
            conf_err = abs(inferred_confidence - final_score)
            corr = 1.0 - conf_err * 2

        corr = max(-1.0, min(1.0, corr))

        # ── Calibration score ─────────────────────────────────────────────────
        calibration_score = max(0.0, 1.0 - ece) * 0.5 + max(0.0, corr) * 0.5
        calibration_score = max(0.0, min(1.0, calibration_score))

        # ── Profile classification ─────────────────────────────────────────────
        conf_diff = inferred_confidence - final_score
        if abs(conf_diff) <= 0.2:
            profile = CalibrationProfile.WELL_CALIBRATED
        elif conf_diff > 0.2:
            profile = CalibrationProfile.OVERCONFIDENT
        elif conf_diff < -0.2:
            profile = CalibrationProfile.UNDERCONFIDENT
        else:
            profile = CalibrationProfile.ERRATIC

        # ── Diagnosis ─────────────────────────────────────────────────────────
        diagnoses = {
            CalibrationProfile.WELL_CALIBRATED: (
                f"Agent is well-calibrated: inferred confidence ({inferred_confidence:.2f}) "
                f"closely matches actual performance ({final_score:.2f}). "
                "This indicates genuine self-awareness — the agent commits when ready and "
                "explores when uncertain."
            ),
            CalibrationProfile.OVERCONFIDENT: (
                f"Agent is overconfident: behavioral confidence ({inferred_confidence:.2f}) "
                f"significantly exceeds actual performance ({final_score:.2f}). "
                "Agent committed to a hypothesis too early, skipped verification, "
                "or submitted without adequate exploration. This is the profile of agents "
                "that 'feel certain but are wrong'."
            ),
            CalibrationProfile.UNDERCONFIDENT: (
                f"Agent is underconfident: behavioral confidence ({inferred_confidence:.2f}) "
                f"is well below actual performance ({final_score:.2f}). "
                "Agent explored far more than necessary, re-read files unnecessarily, "
                "or hesitated to commit despite having the right information. "
                "This wastes compute and steps without improving accuracy."
            ),
            CalibrationProfile.ERRATIC: (
                "Agent calibration is erratic — confidence signals are inconsistent "
                "with behavior. The agent may be applying a rigid strategy regardless "
                "of the task difficulty."
            ),
        }

        recs = []
        if profile == CalibrationProfile.OVERCONFIDENT:
            recs.append("Read more files before writing — commit only when you've seen the full causal chain.")
            recs.append("Always run_tests after writing — don't trust your fix without verification.")
        elif profile == CalibrationProfile.UNDERCONFIDENT:
            recs.append("Commit to hypotheses earlier — excessive re-reading wastes steps.")
            recs.append("After reading tests + source files, write your fix. Stop re-reading.")
        if verification_rate < 0.5:
            recs.append("Increase test verification rate: run_tests after each write.")
        if re_exploration_rate > 0.5:
            recs.append("High re-exploration after writing suggests uncalibrated hypothesis formation.")

        return CalibrationReport(
            episode_id=episode_id,
            task=task,
            profile=profile,
            calibration_score=calibration_score,
            inferred_confidence=inferred_confidence,
            actual_performance=final_score,
            commitment_speed=commitment_speed,
            re_exploration_rate=re_exploration_rate,
            verification_rate=verification_rate,
            submit_speed=submit_speed,
            confidence_trajectory=confidence_traj,
            expected_calibration_error=ece,
            confidence_accuracy_correlation=corr,
            diagnosis=diagnoses[profile],
            recommendations=recs,
        )

    def _pearson_r(self, xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx * dy == 0:
            return 0.0
        return num / (dx * dy)

    def _empty_report(self, episode_id: str, task: str, final_score: float) -> CalibrationReport:
        return CalibrationReport(
            episode_id=episode_id, task=task,
            profile=CalibrationProfile.ERRATIC,
            calibration_score=0.0,
            inferred_confidence=0.0, actual_performance=final_score,
            commitment_speed=0.0, re_exploration_rate=0.0,
            verification_rate=0.0, submit_speed=0.0,
            confidence_trajectory=[],
            expected_calibration_error=1.0,
            confidence_accuracy_correlation=0.0,
            diagnosis="No trajectory data.", recommendations=[],
        )
