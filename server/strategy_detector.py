# server/strategy_detector.py
"""
Strategy Pattern Detector.

Classifies what high-level search/navigation strategy the agent used.
This goes beyond step counting — it classifies the cognitive approach.

Strategies:
  TARGETED_DEBUGGING   — reads test → reads relevant src → fixes → tests
  SYSTEMATIC_SEARCH    — reads all files methodically before writing
  BRUTE_FORCE          — writes and runs tests repeatedly until something passes
  RANDOM_EXPLORATION   — no coherent pattern, reads random files
  SPEC_DRIVEN          — reads spec/docs first, then implements
  MINIMAL_EFFORT       — does the bare minimum (often fails)

Each strategy gets a score (1.0 = ideal for the task), a label, and evidence.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class StrategyReport:
    """Result of strategy pattern detection."""
    strategy: str              # Primary strategy label
    score: float               # 0.0–1.0 (task-appropriate quality)
    confidence: float          # How confident we are in the label (0–1)
    sub_patterns: List[str]    # Additional behavioral sub-patterns
    evidence: List[str]        # Supporting observations
    strategy_description: str  # Human-readable explanation
    exploration_ratio: float   # 0=pure exploit, 1=pure explore
    pivot_count: int           # How many times agent changed strategy mid-episode

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "sub_patterns": self.sub_patterns,
            "evidence": self.evidence,
            "strategy_description": self.strategy_description,
            "exploration_ratio": round(self.exploration_ratio, 3),
            "pivot_count": self.pivot_count,
        }


STRATEGY_DESCRIPTIONS = {
    "TARGETED_DEBUGGING": (
        "Agent reads the failing test to understand expected behavior, "
        "then navigates directly to the relevant source file and makes a targeted fix."
    ),
    "SYSTEMATIC_SEARCH": (
        "Agent reads all available files before writing any code. "
        "Methodical but can waste steps on irrelevant files."
    ),
    "BRUTE_FORCE": (
        "Agent repeatedly writes and runs tests hoping something sticks. "
        "No clear hypothesis about the bug — trial and error approach."
    ),
    "RANDOM_EXPLORATION": (
        "Agent reads files in an incoherent order with no visible strategy. "
        "High entropy — possibly confused by misleading information."
    ),
    "SPEC_DRIVEN": (
        "Agent reads the specification/feature doc first, "
        "then systematically implements what is described. Ideal for task3."
    ),
    "MINIMAL_EFFORT": (
        "Agent took very few steps and submitted early. "
        "May indicate overconfidence or giving up."
    ),
}


class StrategyDetector:
    """
    Detects the behavioral strategy pattern used by an agent.

    Usage:
        detector = StrategyDetector()
        report = detector.detect(
            trajectory_steps=[...],
            task="task1",
            variant_meta={...},
            files_read=[...],
            final_score=0.7,
        )
    """

    def detect(
        self,
        trajectory_steps: List[dict],
        task: str,
        variant_meta: Dict[str, Any],
        files_read: List[str],
        final_score: float,
    ) -> StrategyReport:
        """Detect strategy from trajectory data."""
        if not trajectory_steps:
            return StrategyReport(
                strategy="MINIMAL_EFFORT",
                score=0.0,
                confidence=1.0,
                sub_patterns=[],
                evidence=["No steps taken"],
                strategy_description=STRATEGY_DESCRIPTIONS["MINIMAL_EFFORT"],
                exploration_ratio=0.0,
                pivot_count=0,
            )

        action_seq = [s.get("action_type", "") for s in trajectory_steps]
        read_paths = [
            s.get("action_path", "")
            for s in trajectory_steps
            if s.get("action_type") == "read_file"
        ]
        write_count = action_seq.count("write_file")
        test_count = action_seq.count("run_tests")
        read_count = action_seq.count("read_file")
        search_count = action_seq.count("search_code")
        total = len(action_seq)

        relevant = set(
            variant_meta.get("bug_files", []) +
            variant_meta.get("interface_files", []) +
            variant_meta.get("read_first_files", [])
        )
        test_files = [f for f in read_paths if f and f.startswith("tests/")]
        spec_files = [f for f in read_paths if f and f.endswith(".md")]

        sub_patterns = []
        evidence = []

        # ── Exploration ratio: reads/searches vs writes/tests ─────────────────
        explore_actions = read_count + search_count
        exploit_actions = write_count + test_count
        exploration_ratio = (
            explore_actions / (explore_actions + exploit_actions)
            if (explore_actions + exploit_actions) > 0
            else 0.5
        )

        # ── Pivot detection: strategy changes mid-episode ─────────────────────
        pivots = 0
        blocks = []
        current_block = action_seq[0] if action_seq else None
        for a in action_seq[1:]:
            read_like = a in ("read_file", "search_code")
            write_like = a in ("write_file", "run_tests")
            cur_read = current_block in ("read_file", "search_code")
            cur_write = current_block in ("write_file", "run_tests")
            if (read_like and cur_write) or (write_like and cur_read):
                pivots += 1
            current_block = a

        # ── Strategy classification ────────────────────────────────────────────
        strategy = "RANDOM_EXPLORATION"
        score = 0.4
        confidence = 0.5

        # 1. SPEC_DRIVEN (reads spec/md first, task3)
        if task == "task3" and spec_files and action_seq.index("read_file") == 0:
            strategy = "SPEC_DRIVEN"
            score = 0.85 if final_score > 0.5 else 0.55
            confidence = 0.9
            evidence.append(f"Read spec file(s) first: {spec_files[:2]}")
            sub_patterns.append("SPEC_FIRST")

        # 2. TARGETED_DEBUGGING (test first → relevant src → write)
        elif (test_files and read_paths and read_paths[0].startswith("tests/")
              and write_count >= 1 and test_count >= 1):
            strategy = "TARGETED_DEBUGGING"
            score = 0.85 + (0.15 * final_score)
            confidence = 0.85
            evidence.append(f"First read was test file: {read_paths[0]}")
            evidence.append(f"Followed by write + test verification")
            sub_patterns.append("TEST_FIRST")
            if relevant and any(f in files_read for f in relevant):
                sub_patterns.append("TARGETED_READ")
                score = min(1.0, score + 0.05)

        # 3. SYSTEMATIC_SEARCH (all files read before any write)
        elif write_count > 0:
            first_write_idx = next((i for i, a in enumerate(action_seq) if a == "write_file"), total)
            reads_before_write = sum(1 for i, a in enumerate(action_seq) if a == "read_file" and i < first_write_idx)
            if read_count > 0 and reads_before_write == read_count:
                strategy = "SYSTEMATIC_SEARCH"
                score = 0.65
                confidence = 0.75
                evidence.append(f"Read {reads_before_write} files before first write")
                sub_patterns.append("READ_ALL_FIRST")

            # 4. BRUTE_FORCE (multiple write-test cycles)
            elif write_count >= 2 and test_count >= 2:
                strategy = "BRUTE_FORCE"
                score = 0.35
                confidence = 0.8
                evidence.append(f"{write_count} writes + {test_count} test runs = trial and error")
                sub_patterns.append("TRIAL_AND_ERROR")

        # 5. MINIMAL_EFFORT (tiny episode, or only submit)
        elif total <= 3 or (write_count == 0 and test_count == 0):
            strategy = "MINIMAL_EFFORT"
            score = 0.1
            confidence = 0.95
            evidence.append(f"Only {total} total steps with no fix attempt")
            sub_patterns.append("GAVE_UP")

        # ── Additional sub-pattern detection ──────────────────────────────────
        # Search-before-read
        if search_count > 0:
            first_search = next((i for i, a in enumerate(action_seq) if a == "search_code"), total)
            first_read = next((i for i, a in enumerate(action_seq) if a == "read_file"), total)
            if first_search < first_read:
                sub_patterns.append("SEARCH_GUIDED")
                evidence.append("Used search_code to locate bug before reading")

        # Excessive looping
        path_counts = Counter(p for p in read_paths if p)
        max_rereads = max(path_counts.values()) if path_counts else 0
        if max_rereads >= 3:
            sub_patterns.append("READ_LOOP")
            evidence.append(f"Re-read same file {max_rereads}x — likely confused")
            score = max(0.0, score - 0.2)

        # Verified fix (ran tests and found improvement)
        test_rates = [s.get("test_pass_rate") for s in trajectory_steps if s.get("test_pass_rate") is not None]
        if len(test_rates) >= 2 and test_rates[-1] > test_rates[0]:
            sub_patterns.append("VERIFIED_FIX")
            evidence.append(f"Test pass rate improved: {test_rates[0]:.2f} → {test_rates[-1]:.2f}")
            score = min(1.0, score + 0.1)

        return StrategyReport(
            strategy=strategy,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            sub_patterns=sub_patterns,
            evidence=evidence,
            strategy_description=STRATEGY_DESCRIPTIONS.get(strategy, ""),
            exploration_ratio=exploration_ratio,
            pivot_count=pivots,
        )
