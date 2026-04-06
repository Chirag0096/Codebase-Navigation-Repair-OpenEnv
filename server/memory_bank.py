# server/memory_bank.py
"""
Episodic Memory Bank — v4.0

Cross-episode learning store for AI coding agents.

Every time an agent fails at a specific failure type, we store:
1. The failure pattern (what actions led to it)
2. The remediation hint (what should have been done)
3. A compact "lesson" that can be injected into future prompts

The memory grows across episodes. When a new episode starts:
- We retrieve the most relevant past lessons (by task similarity)
- We inject them as a "memory context" into the agent's system prompt
- This creates a real self-improvement loop

This is NOT implemented in any current agent framework as an
environment-side primitive. Devin, Copilot, etc. start fresh every run.
"""
from __future__ import annotations
import json
import time
import os
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class MemoryEntry:
    """One stored episode lesson."""
    entry_id: str
    episode_id: str
    task: str
    created_at: float

    # Failure details
    failure_type: str
    failure_evidence: str
    score: float

    # Strategy used
    strategy: str
    action_sequence_hash: str  # Compact fingerprint of the action pattern

    # Lesson extracted
    lesson_title: str
    lesson_body: str      # Full explanation of what went wrong
    lesson_hint: str      # Compact hint to inject into future prompts
    lesson_plan: List[str]  # Step-by-step corrective plan

    # Retrieval metadata
    relevance_tags: List[str]    # Tags for retrieval (task1, write_file, read_before_write...)
    times_retrieved: int = 0
    times_helpful: int = 0       # Incremented when retry after this lesson improved score

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**d)


@dataclass
class MemoryContext:
    """Injected memory context for a new episode."""
    relevant_lessons: List[MemoryEntry]
    system_prompt_injection: str   # Full text to prepend to system prompt
    user_context_injection: str    # Full text to prepend to first user message
    lessons_count: int
    most_relevant_lesson: Optional[str]


class EpisodicMemoryBank:
    """
    Persistent cross-episode memory bank.

    Storage: JSON file on disk (or in-memory for Gradio sessions).
    Each entry is a MemoryEntry with lesson + retrieval metadata.

    Usage:
        bank = EpisodicMemoryBank(persist_path="memory.json")
        # After an episode:
        bank.store(episode_result)
        # Before next episode:
        context = bank.retrieve(task="task1", max_lessons=3)
        # Inject context.system_prompt_injection into agent
    """

    MAX_ENTRIES = 50  # Keep last 50 lessons per task

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._entries: List[MemoryEntry] = []
        if persist_path and os.path.exists(persist_path):
            self._load()

    def store(
        self,
        episode_id: str,
        task: str,
        failure_type: str,
        failure_evidence: str,
        score: float,
        strategy: str,
        trajectory_steps: List[dict],
        improvement_plan: Optional[dict] = None,
    ) -> MemoryEntry:
        """Store a lesson from a completed episode."""
        # Build action fingerprint
        actions = [s.get("action_type", "?") for s in trajectory_steps]
        seq_str = "→".join(actions[:12])
        seq_hash = hashlib.md5(seq_str.encode()).hexdigest()[:8]

        # Relevance tags for retrieval
        tags = [task, failure_type, strategy]
        if "read_file" in actions:
            tags.append("read_file")
        if "write_file" in actions:
            tags.append("write_file")
        if "run_tests" not in actions:
            tags.append("no_verification")
        if len(actions) <= 3:
            tags.append("too_short")

        # Extract lesson from improvement plan or failure type
        if improvement_plan:
            lesson_title = improvement_plan.get("failure_type", failure_type)
            lesson_body = improvement_plan.get("what_went_wrong", "Agent failed.")
            lesson_hint = improvement_plan.get("system_prompt_addon", "")
            lesson_plan = improvement_plan.get("step_by_step_plan", [])
        else:
            lesson_title, lesson_body, lesson_hint, lesson_plan = self._default_lesson(
                failure_type, score, strategy
            )

        entry = MemoryEntry(
            entry_id=f"{task}_{seq_hash}_{int(time.time())}",
            episode_id=episode_id,
            task=task,
            created_at=time.time(),
            failure_type=failure_type,
            failure_evidence=failure_evidence[:200],
            score=score,
            strategy=strategy,
            action_sequence_hash=seq_hash,
            lesson_title=lesson_title,
            lesson_body=lesson_body,
            lesson_hint=lesson_hint,
            lesson_plan=lesson_plan,
            relevance_tags=tags,
            times_retrieved=0,
            times_helpful=0,
        )

        self._entries.append(entry)
        self._trim()
        if self.persist_path:
            self._save()
        return entry

    def retrieve(
        self,
        task: str,
        failure_type: Optional[str] = None,
        strategy: Optional[str] = None,
        max_lessons: int = 3,
    ) -> MemoryContext:
        """Retrieve relevant lessons for an upcoming episode."""
        if not self._entries:
            return MemoryContext(
                relevant_lessons=[],
                system_prompt_injection="",
                user_context_injection="",
                lessons_count=0,
                most_relevant_lesson=None,
            )

        # Score each entry by relevance
        scored: List[tuple[float, MemoryEntry]] = []
        for e in self._entries:
            score = 0.0
            if e.task == task:
                score += 3.0
            elif task in e.relevance_tags:
                score += 2.0
            if failure_type and e.failure_type == failure_type:
                score += 2.0
            if strategy and e.strategy == strategy:
                score += 1.0
            # Penalize already-retrieved lessons slightly (freshness)
            score -= e.times_retrieved * 0.1
            # Boost low-score lessons (more informative failures)
            score += max(0, 0.5 - e.score)
            scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        relevant = [e for _, e in scored[:max_lessons]]

        # Mark as retrieved
        for e in relevant:
            e.times_retrieved += 1

        if not relevant:
            return MemoryContext(
                relevant_lessons=[],
                system_prompt_injection="",
                user_context_injection="",
                lessons_count=0,
                most_relevant_lesson=None,
            )

        # Build injection text
        sys_lines = [
            "🧠 AGENT MEMORY — LESSONS FROM PAST EPISODES",
            "=" * 50,
            "You have made these mistakes before. Do NOT repeat them.",
            "",
        ]
        for i, e in enumerate(relevant, 1):
            sys_lines += [
                f"[Lesson {i}] Task: {e.task} | Failure: {e.failure_type} | Score was: {e.score:.2f}",
                f"What went wrong: {e.lesson_body}",
                f"IMPORTANT: {e.lesson_hint}" if e.lesson_hint else "",
                "",
            ]
        sys_lines.append("=" * 50)
        system_injection = "\n".join(l for l in sys_lines if l is not None)

        user_lines = [
            "[MEMORY CONTEXT — Read before you act]",
        ]
        for i, e in enumerate(relevant, 1):
            user_lines.append(f"Past lesson {i}: {e.lesson_title}")
            if e.lesson_plan:
                user_lines.append("Correct approach:")
                user_lines.extend(f"  {step}" for step in e.lesson_plan[:4])
        user_injection = "\n".join(user_lines)

        return MemoryContext(
            relevant_lessons=relevant,
            system_prompt_injection=system_injection,
            user_context_injection=user_injection,
            lessons_count=len(relevant),
            most_relevant_lesson=relevant[0].lesson_title if relevant else None,
        )

    def get_all_entries(self) -> List[dict]:
        return [e.to_dict() for e in self._entries]

    def get_stats(self) -> dict:
        if not self._entries:
            return {"total_entries": 0, "tasks": {}}

        from collections import Counter
        failure_counts = Counter(e.failure_type for e in self._entries)
        task_counts = Counter(e.task for e in self._entries)
        avg_score = sum(e.score for e in self._entries) / len(self._entries)

        return {
            "total_entries": len(self._entries),
            "average_score_of_stored_episodes": round(avg_score, 3),
            "failure_breakdown": dict(failure_counts.most_common()),
            "tasks": dict(task_counts),
            "most_helpful_lesson": max(self._entries, key=lambda e: e.times_helpful).lesson_title
                if any(e.times_helpful > 0 for e in self._entries) else None,
        }

    def mark_helpful(self, episode_id: str):
        """Call this when a retry with a lesson improved the score."""
        for e in self._entries:
            if e.episode_id == episode_id:
                e.times_helpful += 1
        if self.persist_path:
            self._save()

    def clear(self, task: Optional[str] = None):
        if task:
            self._entries = [e for e in self._entries if e.task != task]
        else:
            self._entries = []
        if self.persist_path:
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        with open(self.persist_path, "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self._entries = [MemoryEntry.from_dict(d) for d in data]
        except Exception:
            self._entries = []

    def _trim(self):
        """Keep at most MAX_ENTRIES, dropping oldest low-score entries first."""
        if len(self._entries) <= self.MAX_ENTRIES:
            return
        # Sort by: useful first, then by recency
        self._entries.sort(
            key=lambda e: (
                -e.times_helpful,
                -e.times_retrieved,
                e.created_at,
            ),
            reverse=True,
        )
        self._entries = self._entries[:self.MAX_ENTRIES]

    def _default_lesson(
        self, failure_type: str, score: float, strategy: str
    ) -> tuple[str, str, str, List[str]]:
        lessons = {
            "NEVER_TESTED": (
                "Submitted without verification",
                "Agent submitted code without running tests. No confidence in correctness.",
                "CRITICAL: Run run_tests after EVERY write_file. Never submit without test verification.",
                ["1. Write fix", "2. run_tests to check", "3. If passing → submit", "4. If failing → re-read and fix"],
            ),
            "BLIND_WRITE": (
                "Wrote without reading",
                "Agent wrote to a file without reading it first. Blind writes introduce new bugs.",
                "NEVER use write_file before read_file on the same path.",
                ["1. read_file first", "2. Understand existing code", "3. Then write minimal fix"],
            ),
            "WRONG_FILE_NAVIGATION": (
                "Navigated to wrong files",
                "Agent read files unrelated to the bug. Wasted steps and missed root cause.",
                "ALWAYS start with the failing test file. Its imports show you exactly where to go.",
                ["1. Read failing test", "2. Find its imports", "3. Navigate ONLY there"],
            ),
            "LOOPING_BEHAVIOR": (
                "Read same files repeatedly",
                f"Agent looped reading the same files without progress. Score={score:.2f}.",
                "Each file may be read AT MOST ONCE. Use search_code if confused.",
                ["1. Use search_code with function name", "2. Read matched file — once", "3. commit to fix"],
            ),
        }
        defaults = lessons.get(failure_type, (
            f"{failure_type} failure",
            f"Agent failed with type '{failure_type}', score={score:.2f}.",
            "Read test → read source → fix → run_tests → submit.",
            ["1. read test", "2. read source", "3. write fix", "4. run_tests", "5. submit"],
        ))
        return defaults


# Global singleton (shared across the Gradio session)
_GLOBAL_MEMORY = EpisodicMemoryBank(
    persist_path=os.path.join(
        os.path.dirname(__file__), "..", "agent_memory.json"
    )
)


def get_global_memory() -> EpisodicMemoryBank:
    return _GLOBAL_MEMORY
