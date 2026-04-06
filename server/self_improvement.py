# server/self_improvement.py
"""
Self-Improvement Loop.

After a failure, generates structured feedback and an improved strategy prompt
that can be injected into the agent's next attempt. This closes the loop
between evaluation and agent behavior.

The retry loop:
  1. Run episode → evaluate → classify failures
  2. Generate improvement prompt based on failure type
  3. Re-run episode with improvement prompt injected into agent context
  4. Compare before/after performance
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ImprovementPlan:
    """Structured feedback for the agent's next attempt."""
    episode_id: str
    task: str
    failure_type: str
    original_score: float

    # Actionable feedback
    what_went_wrong: str
    specific_errors: List[str]
    improved_strategy: str
    step_by_step_plan: List[str]

    # For injection into agent prompt
    system_prompt_addon: str    # Extra instructions for the system prompt
    user_context_addon: str     # Extra context for the user prompt

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "failure_type": self.failure_type,
            "original_score": round(self.original_score, 3),
            "what_went_wrong": self.what_went_wrong,
            "specific_errors": self.specific_errors,
            "improved_strategy": self.improved_strategy,
            "step_by_step_plan": self.step_by_step_plan,
            "system_prompt_addon": self.system_prompt_addon,
            "user_context_addon": self.user_context_addon,
        }


@dataclass
class RetryResult:
    """Result of a retry attempt with improvement feedback."""
    original_episode_id: str
    retry_episode_id: str
    original_score: float
    retry_score: float
    improvement: float     # retry_score - original_score
    failure_fixed: bool
    steps_comparison: Dict[str, int]  # {"original": N, "retry": M}

    def to_dict(self) -> dict:
        return {
            "original_episode_id": self.original_episode_id,
            "retry_episode_id": self.retry_episode_id,
            "original_score": round(self.original_score, 3),
            "retry_score": round(self.retry_score, 3),
            "improvement": round(self.improvement, 3),
            "failure_fixed": self.failure_fixed,
            "steps_comparison": self.steps_comparison,
        }


# ── Strategy templates per failure type ──────────────────────────────────────
STRATEGY_TEMPLATES = {
    "WRONG_FILE_NAVIGATION": {
        "what_went_wrong": "Agent navigated to the wrong files and missed the bug location entirely.",
        "strategy": "START with the failing test file. Read its imports. Navigate exclusively to those imported modules.",
        "plan": [
            "1. Read the failing test file FIRST (in tests/ directory)",
            "2. Find the import statements — these point to the buggy module",
            "3. Read ONLY those imported source files",
            "4. Look for the function/method the test is calling",
            "5. Fix the specific function — do not touch other code",
            "6. Run the failing test to verify",
            "7. Submit",
        ],
        "system_addon": (
            "CRITICAL: You previously failed by reading the wrong files. "
            "This time: read the failing test first, identify its imports, "
            "go directly to those source files. Do NOT read any file not referenced by the test."
        ),
    },
    "BLIND_WRITE": {
        "what_went_wrong": "Agent wrote code without reading the existing implementation first.",
        "strategy": "NEVER write before reading. Read the target file. Understand the existing logic. Then write a minimal fix.",
        "plan": [
            "1. Read the failing test to understand expected behavior",
            "2. Read the source file you plan to modify",
            "3. Identify the exact line(s) causing failure",
            "4. Write a FIX (not a rewrite) targeting only those lines",
            "5. Run tests to verify improvement",
            "6. Submit",
        ],
        "system_addon": (
            "CRITICAL: You previously wrote code without reading the file first. "
            "This time: ALWAYS call read_file on any file BEFORE using write_file. "
            "No exceptions. Read → Understand → Write minimal fix."
        ),
    },
    "HALLUCINATED_CODE": {
        "what_went_wrong": "Agent wrote syntactically correct but logically wrong code that made tests worse.",
        "strategy": "Write a targeted, minimal fix. Do not rewrite entire functions. Change only what the test requires.",
        "plan": [
            "1. Read the failing test and note EXACTLY what assertion fails",
            "2. Read the source function — understand its current behavior",
            "3. Identify the gap between current and expected behavior",
            "4. Write the SMALLEST possible change that bridges that gap",
            "5. Run tests BEFORE submitting to verify the fix works",
            "6. If tests still fail, re-read and refine — don't guess",
        ],
        "system_addon": (
            "CRITICAL: Your previous fix made things worse. This indicates hallucination. "
            "This time: make the SMALLEST possible change. "
            "Run run_tests after EVERY write to check if you're improving or degrading. "
            "If tests get worse after a write, immediately read the file again and try a different approach."
        ),
    },
    "NEVER_TESTED": {
        "what_went_wrong": "Agent submitted code changes without running any tests to verify they work.",
        "strategy": "ALWAYS run run_tests after every write_file. Never submit without test verification.",
        "plan": [
            "1. Read test → Read source → Write fix",
            "2. IMMEDIATELY run run_tests pointing to the failing test file",
            "3. If tests pass: submit",
            "4. If tests still fail: re-read, refine, run tests again",
            "5. ONLY submit when you have seen test improvement",
        ],
        "system_addon": (
            "CRITICAL: You submitted without testing. This is invalid. "
            "This time: after EVERY write_file action, you MUST call run_tests. "
            "Only call submit when run_tests shows improvement. "
            "The pattern is: read → write → run_tests → submit. Non-negotiable."
        ),
    },
    "LOOPING_BEHAVIOR": {
        "what_went_wrong": "Agent got stuck reading the same file repeatedly without making progress.",
        "strategy": "Use search_code to find the exact bug location. Read each file at most once.",
        "plan": [
            "1. Use search_code with the function name from the failing test",
            "2. Read the file that contains the matching code — ONCE",
            "3. If you need more context, use search_code again with a different query",
            "4. Once you have read a file, do NOT read it again",
            "5. Write your fix, run tests, submit",
        ],
        "system_addon": (
            "CRITICAL: You read the same files 3+ times without progress. "
            "This time: you may read each file AT MOST ONCE. "
            "Use search_code to pinpoint bug location. "
            "If you are confused, use search_code — do not re-read files."
        ),
    },
    "SECURITY_VIOLATION": {
        "what_went_wrong": "Agent wrote dangerous code patterns that would be harmful in production.",
        "strategy": "Write pure Python logic only. Never use os, subprocess, eval, or exec.",
        "plan": [
            "1. Read the test to understand what pure Python behavior is needed",
            "2. Implement the fix using ONLY standard library functions",
            "3. No os.system(), subprocess, eval(), exec(), or __import__()",
            "4. Run tests and submit",
        ],
        "system_addon": (
            "CRITICAL: Your previous code contained dangerous patterns (os.system, eval, exec, subprocess). "
            "This is automatically penalized. "
            "This time: write ONLY pure Python logic. No shell commands. No dynamic execution. "
            "Use only stdlib modules that do not involve system access."
        ),
    },
    "CORRECT": {
        "what_went_wrong": "No failure — agent succeeded.",
        "strategy": "Continue with same strategy.",
        "plan": ["Maintain current approach."],
        "system_addon": "",
    },
}

# Default template for unknown failures
DEFAULT_TEMPLATE = {
    "what_went_wrong": "Agent failed to fix the bug sufficiently — score too low.",
    "strategy": "Read all relevant files carefully, make a targeted fix, run tests, submit.",
    "plan": [
        "1. Read failing test to understand expected behavior",
        "2. Read each source file referenced by the test",
        "3. Identify the bug: wrong return value, missing case, logic error",
        "4. Write minimal fix",
        "5. Run tests",
        "6. Submit only when tests improve",
    ],
    "system_addon": (
        "IMPORTANT: Your previous attempt scored below 0.5. "
        "This time: focus on understanding what the failing test EXPECTS. "
        "Make a targeted fix. Verify with run_tests before submitting."
    ),
}


class SelfImprovementEngine:
    """
    Generates structured improvement plans from failure analysis.

    Usage:
        engine = SelfImprovementEngine()
        plan = engine.generate_improvement_plan(
            episode_id="abc123",
            task="task1",
            failure_report=failure_report,
            trajectory_steps=[...],
        )
        # Then inject plan.system_prompt_addon into the agent's next run
    """

    def generate_improvement_plan(
        self,
        episode_id: str,
        task: str,
        failure_type: str,
        failure_evidence: List[str],
        original_score: float,
        trajectory_steps: List[dict],
        files_read: List[str],
        files_written: List[str],
    ) -> ImprovementPlan:
        """Generate an actionable improvement plan from failure data."""
        template = STRATEGY_TEMPLATES.get(failure_type, DEFAULT_TEMPLATE)

        # Build specific error list from trajectory
        specific_errors = []
        for step in trajectory_steps:
            if step.get("error"):
                specific_errors.append(
                    f"Step {step.get('step_number', '?')}: {step['error'][:100]}"
                )
        specific_errors.extend(failure_evidence[:3])

        # Build user context addon with trajectory summary
        action_summary = []
        for step in trajectory_steps[:8]:  # First 8 steps for context
            a = step.get("action_type", "?")
            p = step.get("action_path") or step.get("action_query") or ""
            r = step.get("reward", 0)
            err = " ❌" if step.get("error") else ""
            action_summary.append(f"  Step {step.get('step_number', '?')}: {a} {p} → reward={r:+.2f}{err}")

        user_context_addon = (
            f"[PREVIOUS ATTEMPT REVIEW]\n"
            f"Score: {original_score:.2f}/1.0\n"
            f"Primary failure: {failure_type}\n"
            f"What went wrong: {template['what_went_wrong']}\n"
            f"\nYour previous actions:\n" + "\n".join(action_summary) +
            f"\n\n[IMPROVED STRATEGY FOR THIS ATTEMPT]\n{template['strategy']}"
        )

        return ImprovementPlan(
            episode_id=episode_id,
            task=task,
            failure_type=failure_type,
            original_score=original_score,
            what_went_wrong=template["what_went_wrong"],
            specific_errors=specific_errors,
            improved_strategy=template["strategy"],
            step_by_step_plan=template["plan"],
            system_prompt_addon=template["system_addon"],
            user_context_addon=user_context_addon,
        )

    def build_retry_system_prompt(self, base_prompt: str, improvement_plan: ImprovementPlan) -> str:
        """Inject improvement guidance into the agent system prompt."""
        if not improvement_plan.system_prompt_addon:
            return base_prompt
        return (
            f"{base_prompt}\n\n"
            f"{'='*60}\n"
            f"PREVIOUS ATTEMPT FEEDBACK (VERY IMPORTANT):\n"
            f"{'='*60}\n"
            f"{improvement_plan.system_prompt_addon}\n"
            f"{'='*60}"
        )

    def build_retry_user_context(self, improvement_plan: ImprovementPlan) -> str:
        """Build the user context string to prepend to the first prompt in a retry."""
        return improvement_plan.user_context_addon
