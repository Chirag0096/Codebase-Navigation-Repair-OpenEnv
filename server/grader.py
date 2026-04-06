# server/grader.py
"""
Grading utilities for computing final scores.
"""
from .sandbox import run_pytest_sandboxed


def compute_final_score(repo_path: str, test_file: str = None) -> float:
    """
    Run pytest and return the pass rate as the final score.
    Returns float in [0.0, 1.0].
    """
    pass_rate, output, timed_out = run_pytest_sandboxed(repo_path, test_file)
    if timed_out:
        return 0.0
    return min(1.0, max(0.0, pass_rate))
