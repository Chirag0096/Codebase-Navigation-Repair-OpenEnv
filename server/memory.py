# server/memory.py
"""
Context and memory optimization tracker.

Records what the agent has seen, how much context it consumed,
and detects wasteful patterns (re-reading, reading irrelevant content).

This answers: "How efficiently does the agent use its context window?"
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FileReadRecord:
    """Record of a single file read."""
    path: str
    size_bytes: int
    read_count: int
    was_relevant: bool
    first_read_step: int


@dataclass
class MemoryStats:
    """Comprehensive context usage statistics."""
    total_bytes_read: int = 0
    unique_bytes_read: int = 0
    redundant_bytes_read: int = 0
    total_files_read: int = 0
    unique_files_read: int = 0
    redundant_reads: int = 0
    relevant_files_read: int = 0
    irrelevant_files_read: int = 0
    context_efficiency: float = 0.0     # unique_useful / total
    search_queries: int = 0
    total_content_written: int = 0      # bytes written by agent

    def to_dict(self) -> dict:
        return {
            "total_bytes_read": self.total_bytes_read,
            "unique_bytes_read": self.unique_bytes_read,
            "redundant_bytes_read": self.redundant_bytes_read,
            "total_files_read": self.total_files_read,
            "unique_files_read": self.unique_files_read,
            "redundant_reads": self.redundant_reads,
            "relevant_files_read": self.relevant_files_read,
            "irrelevant_files_read": self.irrelevant_files_read,
            "context_efficiency": round(self.context_efficiency, 3),
            "search_queries": self.search_queries,
            "total_content_written": self.total_content_written,
        }


class MemoryTracker:
    """
    Tracks agent's context consumption and memory patterns.

    Usage:
        tracker = MemoryTracker()
        tracker.start_episode(relevant_files=["src/auth.py", "tests/test_auth.py"])
        tracker.record_read("src/auth.py", 500, step=1)
        tracker.record_read("src/auth.py", 500, step=3)  # redundant!
        stats = tracker.get_stats()
    """

    def __init__(self):
        self._reads: Dict[str, FileReadRecord] = {}
        self._relevant_files: set = set()
        self._search_count: int = 0
        self._bytes_written: int = 0

    def start_episode(self, relevant_files: List[str] = None):
        """Reset tracker for new episode."""
        self._reads.clear()
        self._relevant_files = set(relevant_files or [])
        self._search_count = 0
        self._bytes_written = 0

    def record_read(self, path: str, size_bytes: int, step: int):
        """Record a file read action."""
        if path in self._reads:
            self._reads[path].read_count += 1
        else:
            self._reads[path] = FileReadRecord(
                path=path,
                size_bytes=size_bytes,
                read_count=1,
                was_relevant=path in self._relevant_files,
                first_read_step=step,
            )

    def record_search(self):
        """Record a search query."""
        self._search_count += 1

    def record_write(self, content_bytes: int):
        """Record bytes written by agent."""
        self._bytes_written += content_bytes

    def get_stats(self) -> MemoryStats:
        """Compute comprehensive memory statistics."""
        total_bytes = 0
        unique_bytes = 0
        redundant_bytes = 0
        redundant_reads = 0
        relevant_count = 0
        irrelevant_count = 0

        for record in self._reads.values():
            first_read_bytes = record.size_bytes
            unique_bytes += first_read_bytes
            total_bytes += first_read_bytes * record.read_count

            if record.read_count > 1:
                redundant_reads += record.read_count - 1
                redundant_bytes += first_read_bytes * (record.read_count - 1)

            if record.was_relevant:
                relevant_count += 1
            else:
                irrelevant_count += 1

        # Context efficiency: what fraction of bytes read was useful (relevant + unique)?
        relevant_bytes = sum(
            r.size_bytes for r in self._reads.values() if r.was_relevant
        )
        efficiency = relevant_bytes / max(1, total_bytes)

        return MemoryStats(
            total_bytes_read=total_bytes,
            unique_bytes_read=unique_bytes,
            redundant_bytes_read=redundant_bytes,
            total_files_read=sum(r.read_count for r in self._reads.values()),
            unique_files_read=len(self._reads),
            redundant_reads=redundant_reads,
            relevant_files_read=relevant_count,
            irrelevant_files_read=irrelevant_count,
            context_efficiency=efficiency,
            search_queries=self._search_count,
            total_content_written=self._bytes_written,
        )

    def get_wasteful_patterns(self) -> List[str]:
        """Identify specific wasteful patterns for debugging."""
        patterns = []

        # Files read multiple times
        for record in self._reads.values():
            if record.read_count > 1:
                patterns.append(
                    f"REDUNDANT_READ: '{record.path}' read {record.read_count} times "
                    f"({record.size_bytes * record.read_count} bytes wasted)"
                )

        # Irrelevant files read
        for record in self._reads.values():
            if not record.was_relevant and record.read_count > 0:
                patterns.append(
                    f"IRRELEVANT_READ: '{record.path}' not in relevant files "
                    f"({record.size_bytes} bytes wasted)"
                )

        return patterns
