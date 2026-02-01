"""
Feedback loop for RAG result quality improvement.

Tracks which search results are actually used/referenced by Claude,
enabling re-ranking and quality improvements over time.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class SearchFeedback:
    """Feedback for a single search result."""
    query: str
    result_id: str  # ChromaDB document ID
    result_file: str  # File path for easier debugging
    action: str  # "used", "referenced", "ignored", "helpful", "not_helpful"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None  # Group related searches
    position: int = 0  # Position in results (1-indexed)
    distance: float = 0.0  # Original semantic distance


class FeedbackStore:
    """
    Simple file-based feedback storage.

    Stores feedback as JSON lines for easy analysis.
    Can be upgraded to SQLite or ChromaDB later.
    """

    def __init__(self, path: Path):
        """
        Initialize feedback store.

        Args:
            path: Directory to store feedback data
        """
        self.path = path
        self.feedback_file = path / "feedback.jsonl"
        self.stats_file = path / "feedback_stats.json"
        path.mkdir(parents=True, exist_ok=True)

    def record(self, feedback: SearchFeedback) -> None:
        """Record a single feedback entry."""
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(asdict(feedback)) + "\n")

    def record_usage(
        self,
        query: str,
        result_id: str,
        result_file: str,
        position: int = 0,
        distance: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Convenience method to record when a result is used."""
        self.record(SearchFeedback(
            query=query,
            result_id=result_id,
            result_file=result_file,
            action="used",
            position=position,
            distance=distance,
            session_id=session_id,
        ))

    def record_batch(
        self,
        query: str,
        used_ids: list[str],
        all_results: list[dict],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Record feedback for a batch of results.

        Marks used_ids as "used" and others as "ignored".
        """
        used_set = set(used_ids)

        for i, result in enumerate(all_results):
            result_id = result.get("id", "")
            result_file = result.get("metadata", {}).get("file", "")
            distance = result.get("distance", 0.0)

            action = "used" if result_id in used_set else "ignored"

            self.record(SearchFeedback(
                query=query,
                result_id=result_id,
                result_file=result_file,
                action=action,
                position=i + 1,
                distance=distance,
                session_id=session_id,
            ))

    def get_usage_stats(self) -> dict:
        """
        Get aggregated usage statistics.

        Returns:
            Dict with usage counts, popular files, etc.
        """
        if not self.feedback_file.exists():
            return {"total": 0, "used": 0, "ignored": 0}

        stats = {
            "total": 0,
            "used": 0,
            "ignored": 0,
            "helpful": 0,
            "not_helpful": 0,
            "files_used": {},  # file -> count
            "avg_position_used": 0.0,
        }

        positions = []

        with open(self.feedback_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    stats["total"] += 1
                    action = entry.get("action", "")

                    if action == "used":
                        stats["used"] += 1
                        positions.append(entry.get("position", 0))
                        file_path = entry.get("result_file", "")
                        stats["files_used"][file_path] = stats["files_used"].get(file_path, 0) + 1
                    elif action == "ignored":
                        stats["ignored"] += 1
                    elif action == "helpful":
                        stats["helpful"] += 1
                    elif action == "not_helpful":
                        stats["not_helpful"] += 1
                except json.JSONDecodeError:
                    continue

        if positions:
            stats["avg_position_used"] = sum(positions) / len(positions)

        return stats

    def get_boost_scores(self) -> dict[str, float]:
        """
        Calculate boost scores for files based on historical usage.

        Returns:
            Dict mapping file paths to boost multipliers (1.0 = no boost).
        """
        stats = self.get_usage_stats()
        files_used = stats.get("files_used", {})

        if not files_used:
            return {}

        # Normalize to 0-1 range, then convert to boost multiplier
        max_count = max(files_used.values())
        boosts = {}

        for file_path, count in files_used.items():
            # Boost range: 1.0 (no boost) to 1.5 (50% boost for most-used)
            normalized = count / max_count
            boosts[file_path] = 1.0 + (normalized * 0.5)

        return boosts

    def apply_boosts(self, results: list[dict], boosts: dict[str, float]) -> list[dict]:
        """
        Apply historical boost scores to search results.

        Adjusts distances based on historical usage patterns.
        Lower distance = more relevant, so we divide by boost.
        """
        if not boosts:
            return results

        for result in results:
            file_path = result.get("metadata", {}).get("file", "")
            boost = boosts.get(file_path, 1.0)
            if "distance" in result and result["distance"]:
                # Reduce distance for frequently-used files
                result["distance"] = result["distance"] / boost
                result["boosted"] = boost > 1.0

        # Re-sort by adjusted distance
        return sorted(results, key=lambda r: r.get("distance", float("inf")))

    def clear(self) -> None:
        """Clear all feedback data."""
        if self.feedback_file.exists():
            self.feedback_file.unlink()
