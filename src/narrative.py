"""
Narrative retrieval strategy for ragtime.

Uses a manifest index for cheap routing and serves full documents
instead of chunks, preserving narrative context and architectural decisions.
"""

import re
from pathlib import Path
from dataclasses import dataclass

import yaml

from .indexers.docs import parse_frontmatter


@dataclass
class ManifestEntry:
    """A single entry in the narrative manifest index."""
    file_path: str
    scope: str
    trigger: str = ""
    mtime: float = 0.0
    auto_generated: bool = True

    def to_line(self) -> str:
        """Serialize to a markdown list item."""
        parts = f"- `{self.file_path}` — {self.scope}"
        if self.trigger:
            parts += f" {self.trigger}"
        return parts


def parse_manifest(content: str) -> list[ManifestEntry]:
    """Parse a narrative-index.md file into ManifestEntry objects."""
    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue

        # Pattern: - `file/path` — scope text. optional trigger.
        match = re.match(r'^- `([^`]+)`\s*[—–-]\s*(.+)$', line)
        if not match:
            continue

        file_path = match.group(1)
        rest = match.group(2).strip()

        entries.append(ManifestEntry(
            file_path=file_path,
            scope=rest,
            auto_generated=False,  # Parsed from file = potentially hand-edited
        ))

    return entries


def generate_manifest(project_path: Path, doc_paths: list[str] | None = None) -> list[ManifestEntry]:
    """
    Generate manifest entries by scanning .ragtime/app/ and configured doc paths.

    For each markdown file found:
    1. If frontmatter has `scope` and/or `trigger`, use those directly
    2. Otherwise, auto-generate scope from frontmatter fields + first paragraph

    Args:
        project_path: Root of the project
        doc_paths: Additional doc directories to scan (from config.docs.paths)

    Returns:
        List of auto-generated ManifestEntry objects
    """
    entries = []
    ragtime_dir = project_path / ".ragtime"

    # Scan .ragtime/app/ for memories
    app_dir = ragtime_dir / "app"
    if app_dir.exists():
        for md_file in sorted(app_dir.rglob("*.md")):
            entry = _entry_from_file(md_file, project_path)
            if entry:
                entries.append(entry)

    # Scan .ragtime/team/ for team knowledge
    team_dir = ragtime_dir / "team"
    if team_dir.exists():
        for md_file in sorted(team_dir.rglob("*.md")):
            entry = _entry_from_file(md_file, project_path)
            if entry:
                entries.append(entry)

    # Scan configured doc paths
    for doc_path_str in (doc_paths or []):
        doc_dir = project_path / doc_path_str
        if not doc_dir.exists():
            continue
        for md_file in sorted(doc_dir.rglob("*.md")):
            entry = _entry_from_file(md_file, project_path)
            if entry:
                entries.append(entry)

    return entries


def _entry_from_file(file_path: Path, project_path: Path) -> ManifestEntry | None:
    """Create a ManifestEntry from a markdown file."""
    try:
        content = file_path.read_text()
    except (IOError, UnicodeDecodeError):
        return None

    rel_path = str(file_path.relative_to(project_path))
    frontmatter, body = parse_frontmatter(content)

    # Use explicit scope/trigger from frontmatter if present
    scope = frontmatter.get("scope", "")
    trigger = frontmatter.get("trigger", "")

    if not scope:
        scope = _auto_scope(frontmatter, body, file_path)

    if not scope:
        return None

    mtime = file_path.stat().st_mtime

    return ManifestEntry(
        file_path=rel_path,
        scope=scope,
        trigger=trigger,
        mtime=mtime,
        auto_generated=not frontmatter.get("scope"),
    )


def _auto_scope(frontmatter: dict, body: str, file_path: Path) -> str:
    """Auto-generate a scope summary from frontmatter and content."""
    parts = []

    # Pull structured info from frontmatter
    mem_type = frontmatter.get("type", "")
    component = frontmatter.get("component", "")
    namespace = frontmatter.get("namespace", "")

    if component:
        parts.append(component.replace("-", " ").title())
    if mem_type and mem_type not in ("docs", "code", "note"):
        parts.append(mem_type.replace("-", " "))

    # Extract first meaningful paragraph from body
    first_para = _first_paragraph(body)
    if first_para:
        # Truncate to ~120 chars for a concise summary
        if len(first_para) > 120:
            first_para = first_para[:117].rsplit(" ", 1)[0] + "..."
        parts.append(first_para)
    elif not parts:
        # Fallback to filename
        stem = file_path.stem
        # Strip ID prefix (8-char hex + hyphen)
        cleaned = re.sub(r'^[a-f0-9]{8}-', '', stem)
        parts.append(cleaned.replace("-", " ").title())

    return ". ".join(parts) if parts else ""


def _first_paragraph(body: str) -> str:
    """Extract the first non-empty paragraph from markdown body."""
    lines = []
    in_paragraph = False

    for line in body.splitlines():
        stripped = line.strip()

        # Skip headers and empty lines before first paragraph
        if not in_paragraph:
            if not stripped or stripped.startswith("#"):
                continue
            in_paragraph = True

        if in_paragraph:
            if not stripped:
                break  # End of paragraph
            if stripped.startswith("#"):
                break  # Hit next header
            lines.append(stripped)

    return " ".join(lines)


def update_manifest(
    project_path: Path,
    index_file: str,
    doc_paths: list[str] | None = None,
) -> int:
    """
    Update the narrative manifest, preserving manual edits.

    Strategy:
    - Parse existing manifest into dict keyed by file_path
    - Generate fresh entries for all docs
    - For each entry: if file_path exists in manifest and source file hasn't changed,
      keep the existing entry (preserves manual edits)
    - New files get appended, deleted files get removed

    Args:
        project_path: Root of the project
        index_file: Relative path to the manifest file
        doc_paths: Doc directories to scan

    Returns:
        Number of entries in the updated manifest
    """
    manifest_path = project_path / index_file

    # Parse existing manifest and get its mtime for staleness comparison
    existing: dict[str, ManifestEntry] = {}
    manifest_mtime: float = 0.0
    if manifest_path.exists():
        try:
            manifest_mtime = manifest_path.stat().st_mtime
            existing_content = manifest_path.read_text()
            for entry in parse_manifest(existing_content):
                existing[entry.file_path] = entry
        except (IOError, UnicodeDecodeError):
            pass

    # Generate fresh entries
    fresh = generate_manifest(project_path, doc_paths)

    # Merge: preserve manual edits for unchanged files.
    # Compare source file mtime against manifest mtime — if the source file
    # is older than the manifest, it hasn't changed since the last write,
    # so any manual edits to the manifest entry should be preserved.
    merged: list[ManifestEntry] = []
    for entry in fresh:
        old = existing.get(entry.file_path)
        if old and manifest_mtime:
            # Check if source file changed since manifest was last written
            source_path = project_path / entry.file_path
            if source_path.exists():
                source_mtime = source_path.stat().st_mtime
                if source_mtime <= manifest_mtime:
                    # Source hasn't changed since last manifest write — keep existing entry
                    merged.append(old)
                    continue

        merged.append(entry)

    # Write manifest grouped by directory section
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Narrative Index", ""]

    # Group entries by their top-level directory path
    groups: dict[str, list[ManifestEntry]] = {}
    for entry in merged:
        section = _section_for_path(entry.file_path)
        groups.setdefault(section, []).append(entry)

    for section, entries_in_group in groups.items():
        lines.append(f"## {section}")
        lines.append("")
        for entry in entries_in_group:
            lines.append(entry.to_line())
        lines.append("")

    manifest_path.write_text("\n".join(lines) + "\n")

    return len(merged)


def _section_for_path(file_path: str) -> str:
    """Derive a human-readable section name from a file path.

    .ragtime/app/auth/foo.md  → "app/auth"
    .ragtime/app/foo.md       → "app"
    .ragtime/team/foo.md      → "team"
    docs/api/foo.md           → "docs/api"
    docs/foo.md               → "docs"
    """
    parts = file_path.replace("\\", "/").split("/")

    # Strip leading .ragtime/ prefix
    if parts and parts[0] == ".ragtime":
        parts = parts[1:]

    # Use directory path (everything except the filename)
    if len(parts) > 1:
        return "/".join(parts[:-1])
    return parts[0] if parts else "other"


# --- Query matching ---

def match_query(query: str, entries: list[ManifestEntry], max_matches: int = 3) -> list[ManifestEntry]:
    """
    Match a search query against manifest entries.

    Uses keyword overlap scoring — no embeddings needed.
    The manifest is small enough (~10-50 entries) that brute-force is fine.

    Args:
        query: Natural language search query
        entries: Manifest entries to match against
        max_matches: Maximum entries to return

    Returns:
        Matched entries sorted by relevance (best first)
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, ManifestEntry]] = []

    for entry in entries:
        scope_tokens = _tokenize(entry.scope)
        trigger_tokens = _tokenize(entry.trigger)
        path_tokens = _tokenize(entry.file_path.replace("/", " ").replace("-", " ").replace(".", " "))
        all_tokens = scope_tokens | trigger_tokens | path_tokens

        if not all_tokens:
            continue

        # Score by overlap — exact matches + substring matches (auth ⊂ authentication)
        match_count = _fuzzy_overlap_count(query_tokens, all_tokens)
        if match_count == 0:
            continue

        score = match_count / len(query_tokens)

        # Boost if query tokens match trigger-specific words
        trigger_match = _fuzzy_overlap_count(query_tokens, trigger_tokens)
        if trigger_match:
            score += 0.3 * trigger_match / len(query_tokens)

        # Boost for path component matches
        path_match = _fuzzy_overlap_count(query_tokens, path_tokens)
        if path_match:
            score += 0.2 * path_match / len(query_tokens)

        scored.append((score, entry))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: -x[0])

    # Minimum threshold — at least 1 meaningful token must match
    min_score = 1.0 / max(len(query_tokens), 1)
    return [entry for score, entry in scored[:max_matches] if score >= min_score]


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set, filtering stop words."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "because", "if", "when", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "it", "its",
        "read", "use", "about", "any", "all",
    }
    words = set(re.findall(r'\b\w{2,}\b', text.lower()))
    return words - stop_words


def _fuzzy_overlap_count(query_tokens: set[str], target_tokens: set[str]) -> int:
    """Count query tokens that match target tokens, including substring matches.

    Handles cases like 'authentication' matching 'auth' (one contains the other).
    """
    count = 0
    for qt in query_tokens:
        for tt in target_tokens:
            if qt == tt or qt in tt or tt in qt:
                count += 1
                break
    return count


def read_full_doc(project_path: Path, file_path: str, max_tokens: int = 2000) -> str | None:
    """
    Read a full document for narrative retrieval.

    Args:
        project_path: Root of the project
        file_path: Relative path to the document
        max_tokens: Approximate max tokens (using ~4 chars/token heuristic)

    Returns:
        Full document content, or None if too large or unreadable
    """
    full_path = project_path / file_path
    if not full_path.exists():
        return None

    try:
        content = full_path.read_text()
    except (IOError, UnicodeDecodeError):
        return None

    # Rough token estimate: ~4 chars per token
    approx_tokens = len(content) // 4
    if approx_tokens > max_tokens:
        # Truncate to approximate token limit, break at paragraph boundary
        char_limit = max_tokens * 4
        truncated = content[:char_limit]
        # Try to break at last paragraph boundary
        last_para = truncated.rfind("\n\n")
        if last_para > char_limit * 0.5:
            truncated = truncated[:last_para]
        return truncated + "\n\n[... truncated — full doc exceeds token limit ...]"

    return content
