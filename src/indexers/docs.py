"""
Docs indexer - parses markdown files with YAML frontmatter.

Designed for .claude/memory/ style files but works with any markdown.
"""

import re
from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class DocEntry:
    """A parsed document ready for indexing."""
    content: str
    file_path: str
    namespace: str | None = None
    category: str | None = None
    component: str | None = None
    title: str | None = None

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict."""
        return {
            "type": "docs",
            "file": self.file_path,
            "namespace": self.namespace or "default",
            "category": self.category or "",
            "component": self.component or "",
            "title": self.title or Path(self.file_path).stem,
        }


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.

    Returns (metadata_dict, body_content).
    If no frontmatter, returns ({}, full_content).
    """
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
        body = match.group(2)
        return metadata, body
    except yaml.YAMLError:
        return {}, content


def index_file(file_path: Path) -> DocEntry | None:
    """
    Parse a single markdown file into a DocEntry.

    Returns None if file can't be parsed.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return None

    metadata, body = parse_frontmatter(content)

    # Skip empty documents
    if not body.strip():
        return None

    return DocEntry(
        content=body.strip(),
        file_path=str(file_path),
        namespace=metadata.get("namespace"),
        category=metadata.get("category"),
        component=metadata.get("component"),
        title=metadata.get("title"),
    )


def discover_docs(
    root: Path,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    """
    Find all markdown files to index.

    Args:
        root: Directory to search
        patterns: Glob patterns to include (default: ["**/*.md"])
        exclude: Patterns to exclude (default: ["**/node_modules/**", "**/.git/**"])
    """
    patterns = patterns or ["**/*.md"]
    exclude = exclude or ["**/node_modules/**", "**/.git/**", "**/.ragtime/**"]

    files = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                # Check exclusions
                skip = False
                for ex in exclude:
                    if path.match(ex):
                        skip = True
                        break
                if not skip:
                    files.append(path)

    return files


def index_directory(root: Path, **kwargs) -> list[DocEntry]:
    """
    Index all markdown files in a directory.

    Returns list of DocEntry objects ready for vector DB.
    """
    files = discover_docs(root, **kwargs)
    entries = []

    for file_path in files:
        entry = index_file(file_path)
        if entry:
            entries.append(entry)

    return entries
