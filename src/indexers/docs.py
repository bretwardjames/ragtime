"""
Docs indexer - parses markdown files with YAML frontmatter.

Designed for .claude/memory/ style files but works with any markdown.
"""

import os
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
    mtime: float | None = None  # File modification time for incremental indexing
    # Hierarchical chunking fields
    section_path: str | None = None  # e.g., "Installation > Configuration > Environment Variables"
    section_level: int = 0  # Header depth (0=whole doc, 1=h1, 2=h2, etc.)
    chunk_index: int = 0  # Position within file (for stable IDs)

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict."""
        return {
            "type": "docs",
            "file": self.file_path,
            "namespace": self.namespace or "default",
            "category": self.category or "",
            "component": self.component or "",
            "title": self.title or Path(self.file_path).stem,
            "mtime": self.mtime or 0.0,
            "section_path": self.section_path or "",
            "section_level": self.section_level,
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


@dataclass
class Section:
    """A markdown section for hierarchical chunking."""
    title: str
    level: int  # 1-6 for h1-h6
    content: str
    line_start: int
    parent_path: list[str]  # Parent headers for context


def chunk_by_headers(
    content: str,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
) -> list[Section]:
    """
    Split markdown into sections by headers, preserving hierarchy.

    Args:
        content: Markdown body (without frontmatter)
        min_chunk_size: Minimum chars to make a standalone section
        max_chunk_size: Maximum chars before splitting further

    Returns:
        List of Section objects with hierarchical context
    """
    lines = content.split('\n')
    sections: list[Section] = []
    header_stack: list[tuple[int, str]] = []  # (level, title) for building paths

    current_section_lines: list[str] = []
    current_section_start = 0
    current_title = ""
    current_level = 0

    def flush_section():
        """Save accumulated lines as a section."""
        nonlocal current_section_lines, current_section_start, current_title, current_level

        text = '\n'.join(current_section_lines).strip()
        if text:
            # Build parent path from stack (excluding current)
            parent_path = [h[1] for h in header_stack[:-1]] if header_stack else []

            sections.append(Section(
                title=current_title or "Introduction",
                level=current_level,
                content=text,
                line_start=current_section_start,
                parent_path=parent_path,
            ))
        current_section_lines = []

    for i, line in enumerate(lines):
        # Detect markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            # Save previous section
            flush_section()

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Update header stack - pop headers at same or lower level
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))

            current_title = title
            current_level = level
            current_section_start = i
            current_section_lines = [line]  # Include header in content
        else:
            current_section_lines.append(line)

    # Don't forget the last section
    flush_section()

    # Post-process: merge tiny sections into parents, split huge ones
    processed: list[Section] = []
    for section in sections:
        if len(section.content) < min_chunk_size and processed:
            # Merge into previous section
            processed[-1].content += '\n\n' + section.content
        elif len(section.content) > max_chunk_size:
            # Split by paragraphs
            paragraphs = re.split(r'\n\n+', section.content)
            current_chunk = ""
            chunk_num = 0

            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                    processed.append(Section(
                        title=f"{section.title} (part {chunk_num + 1})",
                        level=section.level,
                        content=current_chunk.strip(),
                        line_start=section.line_start,
                        parent_path=section.parent_path,
                    ))
                    current_chunk = para
                    chunk_num += 1
                else:
                    current_chunk += '\n\n' + para if current_chunk else para

            if current_chunk.strip():
                title = f"{section.title} (part {chunk_num + 1})" if chunk_num > 0 else section.title
                processed.append(Section(
                    title=title,
                    level=section.level,
                    content=current_chunk.strip(),
                    line_start=section.line_start,
                    parent_path=section.parent_path,
                ))
        else:
            processed.append(section)

    return processed


def index_file(file_path: Path, hierarchical: bool = True) -> list[DocEntry]:
    """
    Parse a single markdown file into DocEntry objects.

    Args:
        file_path: Path to the markdown file
        hierarchical: If True, chunk by headers for better semantic search.
                     If False, return whole file as single entry.

    Returns:
        List of DocEntry objects (one per section if hierarchical, else one for whole file).
        Empty list if file can't be parsed.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        mtime = os.path.getmtime(file_path)
    except (IOError, UnicodeDecodeError, OSError):
        return []

    metadata, body = parse_frontmatter(content)

    # Skip empty documents
    if not body.strip():
        return []

    # Base metadata from frontmatter
    base_namespace = metadata.get("namespace")
    base_category = metadata.get("category")
    base_component = metadata.get("component")
    base_title = metadata.get("title") or file_path.stem

    # Short docs: return as single entry
    if not hierarchical or len(body) < 500:
        return [DocEntry(
            content=body.strip(),
            file_path=str(file_path),
            namespace=base_namespace,
            category=base_category,
            component=base_component,
            title=base_title,
            mtime=mtime,
            section_path="",
            section_level=0,
            chunk_index=0,
        )]

    # Hierarchical chunking for longer docs
    sections = chunk_by_headers(body)
    entries = []

    for i, section in enumerate(sections):
        # Build full section path: "Parent > Child > Current"
        path_parts = section.parent_path + [section.title]
        section_path = " > ".join(path_parts)

        # Prepend context for better embeddings
        context_prefix = f"# {base_title}\n"
        if section.parent_path:
            context_prefix += f"Section: {' > '.join(section.parent_path)}\n\n"

        entries.append(DocEntry(
            content=context_prefix + section.content,
            file_path=str(file_path),
            namespace=base_namespace,
            category=base_category,
            component=base_component,
            title=section.title,
            mtime=mtime,
            section_path=section_path,
            section_level=section.level,
            chunk_index=i,
        ))

    return entries


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


def index_directory(root: Path, hierarchical: bool = True, **kwargs) -> list[DocEntry]:
    """
    Index all markdown files in a directory.

    Args:
        root: Directory to search
        hierarchical: If True, chunk long docs by headers
        **kwargs: Passed to discover_docs (patterns, exclude)

    Returns:
        List of DocEntry objects ready for vector DB.
    """
    files = discover_docs(root, **kwargs)
    entries = []

    for file_path in files:
        file_entries = index_file(file_path, hierarchical=hierarchical)
        entries.extend(file_entries)

    return entries
