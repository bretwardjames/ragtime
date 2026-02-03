"""
Docs indexer - parses markdown files with YAML frontmatter.

Designed for .ragtime/ style files but works with any markdown.
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
    # Line tracking for section-level ephemeral detection
    line_number: int = 1  # Starting line of this section
    # Status tracking for ephemeral/permanent
    status: str = "ephemeral"     # "ephemeral" (working tree) or "permanent" (from main)
    branch: str | None = None     # Branch name for ephemeral entries

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict."""
        meta = {
            "type": "docs",
            "file": self.file_path,
            "namespace": self.namespace or "default",
            "category": self.category or "",
            "component": self.component or "",
            "title": self.title or Path(self.file_path).stem,
            "mtime": self.mtime or 0.0,
            "section_path": self.section_path or "",
            "section_level": self.section_level,
            "line": self.line_number,
            "status": self.status,
        }
        if self.branch:
            meta["branch"] = self.branch
        return meta


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
    is_convention: bool = False  # True if this section contains conventions


# Headers that indicate convention content
CONVENTION_HEADERS = {
    "conventions", "convention", "rules", "standards", "guidelines",
    "code conventions", "coding conventions", "code standards",
    "coding standards", "style guide", "code style",
}


def chunk_by_headers(
    content: str,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    convention_sections: list[str] | None = None,
    all_conventions: bool = False,
) -> list[Section]:
    """
    Split markdown into sections by headers, preserving hierarchy.

    Args:
        content: Markdown body (without frontmatter)
        min_chunk_size: Minimum chars to make a standalone section
        max_chunk_size: Maximum chars before splitting further
        convention_sections: List of section titles to mark as conventions
        all_conventions: If True, mark ALL sections as conventions

    Returns:
        List of Section objects with hierarchical context
    """
    lines = content.split('\n')
    sections: list[Section] = []
    header_stack: list[tuple[int, str, bool]] = []  # (level, title, is_convention)

    current_section_lines: list[str] = []
    current_section_start = 0
    current_title = ""
    current_level = 0
    current_is_convention = False
    in_convention_marker = False  # Track <!-- convention --> blocks

    def is_convention_header(title: str) -> bool:
        """Check if header indicates convention content."""
        normalized = title.lower().strip()
        return normalized in CONVENTION_HEADERS or any(
            conv in normalized for conv in ["convention", "rule", "standard", "guideline"]
        )

    def flush_section():
        """Save accumulated lines as a section."""
        nonlocal current_section_lines, current_section_start, current_title, current_level, current_is_convention

        text = '\n'.join(current_section_lines).strip()
        if text:
            # Build parent path from stack (excluding current)
            parent_path = [h[1] for h in header_stack[:-1]] if header_stack else []

            # Check if any parent is a convention header
            parent_is_convention = any(h[2] for h in header_stack[:-1]) if header_stack else False

            # Determine if this section is a convention
            is_conv = (
                all_conventions or
                current_is_convention or
                parent_is_convention or
                (convention_sections and current_title in convention_sections)
            )

            sections.append(Section(
                title=current_title or "Introduction",
                level=current_level,
                content=text,
                line_start=current_section_start,
                parent_path=parent_path,
                is_convention=is_conv,
            ))
        current_section_lines = []

    for i, line in enumerate(lines):
        # Detect convention markers
        if '<!-- convention -->' in line.lower() or '<!-- conventions -->' in line.lower():
            in_convention_marker = True
            continue
        if '<!-- /convention -->' in line.lower() or '<!-- /conventions -->' in line.lower():
            in_convention_marker = False
            continue

        # Detect markdown headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            # Save previous section
            flush_section()

            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Check if this is a convention header
            is_conv_header = is_convention_header(title)

            # Update header stack - pop headers at same or lower level
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title, is_conv_header))

            current_title = title
            current_level = level
            current_section_start = i
            current_is_convention = is_conv_header or in_convention_marker
            current_section_lines = [line]  # Include header in content
        else:
            # If inside a convention marker, mark the content
            if in_convention_marker and not current_is_convention:
                current_is_convention = True
            current_section_lines.append(line)

    # Don't forget the last section
    flush_section()

    # Post-process: merge tiny sections into parents, split huge ones
    processed: list[Section] = []
    for section in sections:
        if len(section.content) < min_chunk_size and processed:
            # Merge into previous section (inherit is_convention if either has it)
            processed[-1].content += '\n\n' + section.content
            if section.is_convention:
                processed[-1].is_convention = True
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
                        is_convention=section.is_convention,
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
                    is_convention=section.is_convention,
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

    # Convention detection from frontmatter
    has_conventions = metadata.get("has_conventions", False)
    convention_sections = metadata.get("convention_sections", [])

    # Check if filename indicates conventions
    filename_lower = file_path.stem.lower()
    is_convention_file = any(
        term in filename_lower
        for term in ["convention", "conventions", "rules", "standards", "guidelines"]
    )

    # Short docs: return as single entry
    if not hierarchical or len(body) < 500:
        # Determine category - convention file or has_conventions flag
        category = base_category
        if is_convention_file or has_conventions:
            category = "convention"

        return [DocEntry(
            content=body.strip(),
            file_path=str(file_path),
            namespace=base_namespace,
            category=category,
            component=base_component,
            title=base_title,
            mtime=mtime,
            section_path="",
            section_level=0,
            chunk_index=0,
        )]

    # Hierarchical chunking for longer docs
    sections = chunk_by_headers(
        body,
        convention_sections=convention_sections,
        all_conventions=has_conventions or is_convention_file,
    )
    entries = []

    for i, section in enumerate(sections):
        # Build full section path: "Parent > Child > Current"
        path_parts = section.parent_path + [section.title]
        section_path = " > ".join(path_parts)

        # Prepend context for better embeddings
        context_prefix = f"# {base_title}\n"
        if section.parent_path:
            context_prefix += f"Section: {' > '.join(section.parent_path)}\n\n"

        # Determine category - convention sections get "convention" category
        category = base_category
        if section.is_convention:
            category = "convention"

        entries.append(DocEntry(
            content=context_prefix + section.content,
            file_path=str(file_path),
            namespace=base_namespace,
            category=category,
            component=base_component,
            title=section.title,
            mtime=mtime,
            section_path=section_path,
            section_level=section.level,
            chunk_index=i,
            line_number=section.line_start,
        ))

    return entries


def index_content(file_path: str, content: str, hierarchical: bool = True,
                  status: str = "ephemeral", branch: str | None = None) -> list[DocEntry]:
    """
    Parse markdown content into DocEntry objects.

    Unlike index_file, this takes content directly (for indexing from git refs).

    Args:
        file_path: Path to attribute to the entries (for display/filtering)
        content: The markdown content to parse
        hierarchical: If True, chunk by headers for better semantic search
        status: "ephemeral" or "permanent"
        branch: Branch name (for ephemeral entries)

    Returns:
        List of DocEntry objects
    """
    metadata, body = parse_frontmatter(content)

    if not body.strip():
        return []

    path_obj = Path(file_path)

    # Base metadata from frontmatter
    base_namespace = metadata.get("namespace")
    base_category = metadata.get("category")
    base_component = metadata.get("component")
    base_title = metadata.get("title") or path_obj.stem

    # Convention detection from frontmatter
    has_conventions = metadata.get("has_conventions", False)
    convention_sections = metadata.get("convention_sections", [])

    # Check if filename indicates conventions
    filename_lower = path_obj.stem.lower()
    is_convention_file = any(
        term in filename_lower
        for term in ["convention", "conventions", "rules", "standards", "guidelines"]
    )

    # Short docs: return as single entry
    if not hierarchical or len(body) < 500:
        category = base_category
        if is_convention_file or has_conventions:
            category = "convention"

        return [DocEntry(
            content=body.strip(),
            file_path=file_path,
            namespace=base_namespace,
            category=category,
            component=base_component,
            title=base_title,
            mtime=0.0,  # No mtime for git ref content
            section_path="",
            section_level=0,
            chunk_index=0,
            status=status,
            branch=branch,
        )]

    # Hierarchical chunking for longer docs
    sections = chunk_by_headers(
        body,
        convention_sections=convention_sections,
        all_conventions=has_conventions or is_convention_file,
    )
    entries = []

    for i, section in enumerate(sections):
        path_parts = section.parent_path + [section.title]
        section_path = " > ".join(path_parts)

        context_prefix = f"# {base_title}\n"
        if section.parent_path:
            context_prefix += f"Section: {' > '.join(section.parent_path)}\n\n"

        category = base_category
        if section.is_convention:
            category = "convention"

        entries.append(DocEntry(
            content=context_prefix + section.content,
            file_path=file_path,
            namespace=base_namespace,
            category=category,
            component=base_component,
            title=section.title,
            mtime=0.0,
            section_path=section_path,
            section_level=section.level,
            chunk_index=i,
            line_number=section.line_start,
            status=status,
            branch=branch,
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
