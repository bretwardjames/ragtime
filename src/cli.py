"""
Ragtime CLI - semantic search and memory storage.
"""

from pathlib import Path
import subprocess
import click
import os
import signal
import sys

from .db import RagtimeDB
from .config import RagtimeConfig, init_config
from .indexers import (
    discover_docs, index_doc_file, DocEntry,
    discover_code_files, index_code_file, CodeEntry,
)
from .memory import Memory, MemoryStore


def get_db(project_path: Path) -> RagtimeDB:
    """Get or create database for a project."""
    db_path = project_path / ".ragtime" / "index"
    return RagtimeDB(db_path)


def get_memory_store(project_path: Path) -> MemoryStore:
    """Get memory store for a project."""
    db = get_db(project_path)
    return MemoryStore(project_path, db)


def get_author() -> str:
    """Get the current developer's username."""
    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().lower().replace(" ", "-")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "unknown"


def check_ghp_installed() -> bool:
    """Check if ghp-cli is installed."""
    try:
        result = subprocess.run(
            ["ghp", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_issue_from_ghp(issue_num: int, path: Path) -> dict | None:
    """Get issue details using ghp issue open."""
    import json
    try:
        result = subprocess.run(
            ["ghp", "issue", "open", str(issue_num), "--json"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def get_issue_from_gh(issue_num: int, path: Path) -> dict | None:
    """Get issue details using gh CLI."""
    import json
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_num), "--json", "title,body,labels,number"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def get_current_branch(path: Path) -> str | None:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_branch_slug(ref: str) -> str:
    """Convert a git ref to a branch slug for folder naming."""
    if ref.startswith("origin/"):
        ref = ref[7:]
    return ref.replace("/", "-")


def get_remote_branches_with_ragtime(path: Path) -> list[str]:
    """Get list of remote branches that have .ragtime/branches/ content."""
    try:
        # Get all remote branches
        result = subprocess.run(
            ["git", "branch", "-r", "--format=%(refname:short)"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        branches = []
        for ref in result.stdout.strip().split("\n"):
            if not ref or ref.endswith("/HEAD"):
                continue

            # Check if this branch has ragtime content
            check = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", ref, ".ragtime/branches/"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if check.returncode == 0 and check.stdout.strip():
                branches.append(ref)

        return branches
    except Exception:
        return []


@click.group()
@click.version_option(version="0.2.8")
def main():
    """Ragtime - semantic search over code and documentation."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), default=".")
def init(path: Path):
    """Initialize ragtime config for a project."""
    path = path.resolve()
    config = init_config(path)
    click.echo(f"Created .ragtime/config.yaml with defaults:")
    click.echo(f"  Docs paths: {config.docs.paths}")
    click.echo(f"  Code paths: {config.code.paths}")
    click.echo(f"  Languages: {config.code.languages}")

    # Create directory structure
    ragtime_dir = path / ".ragtime"
    (ragtime_dir / "app").mkdir(parents=True, exist_ok=True)
    (ragtime_dir / "team").mkdir(parents=True, exist_ok=True)
    (ragtime_dir / "branches").mkdir(parents=True, exist_ok=True)
    (ragtime_dir / "archive" / "branches").mkdir(parents=True, exist_ok=True)

    # Create .gitkeep files
    for subdir in ["app", "team", "archive/branches"]:
        gitkeep = ragtime_dir / subdir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    # Create .gitignore for synced branches (dot-prefixed)
    gitignore_path = ragtime_dir / ".gitignore"
    gitignore_content = """# Synced branches from teammates (dot-prefixed)
branches/.*

# Index database
index/
"""
    gitignore_path.write_text(gitignore_content)

    # Create conventions file template
    conventions_file = ragtime_dir / "CONVENTIONS.md"
    if not conventions_file.exists():
        conventions_file.write_text("""# Team Conventions

Rules and patterns that code must follow. These are checked by `/create-pr`.

## Code Style

- [ ] Example: Use async/await, not .then() chains
- [ ] Example: All API endpoints must use auth middleware

## Architecture

- [ ] Example: Services should not directly access repositories from other domains

## Security

- [ ] Example: Never commit .env or credentials files
- [ ] Example: All user input must be validated

## Testing

- [ ] Example: All new features need unit tests

---

Add your team's conventions above. Each rule should be:
- Clear and specific
- Checkable against code
- Actionable (what to do, not just what not to do)
""")

    click.echo(f"\nCreated .ragtime/ structure:")
    click.echo(f"  app/           - Graduated knowledge (tracked)")
    click.echo(f"  team/          - Team conventions (tracked)")
    click.echo(f"  branches/      - Active branches (yours tracked, synced gitignored)")
    click.echo(f"  archive/       - Completed branches (tracked)")
    click.echo(f"  CONVENTIONS.md - Team rules checked by /create-pr")

    # Check for ghp-cli
    if check_ghp_installed():
        click.echo(f"\n✓ ghp-cli detected")
        click.echo(f"  Run 'ragtime setup-ghp' to enable auto-context on 'ghp start'")
    else:
        click.echo(f"\n• ghp-cli not found")
        click.echo(f"  Install for enhanced workflow: npm install -g @bretwardjames/ghp-cli")


# Batch size for ChromaDB upserts (embedding computation happens here)
INDEX_BATCH_SIZE = 100


def _upsert_entries(db, entries, entry_type: str = "docs", label: str = "  Embedding"):
    """Upsert entries to ChromaDB in batches with progress bar."""
    if not entries:
        return

    # Process in batches with progress feedback
    batches = [entries[i:i + INDEX_BATCH_SIZE] for i in range(0, len(entries), INDEX_BATCH_SIZE)]

    with click.progressbar(
        batches,
        label=label,
        show_percent=True,
        show_pos=True,
        item_show_func=lambda b: f"{len(b)} items" if b else "",
    ) as batch_iter:
        for batch in batch_iter:
            if entry_type == "code":
                ids = [f"{e.file_path}:{e.line_number}:{e.symbol_name}" for e in batch]
            else:
                ids = [e.file_path for e in batch]

            documents = [e.content for e in batch]
            metadatas = [e.to_metadata() for e in batch]
            db.upsert(ids=ids, documents=documents, metadatas=metadatas)


def _get_files_to_process(
    all_files: list[Path],
    indexed_files: dict[str, float],
) -> tuple[list[Path], list[str]]:
    """
    Compare files on disk with indexed files to determine what needs processing.

    Returns:
        (files_to_index, files_to_delete)
    """
    disk_files = {str(f): os.path.getmtime(f) for f in all_files}

    to_index = []
    for file_path in all_files:
        path_str = str(file_path)
        disk_mtime = disk_files[path_str]
        indexed_mtime = indexed_files.get(path_str, 0.0)

        # Index if new or modified (with 1-second tolerance for filesystem precision)
        if disk_mtime > indexed_mtime + 1.0:
            to_index.append(file_path)

    # Find deleted files (in index but not on disk)
    to_delete = [f for f in indexed_files.keys() if f not in disk_files]

    return to_index, to_delete


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--type", "index_type", type=click.Choice(["all", "docs", "code"]), default="all")
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
def index(path: Path, index_type: str, clear: bool):
    """Index a project directory.

    Without --clear, performs incremental indexing (only changed files).
    """
    path = path.resolve()
    db = get_db(path)
    config = RagtimeConfig.load(path)

    if clear:
        click.echo("Clearing existing index...")
        if index_type == "all":
            db.clear()
        else:
            db.clear(type_filter=index_type)

    if index_type in ("all", "docs"):
        # Get currently indexed docs
        indexed_docs = {} if clear else db.get_indexed_files("docs")

        # Discover all doc files
        all_doc_files = []
        for docs_path in config.docs.paths:
            docs_root = path / docs_path
            if not docs_root.exists():
                click.echo(f"  Docs path {docs_root} not found, skipping...")
                continue
            files = discover_docs(
                docs_root,
                patterns=config.docs.patterns,
                exclude=config.docs.exclude,
            )
            all_doc_files.extend(files)

        if all_doc_files or indexed_docs:
            # Determine what needs processing
            to_index, to_delete = _get_files_to_process(all_doc_files, indexed_docs)

            click.echo(f"Found {len(all_doc_files)} doc files")
            if not clear:
                unchanged = len(all_doc_files) - len(to_index)
                if unchanged > 0:
                    click.echo(f"  {unchanged} unchanged, {len(to_index)} to index")
                if to_delete:
                    click.echo(f"  {len(to_delete)} to remove (deleted from disk)")

            # Delete removed files
            if to_delete:
                db.delete_by_file(to_delete, "docs")

            # Index new/changed files
            if to_index:
                entries = []
                with click.progressbar(
                    to_index,
                    label="  Parsing",
                    show_percent=True,
                    show_pos=True,
                    item_show_func=lambda f: f.name[:30] if f else "",
                ) as files:
                    for file_path in files:
                        entry = index_doc_file(file_path)
                        if entry:
                            entries.append(entry)

                if entries:
                    _upsert_entries(db, entries, "docs")
                    click.echo(f"  Indexed {len(entries)} documents")
            elif not to_delete:
                click.echo("  All docs up to date")
        else:
            click.echo("  No documents found")

    if index_type in ("all", "code"):
        # Get currently indexed code files
        indexed_code = {} if clear else db.get_indexed_files("code")

        # Build exclusion list for code
        code_exclude = list(config.code.exclude)
        for docs_path in config.docs.paths:
            code_exclude.append(f"**/{docs_path}/**")

        # Discover all code files
        all_code_files = []
        for code_path_str in config.code.paths:
            code_root = path / code_path_str
            if not code_root.exists():
                click.echo(f"  Code path {code_root} not found, skipping...")
                continue
            files = discover_code_files(
                code_root,
                languages=config.code.languages,
                exclude=code_exclude,
            )
            all_code_files.extend(files)

        if all_code_files or indexed_code:
            # Determine what needs processing
            to_index, to_delete = _get_files_to_process(all_code_files, indexed_code)

            click.echo(f"Found {len(all_code_files)} code files")
            if not clear:
                unchanged = len(all_code_files) - len(to_index)
                if unchanged > 0:
                    click.echo(f"  {unchanged} unchanged, {len(to_index)} to index")
                if to_delete:
                    click.echo(f"  {len(to_delete)} to remove (deleted from disk)")

            # Delete removed files
            if to_delete:
                db.delete_by_file(to_delete, "code")

            # Index new/changed files
            if to_index:
                entries = []
                by_type = {}
                with click.progressbar(
                    to_index,
                    label="  Parsing",
                    show_percent=True,
                    show_pos=True,
                    item_show_func=lambda f: f.name[:30] if f else "",
                ) as files:
                    for file_path in files:
                        file_entries = index_code_file(file_path)
                        for entry in file_entries:
                            entries.append(entry)
                            by_type[entry.symbol_type] = by_type.get(entry.symbol_type, 0) + 1

                if entries:
                    click.echo(f"  Found {len(entries)} symbols")
                    _upsert_entries(db, entries, "code")
                    click.echo(f"  Indexed {len(entries)} code symbols")
                    breakdown = ", ".join(f"{count} {typ}s" for typ, count in sorted(by_type.items()))
                    click.echo(f"    ({breakdown})")
            elif not to_delete:
                click.echo("  All code up to date")
        else:
            click.echo("  No code files found")

    stats = db.stats()
    click.echo(f"\nIndex stats: {stats['total']} total ({stats['docs']} docs, {stats['code']} code)")


@main.command()
@click.argument("query")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--type", "type_filter", type=click.Choice(["all", "docs", "code"]), default="all")
@click.option("--namespace", "-n", help="Filter by namespace")
@click.option("--include-archive", is_flag=True, help="Also search archived branches")
@click.option("--limit", "-l", default=5, help="Max results")
@click.option("--verbose", "-v", is_flag=True, help="Show full content")
def search(query: str, path: Path, type_filter: str, namespace: str,
           include_archive: bool, limit: int, verbose: bool):
    """Search indexed content."""
    path = Path(path).resolve()
    db = get_db(path)

    type_arg = None if type_filter == "all" else type_filter

    results = db.search(
        query=query,
        limit=limit,
        type_filter=type_arg,
        namespace=namespace,
    )

    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        score = 1 - distance if distance else None

        click.echo(f"\n{'─' * 60}")
        click.echo(f"[{i}] {meta.get('file', 'unknown')}")
        click.echo(f"    Type: {meta.get('type')} | Namespace: {meta.get('namespace', '-')}")
        if score:
            click.echo(f"    Score: {score:.3f}")

        if verbose:
            click.echo(f"\n{result['content'][:500]}...")
        else:
            preview = result["content"][:150].replace("\n", " ")
            click.echo(f"    {preview}...")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def stats(path: Path):
    """Show index statistics."""
    path = Path(path).resolve()
    db = get_db(path)

    s = db.stats()
    click.echo(f"Total indexed: {s['total']}")
    click.echo(f"  Docs: {s['docs']}")
    click.echo(f"  Code: {s['code']}")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--type", "type_filter", type=click.Choice(["all", "docs", "code"]), default="all")
@click.confirmation_option(prompt="Are you sure you want to clear the index?")
def clear(path: Path, type_filter: str):
    """Clear the index."""
    path = Path(path).resolve()
    db = get_db(path)

    if type_filter == "all":
        db.clear()
        click.echo("Index cleared.")
    else:
        db.clear(type_filter=type_filter)
        click.echo(f"Cleared {type_filter} from index.")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def config(path: Path):
    """Show current configuration."""
    path = Path(path).resolve()
    cfg = RagtimeConfig.load(path)

    click.echo("Docs:")
    click.echo(f"  Paths: {cfg.docs.paths}")
    click.echo(f"  Patterns: {cfg.docs.patterns}")
    click.echo(f"  Exclude: {cfg.docs.exclude}")
    click.echo("\nCode:")
    click.echo(f"  Paths: {cfg.code.paths}")
    click.echo(f"  Languages: {cfg.code.languages}")
    click.echo(f"  Exclude: {cfg.code.exclude}")
    click.echo("\nConventions:")
    click.echo(f"  Files: {cfg.conventions.files}")
    click.echo(f"  Also search memories: {cfg.conventions.also_search_memories}")


# ============================================================================
# Memory Storage Commands
# ============================================================================


@main.command()
@click.argument("content")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--namespace", "-n", required=True, help="Namespace: app, team, user-{name}, branch-{name}")
@click.option("--type", "-t", "memory_type", required=True,
              type=click.Choice(["architecture", "feature", "integration", "convention",
                                 "preference", "decision", "pattern", "task-state", "handoff"]),
              help="Memory type")
@click.option("--component", "-c", help="Component area (e.g., auth, claims, shifts)")
@click.option("--confidence", default="medium",
              type=click.Choice(["high", "medium", "low"]),
              help="Confidence level")
@click.option("--confidence-reason", help="Why this confidence level")
@click.option("--source", "-s", default="remember", help="Source of this memory")
@click.option("--issue", help="Related GitHub issue (e.g., #301)")
@click.option("--epic", help="Parent epic (e.g., #286)")
@click.option("--branch", help="Related branch name")
def remember(content: str, path: Path, namespace: str, memory_type: str,
             component: str, confidence: str, confidence_reason: str,
             source: str, issue: str, epic: str, branch: str):
    """Store a memory with structured metadata.

    Example:
        ragtime remember "Auth uses JWT with 15-min expiry" \\
            --namespace app --type architecture --component auth
    """
    path = Path(path).resolve()
    store = get_memory_store(path)

    memory = Memory(
        content=content,
        namespace=namespace,
        type=memory_type,
        component=component,
        confidence=confidence,
        confidence_reason=confidence_reason,
        source=source,
        author=get_author(),
        issue=issue,
        epic=epic,
        branch=branch,
    )

    file_path = store.save(memory)
    click.echo(f"✓ Memory saved: {memory.id}")
    click.echo(f"  File: {file_path.relative_to(path)}")
    click.echo(f"  Namespace: {namespace}")
    click.echo(f"  Type: {memory_type}")


@main.command("store-doc")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--namespace", "-n", required=True, help="Namespace for the document")
@click.option("--type", "-t", "doc_type", default="handoff",
              type=click.Choice(["handoff", "document", "plan", "notes"]),
              help="Document type")
def store_doc(file: Path, path: Path, namespace: str, doc_type: str):
    """Store a document verbatim (like handoff.md)."""
    path = Path(path).resolve()
    file = Path(file).resolve()
    store = get_memory_store(path)

    memory = store.store_document(file, namespace, doc_type)
    click.echo(f"✓ Document stored: {memory.id}")
    click.echo(f"  Source: {file.name}")
    click.echo(f"  Namespace: {namespace}")
    click.echo(f"  Type: {doc_type}")


@main.command()
@click.argument("memory_id")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.confirmation_option(prompt="Are you sure you want to delete this memory?")
def forget(memory_id: str, path: Path):
    """Delete a memory by ID."""
    path = Path(path).resolve()
    store = get_memory_store(path)

    if store.delete(memory_id):
        click.echo(f"✓ Memory {memory_id} deleted")
    else:
        click.echo(f"✗ Memory {memory_id} not found", err=True)


@main.command()
@click.argument("memory_id")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--confidence", default="high",
              type=click.Choice(["high", "medium", "low"]),
              help="Confidence level for graduated memory")
def graduate(memory_id: str, path: Path, confidence: str):
    """Graduate a branch memory to app namespace."""
    path = Path(path).resolve()
    store = get_memory_store(path)

    try:
        graduated = store.graduate(memory_id, confidence)
        if graduated:
            click.echo(f"✓ Memory graduated to app namespace")
            click.echo(f"  New ID: {graduated.id}")
            click.echo(f"  Original marked as: graduated")
        else:
            click.echo(f"✗ Memory {memory_id} not found", err=True)
    except ValueError as e:
        click.echo(f"✗ {e}", err=True)


@main.command("memories")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--namespace", "-n", help="Filter by namespace (use * suffix for prefix match)")
@click.option("--type", "-t", "type_filter", help="Filter by type")
@click.option("--status", "-s", help="Filter by status (active, graduated, abandoned)")
@click.option("--component", "-c", help="Filter by component")
@click.option("--limit", "-l", default=20, help="Max results")
@click.option("--verbose", "-v", is_flag=True, help="Show full content")
def list_memories(path: Path, namespace: str, type_filter: str, status: str,
                  component: str, limit: int, verbose: bool):
    """List memories with optional filters."""
    path = Path(path).resolve()
    store = get_memory_store(path)

    memories = store.list_memories(
        namespace=namespace,
        type_filter=type_filter,
        status=status,
        component=component,
        limit=limit,
    )

    if not memories:
        click.echo("No memories found.")
        return

    click.echo(f"Found {len(memories)} memories:\n")

    for mem in memories:
        click.echo(f"{'─' * 60}")
        click.echo(f"[{mem.id}] {mem.namespace} / {mem.type}")
        if mem.component:
            click.echo(f"    Component: {mem.component}")
        click.echo(f"    Status: {mem.status} | Confidence: {mem.confidence}")
        click.echo(f"    Added: {mem.added} | Source: {mem.source}")

        if verbose:
            click.echo(f"\n{mem.content[:500]}...")
        else:
            preview = mem.content[:100].replace("\n", " ")
            click.echo(f"    {preview}...")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def reindex(path: Path):
    """Reindex all memory files."""
    path = Path(path).resolve()
    store = get_memory_store(path)

    count = store.reindex()
    click.echo(f"✓ Reindexed {count} memory files")


@main.command("new-branch")
@click.argument("issue", type=int)
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--content", help="Context document content (overrides auto-generated scaffold)")
@click.option("--issue-json", "issue_json", help="Issue data as JSON (from ghp hook, skips fetch)")
@click.option("--branch", "-b", help="Branch name (auto-detected from git if not provided)")
def new_branch(issue: int, path: Path, content: str, issue_json: str, branch: str):
    """Initialize a branch context from a GitHub issue.

    Creates .ragtime/branches/{branch-slug}/context.md with either:
    - Provided content (from --content flag, e.g., LLM-generated plan)
    - Auto-generated scaffold from issue metadata (fallback)
    """
    import json
    from datetime import date

    path = Path(path).resolve()

    if not branch:
        branch = get_current_branch(path)
        if not branch or branch in ("main", "master"):
            click.echo("✗ Not on a feature branch. Use --branch to specify.", err=True)
            return

    # Create branch slug for folder name
    branch_slug = branch.replace("/", "-")
    branch_dir = path / ".ragtime" / "branches" / branch_slug
    branch_dir.mkdir(parents=True, exist_ok=True)

    context_file = branch_dir / "context.md"

    if content:
        context_file.write_text(content)
        click.echo(f"✓ Created context.md with provided content")
        click.echo(f"  Path: {context_file.relative_to(path)}")
        return

    # Get issue data
    issue_data = None
    source = None

    if issue_json:
        try:
            issue_data = json.loads(issue_json)
            source = "ghp-hook"
        except json.JSONDecodeError as e:
            click.echo(f"✗ Invalid JSON: {e}", err=True)
            return
    else:
        click.echo(f"Fetching issue #{issue}...")
        if check_ghp_installed():
            issue_data = get_issue_from_ghp(issue, path)
            source = "ghp"
        if not issue_data:
            issue_data = get_issue_from_gh(issue, path)
            source = "gh"

    if not issue_data:
        click.echo(f"✗ Could not fetch issue #{issue}", err=True)
        return

    title = issue_data.get("title", f"Issue #{issue}")
    body = issue_data.get("body", "")
    labels = issue_data.get("labels", [])

    if labels:
        if isinstance(labels[0], dict):
            label_names = [l.get("name", "") for l in labels]
        else:
            label_names = labels
        labels_str = ", ".join(label_names)
    else:
        labels_str = ""

    scaffold = f"""---
type: context
branch: {branch}
issue: {issue}
status: active
created: '{date.today().isoformat()}'
author: {get_author()}
---

## Issue

**#{issue}**: {title}

{f"**Labels**: {labels_str}" if labels_str else ""}

## Description

{body if body else "_No description provided_"}

## Plan

<!-- Implementation steps - fill in or let Claude generate -->

- [ ] TODO: Define implementation steps

## Acceptance Criteria

<!-- What needs to be true for this to be complete? -->

## Notes

<!-- Additional context, decisions, blockers -->

"""

    context_file.write_text(scaffold)

    click.echo(f"✓ Created context.md from issue #{issue}")
    click.echo(f"  Path: {context_file.relative_to(path)}")
    click.echo(f"  Source: {source}")


# ============================================================================
# Command Installation
# ============================================================================


def get_commands_dir() -> Path:
    """Get the directory containing bundled command templates."""
    return Path(__file__).parent / "commands"


def get_available_commands() -> list[str]:
    """List available command templates."""
    commands_dir = get_commands_dir()
    if not commands_dir.exists():
        return []
    return [f.stem for f in commands_dir.glob("*.md")]


@main.command("install")
@click.option("--global", "global_install", is_flag=True, help="Install to ~/.claude/commands/")
@click.option("--workspace", "workspace_install", is_flag=True, help="Install to .claude/commands/")
@click.option("--list", "list_commands", is_flag=True, help="List available commands")
@click.option("--force", is_flag=True, help="Overwrite existing commands without asking")
@click.argument("commands", nargs=-1)
def install_commands(global_install: bool, workspace_install: bool, list_commands: bool,
                     force: bool, commands: tuple):
    """Install Claude command templates."""
    available = get_available_commands()

    if list_commands:
        click.echo("Available commands:")
        for cmd in available:
            click.echo(f"  - {cmd}")
        return

    if global_install and workspace_install:
        click.echo("Error: Cannot specify both --global and --workspace", err=True)
        return

    if global_install:
        target_dir = Path.home() / ".claude" / "commands"
    elif workspace_install:
        target_dir = Path.cwd() / ".claude" / "commands"
    else:
        target_dir = Path.cwd() / ".claude" / "commands"
        click.echo("Installing to workspace (.claude/commands/)")

    if commands:
        to_install = [c for c in commands if c in available]
        not_found = [c for c in commands if c not in available]
        if not_found:
            click.echo(f"Warning: Commands not found: {', '.join(not_found)}", err=True)
    else:
        to_install = available

    if not to_install:
        click.echo("No commands to install.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    commands_dir = get_commands_dir()
    installed = 0
    skipped = 0
    namespaced = 0

    for cmd in to_install:
        source = commands_dir / f"{cmd}.md"
        target = target_dir / f"{cmd}.md"
        namespaced_target = target_dir / f"ragtime-{cmd}.md"

        if target.exists() and not force:
            # Check if it's our file (contains ragtime marker)
            existing_content = target.read_text()
            is_ragtime_file = "ragtime" in existing_content.lower() and "mcp__ragtime" in existing_content

            if is_ragtime_file:
                # It's our file, safe to overwrite
                target.write_text(source.read_text())
                click.echo(f"  ✓ {cmd}.md (updated)")
                installed += 1
            else:
                # Conflict with non-ragtime command
                click.echo(f"\n⚠️  Conflict: {cmd}.md already exists (not a ragtime command)")
                click.echo(f"   1. Overwrite with ragtime's version")
                click.echo(f"   2. Skip (keep existing)")
                click.echo(f"   3. Install as ragtime-{cmd}.md")

                choice = click.prompt("   Choice", type=click.Choice(["1", "2", "3"]), default="2")

                if choice == "1":
                    target.write_text(source.read_text())
                    click.echo(f"  ✓ {cmd}.md (overwritten)")
                    installed += 1
                elif choice == "2":
                    click.echo(f"  • {cmd}.md (skipped)")
                    skipped += 1
                else:
                    namespaced_target.write_text(source.read_text())
                    click.echo(f"  ✓ ragtime-{cmd}.md")
                    namespaced += 1
        else:
            target.write_text(source.read_text())
            click.echo(f"  ✓ {cmd}.md")
            installed += 1

    click.echo(f"\nInstalled {installed} commands to {target_dir}")
    if namespaced:
        click.echo(f"  ({namespaced} installed with ragtime- prefix)")
    if skipped:
        click.echo(f"  ({skipped} skipped due to conflicts)")


@main.command("setup-ghp")
@click.option("--remove", is_flag=True, help="Remove ragtime hooks from ghp")
def setup_ghp(remove: bool):
    """Register ragtime hooks with ghp-cli."""
    if not check_ghp_installed():
        click.echo("✗ ghp-cli not installed", err=True)
        return

    hook_name = "ragtime-context"

    if remove:
        result = subprocess.run(
            ["ghp", "hooks", "remove", hook_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            click.echo(f"✓ Removed hook: {hook_name}")
        else:
            click.echo(f"• Hook {hook_name} not registered")
        return

    result = subprocess.run(
        ["ghp", "hooks", "show", hook_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        click.echo(f"• Hook {hook_name} already registered")
        return

    # Updated path for .ragtime/
    hook_command = "ragtime new-branch ${issue.number} --issue-json '${issue.json}' --branch '${branch}'"

    result = subprocess.run(
        [
            "ghp", "hooks", "add", hook_name,
            "--event", "issue-started",
            "--command", hook_command,
            "--display-name", "Ragtime Context",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        click.echo(f"✓ Registered hook: {hook_name}")
        click.echo(f"  Event: issue-started")
        click.echo(f"  Action: Creates context.md from issue metadata")
    else:
        click.echo(f"✗ Failed to register hook: {result.stderr}", err=True)


# ============================================================================
# Cross-Branch Sync Commands
# ============================================================================


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output (for automated runs)")
@click.option("--auto-prune", is_flag=True, help="Automatically prune stale synced branches")
def sync(path: Path, quiet: bool, auto_prune: bool):
    """Sync memories from all remote branches.

    Fetches .ragtime/branches/* from remote branches and copies to
    local dot-prefixed folders (e.g., .feature-branch/).
    """
    import shutil

    path = Path(path).resolve()
    branches_dir = path / ".ragtime" / "branches"

    if not quiet:
        click.echo("Fetching remote branches...")

    # Fetch first
    subprocess.run(
        ["git", "fetch", "--quiet"],
        cwd=path,
        capture_output=True,
    )

    # Get current branch to exclude
    current = get_current_branch(path)
    current_slug = get_branch_slug(current) if current else None

    # Find remote branches with ragtime content
    remote_branches = get_remote_branches_with_ragtime(path)

    if not remote_branches and not quiet:
        click.echo("No remote branches with ragtime content found.")

    synced = 0
    for ref in remote_branches:
        branch_slug = get_branch_slug(ref)

        # Skip current branch
        if branch_slug == current_slug:
            continue

        # Synced folders are dot-prefixed
        synced_dir = branches_dir / f".{branch_slug}"

        # Get files from remote
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref, ".ragtime/branches/"],
            cwd=path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            continue

        files = result.stdout.strip().split("\n")

        # Clear and recreate synced folder
        if synced_dir.exists():
            shutil.rmtree(synced_dir)
        synced_dir.mkdir(parents=True, exist_ok=True)

        # Extract files
        for file_path in files:
            if not file_path.endswith(".md"):
                continue

            content_result = subprocess.run(
                ["git", "show", f"{ref}:{file_path}"],
                cwd=path,
                capture_output=True,
                text=True,
            )

            if content_result.returncode == 0:
                filename = Path(file_path).name
                (synced_dir / filename).write_text(content_result.stdout)

        synced += 1
        if not quiet:
            click.echo(f"  ✓ Synced .{branch_slug}")

    # Check for stale synced branches (dot-prefixed with undotted counterpart)
    stale = []
    if branches_dir.exists():
        for folder in branches_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("."):
                undotted = folder.name[1:]
                undotted_path = branches_dir / undotted
                if undotted_path.exists():
                    stale.append(folder)

    if stale:
        if not quiet:
            click.echo(f"\nStale synced branches detected:")
            for folder in stale:
                click.echo(f"  • {folder.name} → {folder.name[1:]} exists (merged)")

        if auto_prune:
            for folder in stale:
                shutil.rmtree(folder)
            if not quiet:
                click.echo(f"\n✓ Pruned {len(stale)} stale branches")
        elif not quiet:
            if click.confirm("\nPrune stale branches?", default=True):
                for folder in stale:
                    shutil.rmtree(folder)
                click.echo(f"✓ Pruned {len(stale)} stale branches")

    if not quiet:
        click.echo(f"\nDone. Synced {synced} branches.")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--dry-run", is_flag=True, help="Show what would be pruned")
def prune(path: Path, dry_run: bool):
    """Remove stale synced branch folders.

    Removes dot-prefixed folders (.branch) when an undotted
    counterpart (branch) exists (indicating the branch was merged).
    """
    import shutil

    path = Path(path).resolve()
    branches_dir = path / ".ragtime" / "branches"

    if not branches_dir.exists():
        click.echo("No branches directory found.")
        return

    # Find dot-prefixed folders with undotted counterparts
    to_prune = []
    for folder in branches_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("."):
            undotted = folder.name[1:]
            if (branches_dir / undotted).exists():
                to_prune.append(folder)

    if not to_prune:
        click.echo("Nothing to prune.")
        return

    click.echo("Will prune:")
    for folder in to_prune:
        click.echo(f"  ✗ {folder.name} → {folder.name[1:]} exists")

    if dry_run:
        click.echo(f"\n--dry-run: Would prune {len(to_prune)} folders")
    else:
        for folder in to_prune:
            shutil.rmtree(folder)
            click.echo(f"  Pruned: {folder.name}")
        click.echo(f"\n✓ Pruned {len(to_prune)} folders")


# ============================================================================
# Daemon Commands
# ============================================================================


def get_pid_file(path: Path) -> Path:
    """Get path to daemon PID file."""
    return path / ".ragtime" / "daemon.pid"


def get_log_file(path: Path) -> Path:
    """Get path to daemon log file."""
    return path / ".ragtime" / "daemon.log"


@main.group()
def daemon():
    """Manage the ragtime sync daemon."""
    pass


@daemon.command("start")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--interval", default="5m", help="Sync interval (e.g., 5m, 1h)")
def daemon_start(path: Path, interval: str):
    """Start the sync daemon.

    Runs git fetch && ragtime sync on an interval to keep
    remote branches synced automatically.

    Note: This command requires Unix (Linux/macOS). On Windows, use Task Scheduler instead.
    """
    # Check for Windows - os.fork() is Unix-only
    if sys.platform == "win32":
        click.echo("✗ Daemon mode is not supported on Windows.", err=True)
        click.echo("  Use Windows Task Scheduler to run 'ragtime sync' periodically instead.")
        return

    path = Path(path).resolve()
    pid_file = get_pid_file(path)
    log_file = get_log_file(path)

    # Check if already running
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            click.echo(f"Daemon already running (PID: {pid})")
            return
        except OSError:
            pid_file.unlink()

    # Parse interval
    interval_seconds = 300  # default 5m
    if interval.endswith("m"):
        interval_seconds = int(interval[:-1]) * 60
    elif interval.endswith("h"):
        interval_seconds = int(interval[:-1]) * 3600
    elif interval.endswith("s"):
        interval_seconds = int(interval[:-1])
    else:
        try:
            interval_seconds = int(interval)
        except ValueError:
            click.echo(f"Invalid interval: {interval}", err=True)
            return

    # Fork daemon process
    pid = os.fork()
    if pid > 0:
        # Parent process
        click.echo(f"✓ Daemon started (PID: {pid})")
        click.echo(f"  Interval: {interval}")
        click.echo(f"  Log: {log_file.relative_to(path)}")
        click.echo(f"\nStop with: ragtime daemon stop")
        return

    # Child process - become daemon
    os.setsid()

    # Write PID file
    pid_file.write_text(str(os.getpid()))

    # Redirect output to log file
    log_fd = open(log_file, "a")
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())

    import time
    from datetime import datetime

    # Set up signal handler for clean shutdown
    running = True

    def handle_shutdown(signum, frame):
        nonlocal running
        running = False
        print(f"\n[{datetime.now().isoformat()}] Received signal {signum}, shutting down...")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    print(f"\n[{datetime.now().isoformat()}] Daemon started (interval: {interval})")

    while running:
        try:
            print(f"[{datetime.now().isoformat()}] Running sync...")

            # Fetch
            subprocess.run(
                ["git", "fetch", "--quiet"],
                cwd=path,
                capture_output=True,
            )

            # Sync
            subprocess.run(
                ["ragtime", "sync", "--quiet", "--auto-prune"],
                cwd=path,
                capture_output=True,
            )

            print(f"[{datetime.now().isoformat()}] Sync complete")

        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Error: {e}")

        # Sleep in small increments to respond to signals faster
        for _ in range(interval_seconds):
            if not running:
                break
            time.sleep(1)

    # Clean up
    print(f"[{datetime.now().isoformat()}] Daemon stopped")
    log_fd.close()
    if pid_file.exists():
        pid_file.unlink()


@daemon.command("stop")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def daemon_stop(path: Path):
    """Stop the sync daemon."""
    path = Path(path).resolve()
    pid_file = get_pid_file(path)

    if not pid_file.exists():
        click.echo("Daemon not running.")
        return

    pid = int(pid_file.read_text().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink()
        click.echo(f"✓ Daemon stopped (PID: {pid})")
    except OSError:
        pid_file.unlink()
        click.echo("Daemon was not running (stale PID file removed).")


@daemon.command("status")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def daemon_status(path: Path):
    """Check daemon status."""
    path = Path(path).resolve()
    pid_file = get_pid_file(path)
    log_file = get_log_file(path)

    if not pid_file.exists():
        click.echo("Daemon: not running")
        return

    pid = int(pid_file.read_text().strip())

    try:
        os.kill(pid, 0)
        click.echo(f"Daemon: running (PID: {pid})")

        # Show last few log lines
        if log_file.exists():
            lines = log_file.read_text().strip().split("\n")
            if lines:
                click.echo(f"\nRecent log:")
                for line in lines[-5:]:
                    click.echo(f"  {line}")
    except OSError:
        click.echo("Daemon: not running (stale PID file)")
        pid_file.unlink()


# ============================================================================
# Debug Commands
# ============================================================================


@main.group()
def debug():
    """Debug and verify the vector index."""
    pass


@debug.command("search")
@click.argument("query")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--limit", "-l", default=5, help="Max results")
@click.option("--show-vectors", is_flag=True, help="Show vector statistics")
def debug_search(query: str, path: Path, limit: int, show_vectors: bool):
    """Debug a search query - show scores and ranking details."""
    path = Path(path).resolve()
    db = get_db(path)

    results = db.search(query=query, limit=limit)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nQuery: \"{query}\"")
    click.echo(f"{'─' * 60}")

    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        similarity = 1 - distance if distance else None

        click.echo(f"\n[{i}] {meta.get('file', 'unknown')}")
        click.echo(f"    Distance: {distance:.4f}")
        click.echo(f"    Similarity: {similarity:.4f} ({similarity * 100:.1f}%)")
        click.echo(f"    Namespace: {meta.get('namespace', '-')}")
        click.echo(f"    Type: {meta.get('type', '-')}")

        # Show content preview
        preview = result["content"][:100].replace("\n", " ")
        click.echo(f"    Preview: {preview}...")

    if show_vectors:
        click.echo(f"\n{'─' * 60}")
        click.echo("Vector Statistics:")
        click.echo(f"  Total indexed: {db.collection.count()}")
        click.echo(f"  Embedding model: all-MiniLM-L6-v2 (ChromaDB default)")
        click.echo(f"  Vector dimensions: 384")
        click.echo(f"  Distance metric: cosine")


@debug.command("similar")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--limit", "-l", default=5, help="Max similar docs")
def debug_similar(file_path: str, path: Path, limit: int):
    """Find documents similar to a given file."""
    path = Path(path).resolve()
    db = get_db(path)

    # Read the file content
    try:
        content = Path(file_path).read_text()
    except Exception as e:
        click.echo(f"✗ Could not read file: {e}", err=True)
        return

    # Use the content as the query
    results = db.search(query=content, limit=limit + 1)  # +1 to exclude self

    click.echo(f"\nDocuments similar to: {file_path}")
    click.echo(f"{'─' * 60}")

    shown = 0
    for result in results:
        # Skip the file itself
        if result["metadata"].get("file", "").endswith(file_path):
            continue

        shown += 1
        if shown > limit:
            break

        meta = result["metadata"]
        distance = result["distance"]
        similarity = 1 - distance if distance else None

        click.echo(f"\n[{shown}] {meta.get('file', 'unknown')}")
        click.echo(f"    Similarity: {similarity:.4f} ({similarity * 100:.1f}%)")

        preview = result["content"][:100].replace("\n", " ")
        click.echo(f"    Preview: {preview}...")


@debug.command("stats")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--by-namespace", is_flag=True, help="Show counts by namespace")
@click.option("--by-type", is_flag=True, help="Show counts by type")
def debug_stats(path: Path, by_namespace: bool, by_type: bool):
    """Show detailed index statistics."""
    path = Path(path).resolve()
    db = get_db(path)

    total = db.collection.count()
    click.echo(f"\nIndex Statistics")
    click.echo(f"{'─' * 40}")
    click.echo(f"Total documents: {total}")

    if total == 0:
        click.echo("\nIndex is empty. Run 'ragtime index' first.")
        return

    # Get all documents for analysis
    all_docs = db.collection.get()

    if by_namespace or (not by_namespace and not by_type):
        namespaces = {}
        for meta in all_docs["metadatas"]:
            ns = meta.get("namespace", "unknown")
            namespaces[ns] = namespaces.get(ns, 0) + 1

        click.echo(f"\nBy Namespace:")
        for ns, count in sorted(namespaces.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            click.echo(f"  {ns}: {count} ({pct:.1f}%)")

    if by_type or (not by_namespace and not by_type):
        types = {}
        for meta in all_docs["metadatas"]:
            t = meta.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        click.echo(f"\nBy Type:")
        for t, count in sorted(types.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            click.echo(f"  {t}: {count} ({pct:.1f}%)")


@debug.command("verify")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def debug_verify(path: Path):
    """Verify index integrity with test queries."""
    path = Path(path).resolve()
    db = get_db(path)

    total = db.collection.count()
    if total == 0:
        click.echo("✗ Index is empty. Run 'ragtime index' first.")
        return

    click.echo(f"\nVerifying index ({total} documents)...")
    click.echo(f"{'─' * 40}")

    issues = []

    # Test 1: Basic search works
    click.echo("\n1. Testing basic search...")
    try:
        results = db.search("test", limit=1)
        if results:
            click.echo("   ✓ Search returns results")
        else:
            click.echo("   ⚠ Search returned no results (might be ok if no relevant docs)")
    except Exception as e:
        click.echo(f"   ✗ Search failed: {e}")
        issues.append("Basic search failed")

    # Test 2: Check for documents with missing metadata
    click.echo("\n2. Checking metadata integrity...")
    all_docs = db.collection.get()
    missing_namespace = 0
    missing_type = 0

    for meta in all_docs["metadatas"]:
        if not meta.get("namespace"):
            missing_namespace += 1
        if not meta.get("type"):
            missing_type += 1

    if missing_namespace:
        click.echo(f"   ⚠ {missing_namespace} docs missing namespace")
    else:
        click.echo("   ✓ All docs have namespace")

    if missing_type:
        click.echo(f"   ⚠ {missing_type} docs missing type")
    else:
        click.echo("   ✓ All docs have type")

    # Test 3: Check for duplicate IDs
    click.echo("\n3. Checking for duplicates...")
    ids = all_docs["ids"]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        dup_count = len(ids) - len(unique_ids)
        click.echo(f"   ✗ {dup_count} duplicate IDs found")
        issues.append("Duplicate IDs")
    else:
        click.echo("   ✓ No duplicate IDs")

    # Test 4: Similarity sanity check
    click.echo("\n4. Testing similarity consistency...")
    if total >= 2:
        # Pick first doc and find similar
        first_content = all_docs["documents"][0]
        results = db.search(first_content[:500], limit=2)
        if results and len(results) >= 1:
            top_similarity = 1 - results[0]["distance"]
            if top_similarity > 0.9:
                click.echo(f"   ✓ Self-similarity check passed ({top_similarity:.2f})")
            else:
                click.echo(f"   ⚠ Self-similarity lower than expected ({top_similarity:.2f})")
        else:
            click.echo("   ⚠ Could not perform similarity check")
    else:
        click.echo("   - Skipped (need at least 2 docs)")

    # Summary
    click.echo(f"\n{'─' * 40}")
    if issues:
        click.echo(f"⚠ Found {len(issues)} issues:")
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.echo("✓ Index verification passed")


# ============================================================================
# Documentation Generation
# ============================================================================


@main.command()
@click.argument("code_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default="docs/api",
              help="Output directory for docs")
@click.option("--stubs", is_flag=True, help="Generate stub docs with TODOs (no AI)")
@click.option("--language", "-l", multiple=True,
              help="Languages to document (python, typescript, javascript)")
@click.option("--include-private", is_flag=True, help="Include private methods (_name)")
def generate(code_path: Path, output: Path, stubs: bool, language: tuple, include_private: bool):
    """Generate documentation from code.

    Creates markdown documentation from code structure.

    Examples:
        ragtime generate src/ --stubs           # Create stub docs
        ragtime generate src/ -o docs/api       # Specify output
        ragtime generate src/ -l python         # Python only
    """
    import ast
    import re as re_module

    code_path = Path(code_path).resolve()
    output = Path(output)

    if not stubs:
        click.echo("Use --stubs for stub generation, or /generate-docs for AI-powered docs")
        click.echo("\nExample: ragtime generate src/ --stubs")
        return

    # Determine languages
    if language:
        languages = list(language)
    else:
        languages = ["python", "typescript", "javascript"]

    # Map extensions to languages
    ext_map = {
        "python": [".py"],
        "typescript": [".ts", ".tsx"],
        "javascript": [".js", ".jsx"],
    }

    extensions = []
    for lang in languages:
        extensions.extend(ext_map.get(lang, []))

    # Find code files
    code_files = []
    for ext in extensions:
        code_files.extend(code_path.rglob(f"*{ext}"))

    # Filter out common exclusions
    exclude_patterns = ["__pycache__", "node_modules", ".venv", "venv", "dist", "build"]
    code_files = [
        f for f in code_files
        if not any(ex in str(f) for ex in exclude_patterns)
    ]

    if not code_files:
        click.echo(f"No code files found in {code_path}")
        return

    click.echo(f"Found {len(code_files)} code files")
    click.echo(f"Output: {output}/")
    click.echo(f"{'─' * 50}")

    output.mkdir(parents=True, exist_ok=True)
    generated = 0

    for code_file in code_files:
        try:
            content = code_file.read_text()
        except Exception:
            continue

        relative = code_file.relative_to(code_path)
        doc_path = output / relative.with_suffix(".md")

        # Parse based on extension
        if code_file.suffix == ".py":
            doc_content = generate_python_stub(code_file, content, include_private)
        elif code_file.suffix in [".ts", ".tsx", ".js", ".jsx"]:
            doc_content = generate_typescript_stub(code_file, content, include_private)
        else:
            continue

        if doc_content:
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(doc_content)
            try:
                doc_display = doc_path.relative_to(Path.cwd())
            except ValueError:
                doc_display = doc_path
            click.echo(f"  ✓ {relative} → {doc_display}")
            generated += 1

    click.echo(f"\n{'─' * 50}")
    click.echo(f"✓ Generated {generated} documentation stubs")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Fill in the TODO placeholders")
    click.echo(f"  2. Or use /generate-docs for AI-generated content")
    click.echo(f"  3. Run 'ragtime index' to make searchable")


def generate_python_stub(file_path: Path, content: str, include_private: bool) -> str:
    """Generate markdown stub from Python code."""
    import ast

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    lines = []
    lines.append(f"# {file_path.stem}")
    lines.append(f"\n> **File:** `{file_path}`")
    lines.append("\n## Overview\n")
    lines.append("TODO: Describe what this module does.\n")

    # Get module docstring
    if ast.get_docstring(tree):
        lines.append(f"> {ast.get_docstring(tree)}\n")

    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if not include_private and node.name.startswith("_"):
                continue
            classes.append(node)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if not include_private and node.name.startswith("_"):
                continue
            functions.append(node)

    # Document classes
    if classes:
        lines.append("---\n")
        lines.append("## Classes\n")

        for cls in classes:
            lines.append(f"### `{cls.name}`\n")
            if ast.get_docstring(cls):
                lines.append(f"{ast.get_docstring(cls)}\n")
            else:
                lines.append("TODO: Describe this class.\n")

            # Find __init__ and methods
            methods = []
            init_node = None
            for item in cls.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        init_node = item
                    elif not item.name.startswith("_") or include_private:
                        methods.append(item)

            # Constructor
            if init_node:
                lines.append("#### Constructor\n")
                sig = get_function_signature(init_node)
                lines.append(f"```python\n{sig}\n```\n")
                params = get_function_params(init_node)
                if params:
                    lines.append("| Parameter | Type | Default | Description |")
                    lines.append("|-----------|------|---------|-------------|")
                    for p in params:
                        lines.append(f"| `{p['name']}` | `{p['type']}` | {p['default']} | TODO |")
                    lines.append("")

            # Methods
            if methods:
                lines.append("#### Methods\n")
                for method in methods:
                    async_prefix = "async " if isinstance(method, ast.AsyncFunctionDef) else ""
                    ret = get_return_annotation(method)
                    lines.append(f"##### `{async_prefix}{method.name}(...) -> {ret}`\n")
                    if ast.get_docstring(method):
                        lines.append(f"{ast.get_docstring(method)}\n")
                    else:
                        lines.append("TODO: Describe this method.\n")

    # Document functions
    if functions:
        lines.append("---\n")
        lines.append("## Functions\n")

        for func in functions:
            async_prefix = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
            ret = get_return_annotation(func)
            lines.append(f"### `{async_prefix}{func.name}(...) -> {ret}`\n")
            if ast.get_docstring(func):
                lines.append(f"{ast.get_docstring(func)}\n")
            else:
                lines.append("TODO: Describe this function.\n")

            params = get_function_params(func)
            if params:
                lines.append("**Parameters:**\n")
                for p in params:
                    lines.append(f"- `{p['name']}` (`{p['type']}`): TODO")
                lines.append("")

            lines.append(f"**Returns:** `{ret}` - TODO\n")

    return "\n".join(lines)


def get_function_signature(node) -> str:
    """Get function signature string."""
    import ast

    args = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        type_hint = ""
        if arg.annotation:
            type_hint = f": {ast.unparse(arg.annotation)}"
        args.append(f"{arg.arg}{type_hint}")

    return f"def {node.name}({', '.join(args)})"


def get_function_params(node) -> list:
    """Get function parameters with types and defaults."""
    import ast

    params = []
    defaults = node.args.defaults
    num_defaults = len(defaults)
    num_args = len(node.args.args)

    for i, arg in enumerate(node.args.args):
        if arg.arg in ("self", "cls"):
            continue

        type_hint = "Any"
        if arg.annotation:
            try:
                type_hint = ast.unparse(arg.annotation)
            except Exception:
                type_hint = "Any"

        default = "-"
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0 and default_idx < len(defaults):
            try:
                default = f"`{ast.unparse(defaults[default_idx])}`"
            except Exception:
                default = "..."

        params.append({
            "name": arg.arg,
            "type": type_hint,
            "default": default,
        })

    return params


def get_return_annotation(node) -> str:
    """Get return type annotation."""
    import ast

    if node.returns:
        try:
            return ast.unparse(node.returns)
        except Exception:
            return "Any"
    return "None"


def generate_typescript_stub(file_path: Path, content: str, include_private: bool) -> str:
    """Generate markdown stub from TypeScript/JavaScript code."""
    import re as re_module

    lines = []
    lines.append(f"# {file_path.stem}")
    lines.append(f"\n> **File:** `{file_path}`")
    lines.append("\n## Overview\n")
    lines.append("TODO: Describe what this module does.\n")

    # Find exports using regex
    class_pattern = r'export\s+(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
    func_pattern = r'export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?'
    const_pattern = r'export\s+const\s+(\w+)\s*(?::\s*([^=]+))?\s*='
    interface_pattern = r'export\s+(?:default\s+)?interface\s+(\w+)'
    type_pattern = r'export\s+type\s+(\w+)'

    classes = re_module.findall(class_pattern, content)
    functions = re_module.findall(func_pattern, content)
    consts = re_module.findall(const_pattern, content)
    interfaces = re_module.findall(interface_pattern, content)
    types = re_module.findall(type_pattern, content)

    # Interfaces and Types
    if interfaces or types:
        lines.append("---\n")
        lines.append("## Types\n")
        for iface in interfaces:
            lines.append(f"### `interface {iface}`\n")
            lines.append("TODO: Describe this interface.\n")
        for t in types:
            lines.append(f"### `type {t}`\n")
            lines.append("TODO: Describe this type.\n")

    # Classes
    if classes:
        lines.append("---\n")
        lines.append("## Classes\n")
        for cls_name, extends in classes:
            lines.append(f"### `{cls_name}`")
            if extends:
                lines.append(f" extends `{extends}`")
            lines.append("\n")
            lines.append("TODO: Describe this class.\n")

    # Functions
    if functions:
        lines.append("---\n")
        lines.append("## Functions\n")
        for func_name, params, return_type in functions:
            ret = return_type.strip() if return_type else "void"
            lines.append(f"### `{func_name}({params}) => {ret}`\n")
            lines.append("TODO: Describe this function.\n")

    # Constants
    if consts:
        lines.append("---\n")
        lines.append("## Constants\n")
        lines.append("| Name | Type | Description |")
        lines.append("|------|------|-------------|")
        for const_name, const_type in consts:
            t = const_type.strip() if const_type else "unknown"
            lines.append(f"| `{const_name}` | `{t}` | TODO |")
        lines.append("")

    if len(lines) <= 5:  # Only header
        return ""

    return "\n".join(lines)


@main.command()
@click.argument("docs_path", type=click.Path(exists=True, path_type=Path), default="docs")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--fix", is_flag=True, help="Interactively add frontmatter to files")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def audit(docs_path: Path, path: Path, fix: bool, as_json: bool):
    """Audit docs for ragtime-compatible frontmatter.

    Scans markdown files and suggests metadata for better indexing.

    Examples:
        ragtime audit docs/              # Audit docs folder
        ragtime audit docs/ --fix        # Interactively add frontmatter
        ragtime audit . --json           # Output suggestions as JSON
    """
    import re
    import json as json_module

    path = Path(path).resolve()
    docs_path = Path(docs_path).resolve()

    if not docs_path.exists():
        click.echo(f"✗ Path not found: {docs_path}", err=True)
        return

    # Find all markdown files
    md_files = list(docs_path.rglob("*.md"))

    if not md_files:
        click.echo(f"No markdown files found in {docs_path}")
        return

    results = []

    for md_file in md_files:
        content = md_file.read_text()
        relative = md_file.relative_to(path) if md_file.is_relative_to(path) else md_file

        # Check for existing frontmatter
        has_frontmatter = content.startswith("---")
        existing_meta = {}

        if has_frontmatter:
            try:
                import yaml
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    existing_meta = yaml.safe_load(parts[1]) or {}
            except Exception:
                pass

        # Analyze file for suggestions
        suggestions = analyze_doc_for_metadata(md_file, content, existing_meta)

        status = "ok" if not suggestions["missing"] else "needs_update"
        if not has_frontmatter:
            status = "no_frontmatter"

        results.append({
            "file": str(relative),
            "status": status,
            "has_frontmatter": has_frontmatter,
            "existing": existing_meta,
            "suggestions": suggestions,
        })

    if as_json:
        click.echo(json_module.dumps(results, indent=2))
        return

    # Summary
    no_fm = [r for r in results if r["status"] == "no_frontmatter"]
    needs_update = [r for r in results if r["status"] == "needs_update"]
    ok = [r for r in results if r["status"] == "ok"]

    click.echo(f"\nAudited {len(md_files)} files in {docs_path.name}/\n")

    if ok:
        click.echo(f"✓ {len(ok)} files have complete frontmatter")

    if needs_update:
        click.echo(f"• {len(needs_update)} files could use more metadata")

    if no_fm:
        click.echo(f"✗ {len(no_fm)} files have no frontmatter\n")

        for r in no_fm[:10]:  # Show first 10
            click.echo(f"{'─' * 60}")
            click.echo(f"  {r['file']}")
            s = r["suggestions"]
            click.echo(f"  Suggested frontmatter:")
            click.echo(f"    namespace: {s.get('namespace', 'app')}")
            click.echo(f"    type: {s.get('type', 'document')}")
            if s.get("component"):
                click.echo(f"    component: {s['component']}")

        if len(no_fm) > 10:
            click.echo(f"\n  ... and {len(no_fm) - 10} more files")

    if fix and no_fm:
        click.echo(f"\n{'─' * 60}")
        if click.confirm(f"\nAdd frontmatter to {len(no_fm)} files?"):
            added = 0
            for r in no_fm:
                file_path = path / r["file"]
                content = file_path.read_text()
                s = r["suggestions"]

                # Build frontmatter
                fm_lines = ["---"]
                fm_lines.append(f"namespace: {s.get('namespace', 'app')}")
                fm_lines.append(f"type: {s.get('type', 'document')}")
                if s.get("component"):
                    fm_lines.append(f"component: {s['component']}")
                fm_lines.append("---")
                fm_lines.append("")

                new_content = "\n".join(fm_lines) + content
                file_path.write_text(new_content)
                added += 1
                click.echo(f"  ✓ {r['file']}")

            click.echo(f"\n✓ Added frontmatter to {added} files")
            click.echo(f"  Run 'ragtime reindex' to update the search index")


def analyze_doc_for_metadata(file_path: Path, content: str, existing: dict) -> dict:
    """Analyze a document and suggest metadata."""
    import re

    suggestions = {}
    missing = []

    # Infer from path
    parts = file_path.parts
    path_lower = str(file_path).lower()

    # Namespace inference
    if "namespace" not in existing:
        missing.append("namespace")
        if ".ragtime" in path_lower or "memory" in path_lower:
            suggestions["namespace"] = "app"
        elif "team" in path_lower or "convention" in path_lower:
            suggestions["namespace"] = "team"
        else:
            suggestions["namespace"] = "app"

    # Type inference
    if "type" not in existing:
        missing.append("type")

        # Check content for clues
        content_lower = content.lower()

        if "api" in path_lower or "endpoint" in content_lower:
            suggestions["type"] = "architecture"
        elif "decision" in path_lower or "adr" in path_lower or "we decided" in content_lower:
            suggestions["type"] = "decision"
        elif "guide" in path_lower or "how to" in content_lower:
            suggestions["type"] = "pattern"
        elif "setup" in path_lower or "install" in path_lower:
            suggestions["type"] = "convention"
        elif "readme" in path_lower:
            suggestions["type"] = "document"
        elif "changelog" in path_lower or "release" in path_lower:
            suggestions["type"] = "document"
        else:
            suggestions["type"] = "document"

    # Component inference from path
    if "component" not in existing:
        # Look for component-like folder names
        component_candidates = []
        skip = {"docs", "src", "lib", "app", "pages", "components", "memory", ".ragtime"}

        for part in parts[:-1]:  # Exclude filename
            if part.lower() not in skip and not part.startswith("."):
                component_candidates.append(part.lower())

        if component_candidates:
            suggestions["component"] = component_candidates[-1]  # Most specific
            missing.append("component")

    suggestions["missing"] = missing
    return suggestions


@main.command()
@click.option("--check", is_flag=True, help="Only check for updates, don't install")
def update(check: bool):
    """Check for and install ragtime updates."""
    import json
    from urllib.request import urlopen
    from urllib.error import URLError

    current = "0.2.8"

    click.echo(f"Current version: {current}")
    click.echo("Checking PyPI for updates...")

    try:
        with urlopen("https://pypi.org/pypi/ragtime-cli/json", timeout=10) as resp:
            data = json.loads(resp.read().decode())
            latest = data["info"]["version"]
    except (URLError, json.JSONDecodeError, KeyError) as e:
        click.echo(f"✗ Could not check for updates: {e}", err=True)
        return

    # Compare versions
    def parse_version(v):
        return tuple(int(x) for x in v.split("."))

    current_v = parse_version(current)
    latest_v = parse_version(latest)

    if latest_v > current_v:
        click.echo(f"✓ New version available: {latest}")

        if check:
            click.echo(f"\nUpdate with: pip install --upgrade ragtime-cli")
            return

        if click.confirm(f"\nInstall {latest}?", default=True):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "ragtime-cli"],
                capture_output=False,
            )
            if result.returncode == 0:
                click.echo(f"\n✓ Updated to {latest}")
                click.echo("  Restart your shell to use the new version")
            else:
                click.echo(f"\n✗ Update failed", err=True)
    elif latest_v < current_v:
        click.echo(f"✓ You're ahead of PyPI ({current} > {latest})")
    else:
        click.echo(f"✓ You're on the latest version ({current})")


if __name__ == "__main__":
    main()
