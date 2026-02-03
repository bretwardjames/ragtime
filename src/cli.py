"""
Ragtime CLI - semantic search and memory storage.
"""

from fnmatch import fnmatch
from pathlib import Path
import json
import subprocess
import click
import os
import signal
import sys

from .db import RagtimeDB
from .config import RagtimeConfig, init_config
from .indexers import (
    discover_docs, index_doc_file, DocEntry, index_doc_content,
    discover_code_files, index_code_file, CodeEntry, index_code_content,
)
from .memory import Memory, MemoryStore


def get_git_common_dir(path: Path) -> Path:
    """Get the common git directory (handles worktrees).

    For regular repos, returns .git directory.
    For worktrees, returns the main repo's .git directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            common_dir = result.stdout.strip()
            # Convert to absolute path if relative
            common_path = Path(common_dir)
            if not common_path.is_absolute():
                common_path = (path / common_path).resolve()
            return common_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Fallback to .git in current path
    return path / ".git"


def get_main_repo_path(path: Path) -> Path:
    """Get the main repository path (handles worktrees).

    For regular repos, returns the given path.
    For worktrees, returns the main repo's root directory.
    """
    common_dir = get_git_common_dir(path)
    # The common dir is typically /path/to/repo/.git
    # So the repo root is its parent
    if common_dir.name == ".git":
        return common_dir.parent
    # For bare repos or unusual setups, try to find the toplevel
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=common_dir.parent if common_dir.exists() else path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return path


def list_worktrees(path: Path) -> list[dict]:
    """List all worktrees for a repository.

    Returns list of dicts with 'path', 'branch', 'head' for each worktree.
    Includes the main repo as the first entry.
    """
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        worktrees = []
        current: dict = {}

        for line in result.stdout.strip().split("\n"):
            if line.startswith("worktree "):
                if current:
                    worktrees.append(current)
                current = {"path": line[9:]}
            elif line.startswith("HEAD "):
                current["head"] = line[5:]
            elif line.startswith("branch "):
                # refs/heads/branch-name -> branch-name
                branch = line[7:]
                if branch.startswith("refs/heads/"):
                    branch = branch[11:]
                current["branch"] = branch
            elif line == "detached":
                current["branch"] = None
                current["detached"] = True

        if current:
            worktrees.append(current)

        return worktrees
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def is_worktree(path: Path) -> bool:
    """Check if the given path is a git worktree (not the main repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_dir = result.stdout.strip()
            # Worktrees have .git files pointing elsewhere, not .git directories
            git_path = path / git_dir
            return git_path.is_file() or "worktrees" in git_dir
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def get_db(project_path: Path) -> RagtimeDB:
    """Get or create database for a project.

    Uses the main repo's .ragtime directory even when in a worktree,
    ensuring all worktrees share the same index.
    """
    main_repo = get_main_repo_path(project_path)
    db_path = main_repo / ".ragtime" / "index"
    return RagtimeDB(db_path)


def get_memory_store(project_path: Path) -> MemoryStore:
    """Get memory store for a project.

    Uses the main repo's .ragtime directory even when in a worktree,
    ensuring all worktrees share the same memories.
    """
    main_repo = get_main_repo_path(project_path)
    db = get_db(project_path)
    return MemoryStore(main_repo, db)


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


def git_show_file(path: Path, ref: str, file_path: str) -> str | None:
    """Read a file's contents from a git ref without checkout."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def git_ls_files(path: Path, ref: str, patterns: list[str] | None = None) -> list[str]:
    """List files in a git ref, optionally filtered by patterns."""
    try:
        cmd = ["git", "ls-tree", "-r", "--name-only", ref]
        result = subprocess.run(
            cmd,
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []

        files = [f for f in result.stdout.strip().split("\n") if f]

        if patterns:
            filtered = []
            for f in files:
                for pattern in patterns:
                    if fnmatch(f, pattern):
                        filtered.append(f)
                        break
            return filtered

        return files
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def get_changed_files_from_main(path: Path) -> set[str]:
    """Get set of files that have changed from origin/main (or origin/master).

    Includes modified, added, and deleted files. Returns absolute paths.
    """
    main_ref = get_main_ref(path)
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", main_ref],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            # Convert to absolute paths
            return {str((path / f).resolve()) for f in files}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return set()




def get_main_ref(path: Path) -> str:
    """Get the appropriate main branch ref (origin/main or origin/master)."""
    try:
        # Check if origin/main exists
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "origin/main"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "origin/main"

        # Fall back to origin/master
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "origin/master"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "origin/master"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Default to origin/main
    return "origin/main"


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


def setup_mcp_server(project_path: Path, force: bool = False) -> bool:
    """Offer to configure MCP server for Claude Code integration.

    Args:
        project_path: The project directory
        force: If True, add MCP server without prompting

    Returns True if MCP was configured, False otherwise.
    """
    mcp_config_path = project_path / ".mcp.json"

    ragtime_config = {
        "command": "ragtime-mcp",
        "args": ["--path", "."]
    }

    if mcp_config_path.exists():
        # Read file once
        try:
            existing = json.loads(mcp_config_path.read_text())
        except (IOError, OSError) as e:
            click.echo(f"\n✗ Could not read .mcp.json: {e}", err=True)
            return False
        except json.JSONDecodeError as e:
            click.echo(f"\n! Warning: .mcp.json contains invalid JSON: {e}", err=True)
            if not (force or click.confirm("? Overwrite with new config?", default=False)):
                return False
            existing = {}

        # Check if ragtime is already configured
        if "mcpServers" in existing and "ragtime" in existing.get("mcpServers", {}):
            click.echo("\n✓ MCP server already configured in .mcp.json")
            return True

        # Add ragtime to existing config
        if force or click.confirm("\n? Add ragtime MCP server to existing .mcp.json?", default=True):
            try:
                if "mcpServers" not in existing:
                    existing["mcpServers"] = {}
                existing["mcpServers"]["ragtime"] = ragtime_config
                mcp_config_path.write_text(json.dumps(existing, indent=2) + "\n")
                click.echo("\n✓ Added ragtime to .mcp.json")
                return True
            except IOError as e:
                click.echo(f"\n✗ Failed to update .mcp.json: {e}", err=True)
                return False
    else:
        # Create new config
        if force or click.confirm("\n? Create .mcp.json to enable Claude Code MCP integration?", default=True):
            mcp_config = {
                "mcpServers": {
                    "ragtime": ragtime_config
                }
            }
            try:
                mcp_config_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
                click.echo("\n✓ Created .mcp.json with ragtime server")
                return True
            except IOError as e:
                click.echo(f"\n✗ Failed to create .mcp.json: {e}", err=True)
                return False

    return False


def setup_mcp_global(force: bool = False) -> bool:
    """Add ragtime MCP server to global Claude settings.

    Args:
        force: If True, add without prompting

    Returns True if configured, False otherwise.
    """
    claude_dir = Path.home() / ".claude"
    settings_path = claude_dir / "settings.json"

    ragtime_config = {
        "command": "ragtime-mcp",
        "args": ["--path", "."]
    }

    # Ensure ~/.claude exists
    try:
        claude_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        click.echo(f"✗ Failed to create {claude_dir}: {e}", err=True)
        return False

    if settings_path.exists():
        # Read file once
        try:
            existing = json.loads(settings_path.read_text())
        except (IOError, OSError) as e:
            click.echo(f"✗ Could not read ~/.claude/settings.json: {e}", err=True)
            return False
        except json.JSONDecodeError as e:
            click.echo(f"! Warning: ~/.claude/settings.json contains invalid JSON: {e}", err=True)
            if not (force or click.confirm("? Overwrite with new config?", default=False)):
                return False
            existing = {}

        # Check if ragtime is already configured
        if "mcpServers" in existing and "ragtime" in existing.get("mcpServers", {}):
            click.echo("✓ MCP server already configured in ~/.claude/settings.json")
            return True

        # Add ragtime to existing config
        if force or click.confirm("? Add ragtime MCP server to ~/.claude/settings.json?", default=True):
            try:
                if "mcpServers" not in existing:
                    existing["mcpServers"] = {}
                existing["mcpServers"]["ragtime"] = ragtime_config
                settings_path.write_text(json.dumps(existing, indent=2) + "\n")
                click.echo("✓ Added ragtime to ~/.claude/settings.json")
                return True
            except IOError as e:
                click.echo(f"✗ Failed to update settings: {e}", err=True)
                return False
    else:
        # Create new settings file
        if force or click.confirm("? Create ~/.claude/settings.json with ragtime MCP server?", default=True):
            settings = {
                "mcpServers": {
                    "ragtime": ragtime_config
                }
            }
            try:
                settings_path.write_text(json.dumps(settings, indent=2) + "\n")
                click.echo("✓ Created ~/.claude/settings.json with ragtime server")
                return True
            except IOError as e:
                click.echo(f"✗ Failed to create settings: {e}", err=True)
                return False

    return False


def get_version():
    """Get version from package metadata."""
    try:
        from importlib.metadata import version
        return version("ragtime-cli")
    except Exception:
        return "unknown"


@click.group()
@click.version_option(version=get_version())
@click.option("-y", "--force-defaults", is_flag=True, help="Accept all defaults without prompting")
@click.pass_context
def main(ctx, force_defaults: bool):
    """Ragtime - semantic search over code and documentation."""
    ctx.ensure_object(dict)
    ctx.obj["force_defaults"] = force_defaults


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path), default=".", required=False)
@click.option("-G", "--global", "global_install", is_flag=True, help="Add MCP server to ~/.claude/settings.json")
@click.pass_context
def init(ctx, path: Path, global_install: bool):
    """Initialize ragtime config for a project, or globally with -G."""
    force = (ctx.obj or {}).get("force_defaults", False)

    # Global install: just add MCP to ~/.claude/settings.json
    if global_install:
        setup_mcp_global(force=force)
        return

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

    # Offer MCP server setup for Claude Code integration
    setup_mcp_server(path, force=(ctx.obj or {}).get("force_defaults", False))


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
                # Include chunk_index for hierarchical doc chunks
                ids = [f"{e.file_path}:{e.chunk_index}" for e in batch]

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
    """Index local changes as ephemeral.

    Only indexes files that differ from origin/main. These are tagged as
    "ephemeral" with the current branch name.

    Use 'ragtime sync' to index from origin/main as "permanent" entries.
    Together they form two logical databases:
    - Permanent: from origin/main (convention checks use this)
    - Ephemeral: your local changes (search sees both)
    """
    path = path.resolve()
    db = get_db(path)
    config = RagtimeConfig.load(path)

    # Get current branch for tagging ephemeral entries
    current_branch = get_current_branch(path)

    # Get files changed from main - only these get indexed as ephemeral
    changed_files = get_changed_files_from_main(path)
    if not changed_files:
        click.echo("No files changed from main - nothing to index as ephemeral")
        click.echo("Run 'ragtime sync' to index permanent entries from main")
        return

    click.echo(f"Indexing {len(changed_files)} changed files as ephemeral")
    if current_branch:
        click.echo(f"  Branch: {current_branch}")

    # Clear existing ephemeral entries for this branch
    if current_branch:
        cleared = db.delete_by_branch(current_branch)
        if cleared:
            click.echo(f"  Cleared {cleared} old ephemeral entries")

    # Filter to only changed files
    changed_file_paths = []
    for f in changed_files:
        fp = Path(f)
        if fp.exists():
            changed_file_paths.append(fp)

    # Index docs from changed files
    if index_type in ("all", "docs"):
        doc_files = [f for f in changed_file_paths if f.suffix in (".md", ".txt")]
        if doc_files:
            entries = []
            for file_path in doc_files:
                file_entries = index_doc_file(file_path)
                for entry in file_entries:
                    entry.status = "ephemeral"
                    entry.branch = current_branch
                entries.extend(file_entries)

            if entries:
                _upsert_entries(db, entries, "docs")
                click.echo(f"  Indexed {len(entries)} doc sections (ephemeral)")

    # Index code from changed files
    if index_type in ("all", "code"):
        code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".dart"}
        code_files = [f for f in changed_file_paths if f.suffix in code_extensions]
        if code_files:
            entries = []
            by_type: dict[str, int] = {}
            for file_path in code_files:
                file_entries = index_code_file(file_path)
                for entry in file_entries:
                    entry.status = "ephemeral"
                    entry.branch = current_branch
                    entries.append(entry)
                    by_type[entry.symbol_type] = by_type.get(entry.symbol_type, 0) + 1

            if entries:
                _upsert_entries(db, entries, "code")
                click.echo(f"  Indexed {len(entries)} code symbols (ephemeral)")
                breakdown = ", ".join(f"{count} {typ}s" for typ, count in sorted(by_type.items()))
                click.echo(f"    ({breakdown})")

    stats = db.stats()
    ephemeral = db.get_ephemeral_stats()
    permanent_count = stats['total'] - ephemeral['total']
    click.echo(f"\nIndex stats: {stats['total']} total ({stats['docs']} docs, {stats['code']} code)")
    click.echo(f"  Permanent: {permanent_count}, Ephemeral: {ephemeral['total']}")
    if ephemeral['by_branch']:
        branches = ", ".join(f"{b}: {c}" for b, c in ephemeral['by_branch'].items())
        click.echo(f"  Ephemeral by branch: {branches}")


@main.command()
@click.argument("query")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--type", "type_filter", type=click.Choice(["all", "docs", "code"]), default="all")
@click.option("--namespace", "-n", help="Filter by namespace")
@click.option("--require", "-r", "require_terms", multiple=True,
              help="Additional terms that MUST appear (usually auto-detected)")
@click.option("--raw", is_flag=True, help="Disable auto-detection of qualifiers")
@click.option("--include-archive", is_flag=True, help="Also search archived branches")
@click.option("--limit", "-l", default=5, help="Max results")
@click.option("--verbose", "-v", is_flag=True, help="Show full content")
def search(query: str, path: Path, type_filter: str, namespace: str,
           require_terms: tuple, raw: bool, include_archive: bool, limit: int, verbose: bool):
    """
    Smart search: auto-detects qualifiers like 'mobile', 'auth', 'dart'.

    \b
    Examples:
      ragtime search "error handling in mobile"  # auto-requires 'mobile'
      ragtime search "auth flow"                 # auto-requires 'auth'
      ragtime search "useAsyncState" --raw       # literal search, no extraction
    """
    path = Path(path).resolve()
    db = get_db(path)

    type_arg = None if type_filter == "all" else type_filter

    results = db.search(
        query=query,
        limit=limit,
        type_filter=type_arg,
        namespace=namespace,
        require_terms=list(require_terms) if require_terms else None,
        auto_extract=not raw,
    )

    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        score = 1 - distance if distance else None

        click.echo(f"\n{'─' * 60}")

        # Build location string with line number for code
        file_path = meta.get('file', 'unknown')
        line_num = meta.get('line')
        if line_num:
            location = f"{file_path}:{line_num}"
        else:
            location = file_path

        click.echo(f"[{i}] {location}")

        # Show symbol info for code, section info for docs
        result_type = meta.get('type')
        if result_type == "code":
            symbol = meta.get('symbol_name', '')
            symbol_type = meta.get('symbol_type', '')
            status = meta.get('status', '')
            info_parts = [f"Type: {result_type}"]
            if symbol:
                info_parts.append(f"{symbol_type}: {symbol}")
            if status:
                info_parts.append(f"Status: {status}")
            click.echo(f"    {' | '.join(info_parts)}")
        else:
            section = meta.get('section_path', '')
            result_namespace = meta.get('namespace', '-')
            status = meta.get('status', '')
            info_parts = [f"Type: {result_type}", f"Namespace: {result_namespace}"]
            if section:
                info_parts.append(f"Section: {section}")
            if status:
                info_parts.append(f"Status: {status}")
            click.echo(f"    {' | '.join(info_parts)}")

        if score:
            click.echo(f"    Score: {score:.3f}")

        if verbose:
            click.echo(f"\n{result['content']}")
        else:
            preview = result["content"][:150].replace("\n", " ")
            click.echo(f"    {preview}...")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
def stats(path: Path):
    """Show index statistics including worktree and ephemeral/permanent breakdown."""
    path = Path(path).resolve()
    db = get_db(path)
    main_repo = get_main_repo_path(path)

    # Worktree info
    if is_worktree(path):
        click.echo(f"Worktree: {path}")
        click.echo(f"Main repo: {main_repo}")
    else:
        click.echo(f"Repository: {path}")

    worktrees = list_worktrees(path)
    if len(worktrees) > 1:
        click.echo(f"Worktrees: {len(worktrees)}")
        for wt in worktrees:
            branch = wt.get("branch", "detached")
            click.echo(f"  • {wt['path']} ({branch})")

    click.echo("")

    # Index stats
    s = db.stats()
    ephemeral = db.get_ephemeral_stats()
    permanent_count = s['total'] - ephemeral['total']

    click.echo(f"Total indexed: {s['total']}")
    click.echo(f"  Docs: {s['docs']}")
    click.echo(f"  Code: {s['code']}")
    click.echo(f"\nPermanent: {permanent_count}")
    click.echo(f"Ephemeral: {ephemeral['total']}")

    if ephemeral['by_branch']:
        click.echo("  By branch:")
        for branch, count in ephemeral['by_branch'].items():
            click.echo(f"    • {branch}: {count}")


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
@click.argument("memory_id", required=False)
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--list", "list_candidates", is_flag=True, help="List graduation candidates")
@click.option("--branch", "-b", help="Branch name or slug (default: current branch)")
@click.option("--confidence", default="high",
              type=click.Choice(["high", "medium", "low"]),
              help="Confidence level for graduated memory")
def graduate(memory_id: str, path: Path, list_candidates: bool, branch: str, confidence: str):
    """Graduate a branch memory to app namespace.

    With --list: Show branch memories that are candidates for graduation.
    With MEMORY_ID: Graduate a specific memory to app namespace.

    Examples:
        ragtime graduate --list              # List candidates for current branch
        ragtime graduate --list -b feature/auth  # List candidates for specific branch
        ragtime graduate abc123              # Graduate specific memory
    """
    path = Path(path).resolve()
    store = get_memory_store(path)

    # List mode: show graduation candidates
    if list_candidates or not memory_id:
        # Determine branch
        if not branch:
            branch = get_current_branch(path)
            if not branch:
                click.echo("✗ Not in a git repository or no branch specified", err=True)
                return

        # Create namespace pattern (handle both original and slugified)
        branch_slug = branch.replace("/", "-")
        namespace = f"branch-{branch_slug}"

        # Get memories for this branch
        memories = store.list_memories(namespace=namespace, status="active")

        if not memories:
            click.echo(f"No active memories found for branch: {branch}")
            click.echo(f"  (namespace: {namespace})")
            return

        # Filter to graduation candidates (exclude context type)
        candidates = [m for m in memories if m.type != "context"]

        if not candidates:
            click.echo(f"No graduation candidates for branch: {branch}")
            click.echo(f"  (found {len(memories)} memories, but all are context type)")
            return

        click.echo(f"\nGraduation candidates for branch: {branch}")
        click.echo(f"{'─' * 50}")

        for i, mem in enumerate(candidates, 1):
            type_badge = f"[{mem.type}]" if mem.type else "[unknown]"
            confidence_badge = f"({mem.confidence})" if hasattr(mem, 'confidence') and mem.confidence else ""

            click.echo(f"\n  {i}. {type_badge} {confidence_badge}")
            click.echo(f"     ID: {mem.id}")

            # Show preview
            preview = mem.content[:150].replace("\n", " ").strip()
            if len(mem.content) > 150:
                preview += "..."
            click.echo(f"     {preview}")

            if hasattr(mem, 'added') and mem.added:
                click.echo(f"     Added: {mem.added}")

        click.echo(f"\n{'─' * 50}")
        click.echo(f"{len(candidates)} candidate(s) found.")
        click.echo(f"\nTo graduate: ragtime graduate <ID>")
        return

    # Graduate mode: graduate specific memory
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


@main.command("check-conventions")
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--files", "-f", help="Files to check (comma-separated or JSON array)")
@click.option("--branch", "-b", help="Branch to diff against main (default: current branch)")
@click.option("--event-file", type=click.Path(exists=True, path_type=Path), help="GHP event file (JSON)")
@click.option("--include-memories", is_flag=True, default=True, help="Include convention memories")
@click.option("--all", "return_all", is_flag=True, help="Return ALL conventions (for AI workflows)")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON (for MCP/hooks)")
def check_conventions(path: Path, files: str, branch: str, event_file: Path,
                      include_memories: bool, return_all: bool, json_output: bool):
    """Show conventions applicable to changed files.

    Used by pre-PR hooks to check code against team conventions.

    IMPORTANT: Conventions are read from origin/main (not working tree) to prevent
    PRs from adding conventions that match their own code. Only merged conventions
    are enforced.

    By default, returns only conventions RELEVANT to the changed files (semantic search).
    Use --all to return ALL conventions (useful for AI workflows that analyze edge cases).

    Examples:
        ragtime check-conventions                    # Relevant conventions only
        ragtime check-conventions --all             # ALL conventions (for AI)
        ragtime check-conventions -f "src/auth.ts"  # Conventions relevant to specific file
        ragtime check-conventions --event-file /tmp/ghp-event.json --all  # From GHP hook
    """
    import json as json_module

    path = Path(path).resolve()
    config = RagtimeConfig.load(path)

    # Load from event file if provided (GHP hook pattern)
    if event_file:
        try:
            event_data = json_module.loads(Path(event_file).read_text())
            # Event file can provide changed_files and branch
            if not files and "changed_files" in event_data:
                files = event_data["changed_files"]
            if not branch and "branch" in event_data:
                branch = event_data["branch"]
        except (json_module.JSONDecodeError, IOError) as e:
            click.echo(f"⚠ Could not read event file: {e}", err=True)

    # Determine files to check
    changed_files = []
    if files:
        # Handle list (from event file) or string (from CLI)
        if isinstance(files, list):
            changed_files = files
        elif files.startswith("["):
            try:
                changed_files = json_module.loads(files)
            except json_module.JSONDecodeError:
                changed_files = [f.strip() for f in files.split(",")]
        else:
            changed_files = [f.strip() for f in files.split(",")]
    else:
        # Get changed files from git diff
        if not branch:
            branch = get_current_branch(path)
        if branch and branch not in ("main", "master"):
            result = subprocess.run(
                ["git", "diff", "--name-only", "main...HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                changed_files = [f for f in result.stdout.strip().split("\n") if f]

    # Gather conventions from files
    conventions_data = {
        "files": [],
        "memories": [],
        "changed_files": changed_files,
    }

    # Read convention files from origin/main (not working tree) to prevent gaming
    main_ref = get_main_ref(path)
    modified_conventions = []

    for conv_file in config.conventions.files:
        conv_path = path / conv_file
        conv_str = str(conv_file)

        # Try to read from origin/main first
        content = git_show_file(path, main_ref, conv_str)

        if content:
            conventions_data["files"].append({
                "path": conv_str,
                "content": content,
                "source": main_ref,
            })
            # Check if modified locally
            if conv_path.exists():
                local_content = conv_path.read_text()
                if local_content != content:
                    modified_conventions.append(conv_str)
        elif conv_path.exists():
            # Fall back to working tree if not in main (new convention file)
            if conv_path.is_file():
                content = conv_path.read_text()
                conventions_data["files"].append({
                    "path": conv_str,
                    "content": content,
                    "source": "working_tree",
                    "warning": "Not in main - will not be enforced until merged",
                })
            elif conv_path.is_dir():
                for f in conv_path.rglob("*"):
                    if f.is_file() and f.suffix in (".md", ".txt", ".yaml", ".yml"):
                        rel_path = str(f.relative_to(path))
                        content = git_show_file(path, main_ref, rel_path)
                        if content:
                            conventions_data["files"].append({
                                "path": rel_path,
                                "content": content,
                                "source": main_ref,
                            })
                        else:
                            conventions_data["files"].append({
                                "path": rel_path,
                                "content": f.read_text(),
                                "source": "working_tree",
                                "warning": "Not in main",
                            })

    if modified_conventions and not json_output:
        click.echo(f"⚠ Convention files modified locally (using {main_ref} version):")
        for f in modified_conventions:
            click.echo(f"  • {f}")

    # Search memories for conventions
    if include_memories and config.conventions.also_search_memories:
        store = get_memory_store(path)
        db = get_db(path)

        if return_all:
            # Return ALL convention/pattern memories (for AI workflows)
            for ns in ["team", "app"]:
                memories = store.list_memories(namespace=ns, type_filter="convention", status="active")
                for mem in memories:
                    conventions_data["memories"].append({
                        "id": mem.id,
                        "namespace": mem.namespace,
                        "content": mem.content,
                        "component": mem.component,
                    })
                # Also get pattern-type memories
                patterns = store.list_memories(namespace=ns, type_filter="pattern", status="active")
                for mem in patterns:
                    conventions_data["memories"].append({
                        "id": mem.id,
                        "namespace": mem.namespace,
                        "type": "pattern",
                        "content": mem.content,
                        "component": mem.component,
                    })
        else:
            # Semantic search for RELEVANT conventions based on changed files
            if changed_files:
                # Build search query from file paths and names
                # Extract meaningful terms: paths, extensions, inferred components
                search_terms = set()
                for f in changed_files:
                    parts = Path(f).parts
                    search_terms.update(parts)  # Add path components
                    search_terms.add(Path(f).suffix.lstrip("."))  # Add extension
                    search_terms.add(Path(f).stem)  # Add filename without ext

                # Remove common noise
                noise = {"src", "lib", "test", "tests", "spec", "specs", "", "js", "ts", "py", "md"}
                search_terms = search_terms - noise

                query = f"conventions patterns rules standards for {' '.join(search_terms)}"

                # Search both namespaces using the db directly
                # Only search "permanent" entries to prevent gaming via branch conventions
                seen_ids = set()
                for ns in ["team", "app"]:
                    results = db.search(query, namespace=ns, status="permanent", limit=50)
                    for result in results:
                        meta = result.get("metadata", {})
                        result_type = meta.get("type", "")
                        result_category = meta.get("category", "")
                        # Include convention/pattern types OR docs with category="convention"
                        is_convention_content = (
                            result_type in ("convention", "pattern") or
                            result_category == "convention"
                        )
                        if is_convention_content:
                            mem_id = meta.get("id") or meta.get("file", "")
                            if mem_id not in seen_ids:
                                seen_ids.add(mem_id)
                                distance = result.get("distance", 1)
                                score = 1 - distance if distance else 0
                                conventions_data["memories"].append({
                                    "id": mem_id,
                                    "namespace": ns,
                                    "type": result_type,
                                    "content": result.get("content", ""),
                                    "component": meta.get("component"),
                                    "relevance": score,
                                })
            else:
                # No changed files - fall back to listing all
                for ns in ["team", "app"]:
                    memories = store.list_memories(namespace=ns, type_filter="convention", status="active")
                    for mem in memories:
                        conventions_data["memories"].append({
                            "id": mem.id,
                            "namespace": mem.namespace,
                            "content": mem.content,
                            "component": mem.component,
                        })

    # Output
    if json_output:
        click.echo(json_module.dumps(conventions_data, indent=2))
        return

    # Human-friendly output
    click.echo(f"\nConventions Check")
    click.echo(f"{'═' * 50}")

    if changed_files:
        click.echo(f"\nFiles to check ({len(changed_files)}):")
        for f in changed_files[:10]:
            click.echo(f"  • {f}")
        if len(changed_files) > 10:
            click.echo(f"  ... and {len(changed_files) - 10} more")

    click.echo(f"\n{'─' * 50}")
    click.echo(f"Convention Files ({len(conventions_data['files'])}):")

    if not conventions_data["files"]:
        click.echo("  No convention files found.")
        click.echo(f"  Configure in .ragtime/config.yaml under 'conventions.files'")
    else:
        for conv in conventions_data["files"]:
            click.echo(f"\n  📄 {conv['path']}")
            # Show first few lines as preview
            lines = conv["content"].strip().split("\n")[:5]
            for line in lines:
                if line.strip():
                    click.echo(f"     {line[:70]}{'...' if len(line) > 70 else ''}")

    if conventions_data["memories"]:
        click.echo(f"\n{'─' * 50}")
        mode_label = "All" if return_all else "Relevant"
        click.echo(f"Convention Memories - {mode_label} ({len(conventions_data['memories'])}):")

        for mem in conventions_data["memories"]:
            type_str = mem.get("type", "convention")
            relevance = mem.get("relevance")
            relevance_str = f" (relevance: {relevance:.2f})" if relevance else ""
            click.echo(f"\n  [{mem['id'][:8] if mem['id'] else '?'}] {mem['namespace']} / {type_str}{relevance_str}")
            if mem.get("component"):
                click.echo(f"     Component: {mem['component']}")
            preview = mem["content"][:100].replace("\n", " ")
            click.echo(f"     {preview}...")

    click.echo(f"\n{'═' * 50}")
    total = len(conventions_data["files"]) + len(conventions_data["memories"])
    mode_note = " (all)" if return_all else " (filtered by relevance)"
    click.echo(f"Total: {total} convention source(s){mode_note}")


@main.command("add-convention")
@click.argument("content", required=False)
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--component", "-c", help="Component this convention applies to")
@click.option("--to-file", type=click.Path(path_type=Path), help="Add to specific file")
@click.option("--to-memory", is_flag=True, help="Store as memory only (not in git)")
@click.option("--quiet", "-q", is_flag=True, help="Non-interactive mode")
def add_convention(content: str | None, path: Path, component: str | None,
                   to_file: Path | None, to_memory: bool, quiet: bool):
    """Add a new convention with smart storage routing.

    Finds the best place to store the convention:
    - Existing doc with ## Conventions section (if component matches)
    - Convention folder (.ragtime/conventions/)
    - Central conventions file (.ragtime/CONVENTIONS.md)
    - Memory only (searchable but not committed)

    Examples:
        ragtime add-convention "Always use async/await, never .then()"
        ragtime add-convention --component auth "JWT tokens expire after 15 minutes"
        ragtime add-convention --to-file docs/api.md "Use snake_case for JSON"
        ragtime add-convention --to-memory "Prefer composition over inheritance"
    """
    path = Path(path).resolve()
    config = RagtimeConfig.load(path)

    # Get content interactively if not provided
    if not content:
        content = click.prompt("Convention to add")

    if not content.strip():
        click.echo("✗ No content provided", err=True)
        return

    # If explicit destination provided, use it
    if to_memory:
        _store_convention_as_memory(path, content, component)
        return

    if to_file:
        _append_convention_to_file(path, to_file, content)
        return

    # Auto-routing based on config
    storage_mode = config.conventions.storage

    if storage_mode == "memory":
        _store_convention_as_memory(path, content, component)
        return

    # Find storage options
    options = _find_convention_storage_options(path, config, component)

    if storage_mode == "file" or (storage_mode == "auto" and options):
        # Use first available file option, or default
        if options:
            target = options[0]
        else:
            target = {
                "type": "default_file",
                "path": config.conventions.default_file,
                "description": "Central conventions file",
            }
        _store_convention_to_target(path, target, content, component)
        return

    if storage_mode == "ask" or (storage_mode == "auto" and not quiet):
        # Present options to user
        click.echo("\nWhere should this convention be stored?\n")

        choices = []
        for i, opt in enumerate(options, 1):
            icon = "📄" if opt["type"] == "existing_section" else "📁" if opt["type"] == "folder" else "📋"
            click.echo(f"  {i}. {icon} {opt['description']}")
            click.echo(f"       {opt['path']}")
            choices.append(opt)

        # Add default options if not already present
        default_file_present = any(o["path"] == config.conventions.default_file for o in choices)
        if not default_file_present:
            i = len(choices) + 1
            click.echo(f"  {i}. 📋 Central conventions file")
            click.echo(f"       {config.conventions.default_file}")
            choices.append({
                "type": "default_file",
                "path": config.conventions.default_file,
                "description": "Central conventions file",
            })

        i = len(choices) + 1
        click.echo(f"  {i}. 🧠 Memory only (not committed)")
        click.echo(f"       Stored in team namespace, searchable but not in git")
        choices.append({"type": "memory", "path": None, "description": "Memory only"})

        click.echo()
        choice = click.prompt("Choice", type=int, default=1)

        if choice < 1 or choice > len(choices):
            click.echo("✗ Invalid choice", err=True)
            return

        target = choices[choice - 1]
        if target["type"] == "memory":
            _store_convention_as_memory(path, content, component)
        else:
            _store_convention_to_target(path, target, content, component)
        return

    # Fallback: memory only
    _store_convention_as_memory(path, content, component)


def _find_convention_storage_options(path: Path, config: RagtimeConfig, component: str | None) -> list[dict]:
    """Find potential storage locations for a convention."""
    import re
    options = []

    # 1. Check for existing docs with ## Conventions sections
    for scan_path in config.conventions.scan_docs_for_sections:
        scan_dir = path / scan_path
        if scan_dir.exists() and scan_dir.is_dir():
            for md_file in scan_dir.rglob("*.md"):
                content = md_file.read_text()
                # Look for ## Conventions header
                if re.search(r'^##\s+(Conventions?|Rules|Standards|Guidelines)', content, re.MULTILINE | re.IGNORECASE):
                    # If component specified, check if file relates to it
                    file_component = md_file.stem.lower()
                    rel_path = md_file.relative_to(path)

                    if component:
                        if component.lower() in str(rel_path).lower():
                            options.insert(0, {  # Prioritize component match
                                "type": "existing_section",
                                "path": str(rel_path),
                                "description": f"Add to existing Conventions section ({file_component})",
                            })
                    else:
                        options.append({
                            "type": "existing_section",
                            "path": str(rel_path),
                            "description": f"Add to existing Conventions section",
                        })

    # 2. Check if convention folder exists
    folder = path / config.conventions.folder
    if folder.exists() and folder.is_dir():
        if component:
            target_file = f"{component.lower()}.md"
            options.append({
                "type": "folder",
                "path": f"{config.conventions.folder}{target_file}",
                "description": f"Create/append to {target_file} in conventions folder",
            })
        else:
            options.append({
                "type": "folder",
                "path": f"{config.conventions.folder}general.md",
                "description": "Add to general.md in conventions folder",
            })

    # 3. Check if default conventions file exists
    default_file = path / config.conventions.default_file
    if default_file.exists():
        options.append({
            "type": "default_file",
            "path": config.conventions.default_file,
            "description": "Append to central conventions file",
        })

    return options


def _store_convention_as_memory(path: Path, content: str, component: str | None):
    """Store convention as a memory in team namespace."""
    from datetime import date
    store = get_memory_store(path)

    memory = Memory(
        content=content,
        namespace="team",
        type="convention",
        component=component,
        confidence="high",
        confidence_reason="user-added",
        source="add-convention",
        status="active",
        added=date.today().isoformat(),
        author=get_author(),
    )

    file_path = store.save(memory)
    click.echo(f"✓ Convention stored as memory")
    click.echo(f"  ID: {memory.id}")
    click.echo(f"  File: {file_path.relative_to(path)}")
    click.echo(f"  Namespace: team")


def _store_convention_to_target(path: Path, target: dict, content: str, component: str | None):
    """Store convention to a file target."""
    import re

    target_path = path / target["path"]
    target_type = target["type"]

    if target_type == "existing_section":
        # Append to existing ## Conventions section
        if not target_path.exists():
            click.echo(f"✗ File not found: {target['path']}", err=True)
            return

        file_content = target_path.read_text()

        # Find the Conventions section and append
        pattern = r'(^##\s+(?:Conventions?|Rules|Standards|Guidelines).*?)(\n##|\Z)'
        match = re.search(pattern, file_content, re.MULTILINE | re.IGNORECASE | re.DOTALL)

        if match:
            section_end = match.end(1)
            new_content = file_content[:section_end] + f"\n\n- {content}" + file_content[section_end:]
            target_path.write_text(new_content)
            click.echo(f"✓ Convention added to {target['path']}")
            click.echo(f"  Added to ## Conventions section")
        else:
            click.echo(f"✗ Could not find Conventions section in {target['path']}", err=True)

    elif target_type == "folder":
        # Create or append to file in conventions folder
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            # Append to existing file
            existing = target_path.read_text()
            new_content = existing.rstrip() + f"\n\n- {content}\n"
        else:
            # Create new file
            title = target_path.stem.replace("-", " ").replace("_", " ").title()
            new_content = f"""---
namespace: team
type: convention
component: {component or ''}
---

# {title} Conventions

- {content}
"""
        target_path.write_text(new_content)
        click.echo(f"✓ Convention added to {target['path']}")

    elif target_type == "default_file":
        # Create or append to default conventions file
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            existing = target_path.read_text()
            # Check if there's a component section
            if component:
                section_pattern = rf'^##\s+{re.escape(component)}.*?(?=\n##|\Z)'
                match = re.search(section_pattern, existing, re.MULTILINE | re.IGNORECASE | re.DOTALL)
                if match:
                    # Append to component section
                    section_end = match.end()
                    new_content = existing[:section_end] + f"\n- {content}" + existing[section_end:]
                else:
                    # Create new component section
                    new_content = existing.rstrip() + f"\n\n## {component.title()}\n\n- {content}\n"
            else:
                # Append to general section or end
                new_content = existing.rstrip() + f"\n\n- {content}\n"
        else:
            # Create new file
            header = f"## {component.title()}\n\n" if component else ""
            new_content = f"""# Team Conventions

{header}- {content}
"""
        target_path.write_text(new_content)
        click.echo(f"✓ Convention added to {target['path']}")

    # Reindex the file
    _reindex_convention_file(path, target_path)


def _append_convention_to_file(path: Path, to_file: Path, content: str):
    """Append a convention directly to a user-specified file."""
    import re

    target_path = path / to_file if not to_file.is_absolute() else to_file

    if not target_path.exists():
        click.echo(f"✗ File not found: {to_file}", err=True)
        return

    file_content = target_path.read_text()

    # Try to find a Conventions section to append to
    pattern = r'(^##\s+(?:Conventions?|Rules|Standards|Guidelines).*?)(\n##|\Z)'
    match = re.search(pattern, file_content, re.MULTILINE | re.IGNORECASE | re.DOTALL)

    if match:
        # Append to existing section
        section_end = match.end(1)
        new_content = file_content[:section_end] + f"\n\n- {content}" + file_content[section_end:]
        target_path.write_text(new_content)
        click.echo(f"✓ Convention added to {to_file}")
        click.echo(f"  Added to ## Conventions section")
    else:
        # Append at end of file
        new_content = file_content.rstrip() + f"\n\n- {content}\n"
        target_path.write_text(new_content)
        click.echo(f"✓ Convention appended to {to_file}")

    # Reindex the file
    _reindex_convention_file(path, target_path)


def _reindex_convention_file(path: Path, target_path: Path):
    """Reindex a convention file after modification."""
    try:
        db = get_db(path)
        from .indexers.docs import index_file
        entries = index_file(target_path)
        for entry in entries:
            db.upsert(
                ids=[f"{entry.file_path}:{entry.chunk_index}"],
                documents=[entry.content],
                metadatas=[entry.to_metadata()],
            )
        click.echo(f"  Indexed {len(entries)} section(s)")
    except Exception as e:
        click.echo(f"  ⚠ Could not reindex: {e}")


@main.command()
@click.option("--path", type=click.Path(exists=True, path_type=Path), default=".")
@click.option("--dry-run", is_flag=True, help="Show what would be removed")
def dedupe(path: Path, dry_run: bool):
    """Clean up index: remove duplicates and orphaned entries.

    - Removes duplicate entries (keeps one per file path)
    - Removes orphaned entries (files that no longer exist on disk)
    """
    path = Path(path).resolve()
    db = get_db(path)
    memory_dir = path / ".ragtime"

    # Get all entries with their file paths
    results = db.collection.get(include=["metadatas"])

    # Group by file path and track orphans
    by_file: dict[str, list[str]] = {}
    orphans: list[str] = []

    for i, mem_id in enumerate(results["ids"]):
        file_path = results["metadatas"][i].get("file", "")
        entry_type = results["metadatas"][i].get("type", "")

        # Skip docs/code entries - only clean up memory entries
        if entry_type in ("docs", "code"):
            continue

        if not file_path:
            orphans.append(mem_id)
            continue

        # Check if file exists on disk
        full_path = memory_dir / file_path
        if not full_path.exists():
            orphans.append(mem_id)
            if dry_run:
                click.echo(f"  Orphan: {file_path} (file missing)")
            continue

        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(mem_id)

    # Find duplicates (keep first, remove rest)
    duplicates: list[str] = []
    for file_path, ids in by_file.items():
        if len(ids) > 1:
            duplicates.extend(ids[1:])
            if dry_run:
                click.echo(f"  Duplicate: {file_path} ({len(ids)} copies, removing {len(ids) - 1})")

    to_remove = orphans + duplicates

    if not to_remove:
        click.echo("✓ Index is clean (no duplicates or orphans)")
        return

    if dry_run:
        click.echo(f"\nWould remove {len(orphans)} orphans + {len(duplicates)} duplicates = {len(to_remove)} entries")
        click.echo("Run without --dry-run to remove them")
    else:
        db.delete(to_remove)
        click.echo(f"✓ Removed {len(orphans)} orphans + {len(duplicates)} duplicates = {len(to_remove)} entries")


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

    title = issue_data.get("title") or f"Issue #{issue}"
    body = issue_data.get("body") or ""  # Handle null and empty
    labels = issue_data.get("labels") or []

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
# Usage Documentation
# ============================================================================


@main.command("usage")
@click.option("--section", "-s", help="Show specific section (mcp, cli, workflows, conventions)")
def usage(section: str | None):
    """Show how to use ragtime effectively with AI agents.

    Prints documentation on integrating ragtime into your AI workflow,
    whether using the MCP server or CLI directly.
    """
    sections = {
        "mcp": USAGE_MCP,
        "cli": USAGE_CLI,
        "workflows": USAGE_WORKFLOWS,
        "conventions": USAGE_CONVENTIONS,
    }

    if section:
        if section.lower() in sections:
            click.echo(sections[section.lower()])
        else:
            click.echo(f"Unknown section: {section}")
            click.echo(f"Available: {', '.join(sections.keys())}")
        return

    # Print all sections
    click.echo(USAGE_HEADER)
    for content in sections.values():
        click.echo(content)
        click.echo()


USAGE_HEADER = """
# Ragtime Usage Guide

Ragtime provides persistent memory for AI coding sessions. Use it via:
- **MCP Server** (recommended for Claude Code / AI tools)
- **CLI** (for scripts, hooks, and manual use)
"""

USAGE_MCP = """
## MCP Server Tools

After `ragtime init`, add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "ragtime": {
      "command": "ragtime",
      "args": ["serve"]
    }
  }
}
```

### Core Tools

| Tool | Purpose |
|------|---------|
| `mcp__ragtime__search` | Find relevant memories, docs, and code |
| `mcp__ragtime__remember` | Store new knowledge |
| `mcp__ragtime__list_memories` | Browse stored memories |
| `mcp__ragtime__forget` | Delete a memory |
| `mcp__ragtime__graduate` | Promote branch memory to app namespace |

### Search Examples

```
# Find architecture knowledge
mcp__ragtime__search(query="authentication flow", namespace="app")

# Find team conventions
mcp__ragtime__search(query="error handling", namespace="team", type="convention")

# Search with auto-qualifier detection
mcp__ragtime__search(query="how does auth work in mobile", tiered=true)
```

### Remember Examples

```
# Store architecture insight
mcp__ragtime__remember(
  content="JWT tokens expire after 15 minutes, refresh tokens after 7 days",
  namespace="app",
  type="architecture",
  component="auth"
)

# Store team convention
mcp__ragtime__remember(
  content="Always use async/await, never .then() chains",
  namespace="team",
  type="convention"
)

# Store branch-specific decision
mcp__ragtime__remember(
  content="Using Redis for session storage because of horizontal scaling needs",
  namespace="branch-feature/auth",
  type="decision"
)
```
"""

USAGE_CLI = """
## CLI Commands

### Memory Management

```bash
# Search memories
ragtime search "authentication patterns"
ragtime search "conventions" --namespace team

# Add a convention
ragtime add-convention "Always validate user input"
ragtime add-convention -c auth "JWT must be validated on every request"

# List memories
ragtime memories --namespace app
ragtime memories --type convention

# Check conventions for changed files
ragtime check-conventions
ragtime check-conventions --all  # For AI (comprehensive)
```

### Branch Context

```bash
# Create branch context from GitHub issue
ragtime new-branch 123 --branch feature/auth

# Graduate branch memories after PR merge
ragtime graduate --branch feature/auth
```

### Indexing

```bash
# Index docs and code
ragtime index
ragtime reindex  # Full reindex
```
"""

USAGE_WORKFLOWS = """
## Recommended Workflows

### Starting Work on an Issue

1. AI reads the issue and existing context:
   ```
   mcp__ragtime__search(query="auth implementation", namespace="app")
   ```

2. Check for branch context:
   ```bash
   # Look for .ragtime/branches/{branch}/context.md
   ```

3. As you make decisions, store them:
   ```
   mcp__ragtime__remember(
     content="Chose PKCE flow for mobile OAuth",
     namespace="branch-feature/oauth",
     type="decision"
   )
   ```

### Before Creating a PR

1. Check conventions:
   ```bash
   ragtime check-conventions
   ```

2. Review branch memories for graduation candidates

### After PR Merge

1. Graduate valuable branch memories to app namespace:
   ```bash
   ragtime graduate --branch feature/auth
   ```

### Session Handoff

Store context in `.ragtime/branches/{branch}/context.md`:
- Current state
- What's left to do
- Key decisions made
- Blockers or notes for next session
"""

USAGE_CONVENTIONS = """
## Convention System

### Storing Conventions

Conventions can live in:
- **Files**: `.ragtime/CONVENTIONS.md` or `docs/conventions/`
- **Doc sections**: Any `## Conventions` section in markdown
- **Memories**: `team` namespace with type `convention`

```bash
# Add via CLI (routes automatically)
ragtime add-convention "Use snake_case for JSON fields"

# Add to specific file
ragtime add-convention --to-file docs/api.md "Return 404 for missing resources"

# Add as memory only
ragtime add-convention --to-memory "Prefer composition over inheritance"
```

### Convention Detection

The indexer automatically detects conventions from:
- Files named `*conventions*`, `*rules*`, `*standards*`
- Sections with headers like `## Conventions`, `## Rules`
- Content between `<!-- convention -->` markers
- Frontmatter: `has_conventions: true` or `convention_sections: [...]`

### Checking Conventions

```bash
# Human use (filtered to relevant)
ragtime check-conventions

# AI use (comprehensive)
ragtime check-conventions --all --json
```
"""


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
    # NOTE: No quotes around template vars - GHP's shellEscape() handles escaping
    hook_command = "ragtime new-branch ${issue.number} --issue-json ${issue.json} --branch ${branch}"

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
@click.option("--no-reindex", is_flag=True, help="Skip reindexing from main")
def sync(path: Path, quiet: bool, auto_prune: bool, no_reindex: bool):
    """Sync memories from all remote branches and reindex from main.

    Fetches .ragtime/branches/* from remote branches and copies to
    local dot-prefixed folders (e.g., .feature-branch/).

    Also reindexes code and docs from origin/main as "permanent" embeddings,
    used for convention checks and as the baseline for search.
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
        click.echo(f"\nSynced {synced} branches.")

    # Reindex from origin/main as permanent
    if not no_reindex:
        _sync_permanent_index(path, quiet)

    if not quiet:
        click.echo("\nDone.")


def _sync_permanent_index(path: Path, quiet: bool):
    """Reindex code and docs from origin/main as permanent embeddings."""
    main_ref = get_main_ref(path)
    db = get_db(path)
    config = RagtimeConfig.load(path)

    if not quiet:
        click.echo(f"\nReindexing from {main_ref}...")

    # Clear existing permanent entries
    cleared = db.clear_permanent()
    if not quiet and cleared > 0:
        click.echo(f"  Cleared {cleared} existing permanent entries")

    # Get file patterns for code
    code_extensions = []
    for lang in config.code.languages:
        if lang == "python":
            code_extensions.append("*.py")
        elif lang == "typescript":
            code_extensions.extend(["*.ts", "*.tsx"])
        elif lang == "javascript":
            code_extensions.extend(["*.js", "*.jsx"])
        elif lang == "vue":
            code_extensions.append("*.vue")
        elif lang == "dart":
            code_extensions.append("*.dart")

    # List all files from main ref
    all_files = git_ls_files(path, main_ref)
    if not all_files:
        if not quiet:
            click.echo("  No files found in main ref")
        return

    # Filter to configured paths
    code_paths = [str(p) for p in config.code.paths]
    doc_paths = [str(p) for p in config.docs.paths]

    code_entries = []
    doc_entries = []

    for file_path in all_files:
        # Skip excluded patterns
        skip = False
        for exclude in config.code.exclude + config.docs.exclude:
            if fnmatch(file_path, exclude):
                skip = True
                break
        if skip:
            continue

        # Check if it's a code file
        is_code = False
        for ext in code_extensions:
            if fnmatch(file_path, f"**/{ext}") or fnmatch(file_path, ext):
                is_code = True
                break

        # Check if in configured paths (handle "." as meaning all files)
        in_code_path = (
            not code_paths or
            "." in code_paths or
            any(file_path.startswith(p) for p in code_paths if p != ".")
        )
        in_doc_path = (
            not doc_paths or
            "." in doc_paths or
            any(file_path.startswith(p) for p in doc_paths if p != ".")
        )

        if is_code and in_code_path:
            content = git_show_file(path, main_ref, file_path)
            if content:
                entries = index_code_content(file_path, content, status="permanent")
                code_entries.extend(entries)
        elif file_path.endswith(".md") and in_doc_path:
            content = git_show_file(path, main_ref, file_path)
            if content:
                entries = index_doc_content(file_path, content, status="permanent")
                doc_entries.extend(entries)

    # Batch insert entries
    if code_entries:
        _upsert_entries(db, code_entries, "code", label="  Indexing code")
        if not quiet:
            click.echo(f"  Indexed {len(code_entries)} code symbols as permanent")

    if doc_entries:
        _upsert_entries(db, doc_entries, "docs", label="  Indexing docs")
        if not quiet:
            click.echo(f"  Indexed {len(doc_entries)} doc chunks as permanent")

    # Show final stats
    if not quiet:
        stats = db.stats()
        ephemeral = db.get_ephemeral_stats()
        click.echo(f"  Total: {stats['total']} ({ephemeral['total']} ephemeral)")


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

            # Sync (skip reindex - that's expensive and only needed after pulls)
            subprocess.run(
                ["ragtime", "sync", "--quiet", "--auto-prune", "--no-reindex"],
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
    from importlib.metadata import version as get_version

    try:
        current = get_version("ragtime-cli")
    except Exception:
        current = "0.0.0"  # Fallback if not installed as package

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
