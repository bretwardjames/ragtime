# ragtime-cli

Local-first memory and RAG system for Claude Code. Semantic search over code, docs, and team knowledge.

## Features

- **Memory Storage**: Store structured knowledge with namespaces, types, and metadata
- **Semantic Search**: Query memories and docs with natural language
- **Cross-Branch Sync**: Share context with teammates before PRs merge
- **Convention Checking**: Verify code follows team standards before PRs
- **Doc Generation**: Generate documentation from code (stubs or AI-powered)
- **Debug Tools**: Verify index integrity, inspect similarity scores
- **MCP Server**: Native Claude Code integration
- **Claude Commands**: Pre-built `/remember`, `/recall`, `/create-pr`, `/generate-docs` commands
- **ghp-cli Integration**: Auto-context when starting issues

## Installation

```bash
pip install ragtime-cli
```

## Quick Start

```bash
# Initialize in your project
ragtime init

# Index your docs
ragtime index

# Store a memory
ragtime remember "Auth uses JWT with 15-min expiry" \
  --namespace app \
  --type architecture \
  --component auth

# Search memories
ragtime search "authentication" --namespace app

# Install Claude commands
ragtime install --workspace

# Check for updates
ragtime update --check
```

## CLI Commands

### Memory Storage

```bash
# Store a memory
ragtime remember "content" --namespace app --type architecture --component auth

# List memories
ragtime memories --namespace app --type decision

# Graduate branch memory to app
ragtime graduate <memory-id>

# Delete a memory
ragtime forget <memory-id>
```

### Search & Indexing

```bash
# Index docs
ragtime index --type docs

# Semantic search
ragtime search "how does auth work" --namespace app --limit 10

# Reindex memory files
ragtime reindex

# Audit docs for missing frontmatter
ragtime audit docs/
ragtime audit docs/ --fix    # Interactively add frontmatter
ragtime audit docs/ --json   # Machine-readable output
```

### Documentation Generation

```bash
# Generate doc stubs from code
ragtime generate src/ --stubs

# Specify output location
ragtime generate src/ --stubs -o docs/api

# Python only
ragtime generate src/ --stubs -l python

# Include private methods
ragtime generate src/ --stubs --include-private
```

### Debug & Verification

```bash
# Debug a search query (show similarity scores)
ragtime debug search "authentication"
ragtime debug search "auth" --show-vectors

# Find similar documents
ragtime debug similar docs/auth/jwt.md

# Index statistics by namespace/type
ragtime debug stats
ragtime debug stats --by-namespace
ragtime debug stats --by-type

# Verify index integrity
ragtime debug verify
```

### Cross-Branch Sync

```bash
# Sync all teammate branch memories
ragtime sync

# Auto-prune stale synced folders
ragtime sync --auto-prune

# Manual prune
ragtime prune --dry-run
ragtime prune
```

### Daemon (Auto-Sync)

```bash
# Start background sync daemon
ragtime daemon start --interval 5m

# Check status
ragtime daemon status

# Stop daemon
ragtime daemon stop
```

### Claude Integration

```bash
# Install Claude commands to workspace
ragtime install --workspace

# Install globally
ragtime install --global

# List available commands
ragtime install --list

# Set up ghp-cli hooks
ragtime setup-ghp
```

## Storage Structure

```
.ragtime/
├── config.yaml              # Configuration
├── CONVENTIONS.md           # Team rules (checked by /create-pr)
├── app/{component}/         # Graduated app knowledge (tracked)
│   └── {id}-{slug}.md
├── team/                    # Team conventions (tracked)
│   └── {id}-{slug}.md
├── branches/
│   ├── {branch-slug}/       # Your branch (tracked in git)
│   │   ├── context.md
│   │   └── {id}-{slug}.md
│   └── .{branch-slug}/      # Synced from teammates (gitignored, dot-prefix)
├── archive/branches/        # Archived completed branches (tracked)
└── index/                   # ChromaDB vector store (gitignored)
```

## Configuration

`.ragtime/config.yaml`:

```yaml
docs:
  paths: ["docs", ".ragtime"]
  patterns: ["**/*.md"]
  exclude: ["**/node_modules/**"]

code:
  paths: ["."]
  languages: ["python", "typescript", "dart"]

conventions:
  files: [".ragtime/CONVENTIONS.md"]
  also_search_memories: true
```

## Memory Format

Memories are markdown files with YAML frontmatter:

```markdown
---
id: abc123
namespace: app
type: architecture
component: auth
confidence: high
status: active
added: '2025-01-31'
author: bretwardjames
---

Auth uses JWT tokens with 15-minute expiry for security.
Sessions are stored in Redis, not cookies.
```

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `app` | How the codebase works (architecture, decisions) |
| `team` | Team conventions and standards |
| `user-{name}` | Individual preferences |
| `branch-{name}` | Work-in-progress context |

## Memory Types

| Type | Description |
|------|-------------|
| `architecture` | System design, patterns |
| `feature` | How features work |
| `decision` | Why we chose X over Y |
| `convention` | Team standards |
| `pattern` | Reusable approaches |
| `integration` | External service connections |
| `context` | Session handoff |

## Claude Commands

After `ragtime install --workspace`:

| Command | Purpose |
|---------|---------|
| `/remember` | Capture knowledge mid-session |
| `/recall` | Search memories |
| `/handoff` | Save session context |
| `/start` | Resume work on an issue |
| `/create-pr` | Check conventions, graduate memories, create PR |
| `/generate-docs` | AI-powered documentation generation from code |
| `/import-docs` | Migrate existing docs to memories |
| `/pr-graduate` | Curate branch knowledge (fallback if forgot before PR) |
| `/audit` | Find duplicates/conflicts in memories |

## MCP Server

Add to your Claude config (`.mcp.json`):

```json
{
  "mcpServers": {
    "ragtime": {
      "command": "ragtime-mcp",
      "args": ["--path", "."]
    }
  }
}
```

Available tools:
- `remember` - Store a memory
- `search` - Semantic search
- `list_memories` - List with filters
- `get_memory` - Get by ID
- `store_doc` - Store document verbatim
- `forget` - Delete memory
- `graduate` - Promote branch → app
- `update_status` - Change memory status

## ghp-cli Integration

If you use [ghp-cli](https://github.com/bretwardjames/ghp-cli):

```bash
# Register ragtime hooks
ragtime setup-ghp
```

This auto-creates `context.md` from issue details when you run `ghp start`.

## Workflow

### Starting Work

```bash
ghp start 123              # Creates branch + context.md
# or
ragtime new-branch 123     # Just the context
```

### During Development

```bash
/remember "API uses rate limiting"   # Capture insights
/handoff                              # Save progress for later
```

### Before PR

```bash
/create-pr
# 1. Checks code against CONVENTIONS.md
# 2. Reviews branch memories
# 3. Graduates selected memories to app/
# 4. Commits knowledge with code
# 5. Creates PR
```

### After Merge

Graduated knowledge is already in the PR. Run `ragtime prune` to clean up synced folders.

## License

MIT
