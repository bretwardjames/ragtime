# ragtime-cli

Local-first memory and RAG system for Claude Code. Semantic search over code, docs, and team knowledge.

## Features

- **Memory Storage**: Store structured knowledge with namespaces, types, and metadata
- **Semantic Search**: Query memories, docs, and code with natural language
- **Code Indexing**: Index functions, classes, and composables from Python, TypeScript, Vue, and Dart
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
# Index everything (docs + code)
ragtime index

# Incremental index (only changed files - fast!)
ragtime index  # ~8 seconds vs ~5 minutes for unchanged codebases

# Index only docs
ragtime index --type docs

# Index only code (functions, classes, composables)
ragtime index --type code

# Full re-index (removes old entries, recomputes all embeddings)
ragtime index --clear

# Semantic search across all content
ragtime search "how does auth work" --limit 10

# Search only code
ragtime search "useAsyncState" --type code

# Search only docs
ragtime search "authentication" --type docs --namespace app

# Hybrid search: semantic + keyword filtering
# Use -r/--require to ensure terms appear in results
ragtime search "error handling" -r mobile -r dart

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
  paths: ["docs"]
  patterns: ["**/*.md"]
  exclude: ["**/node_modules/**", "**/.ragtime/**"]

code:
  paths: ["."]
  languages: ["python", "typescript", "javascript", "vue", "dart"]
  exclude: ["**/node_modules/**", "**/build/**", "**/dist/**"]

conventions:
  files: [".ragtime/CONVENTIONS.md"]
  also_search_memories: true
```

## How Search Works

Search returns **summaries with locations**, not full code:

1. **What you get**: Function signatures, docstrings, class definitions
2. **What you don't get**: Full implementations
3. **What to do**: Use the file path + line number to read the full code

This is intentional - embeddings work better on focused summaries than large code blocks. The search tells you *what exists and where*, then you read the file for details.

For Claude/MCP usage: The search tool description instructs Claude to read returned file paths for full implementations before making code changes.

### Smart Query Understanding

Search automatically detects qualifiers in natural language:

```bash
# These are equivalent - qualifiers are auto-detected
ragtime search "error handling in mobile app"
ragtime search "error handling" -r mobile

# Use --raw for literal/exact search
ragtime search "mobile error handling" --raw
```

Auto-detected qualifiers include: mobile, web, desktop, ios, android, flutter, react, vue, dart, python, typescript, auth, api, database, frontend, backend, and more.

### Tiered Search

Use tiered search to prioritize curated knowledge over raw code:

```bash
# Via MCP
search(query="authentication", tiered=True)
```

Tiered search returns results in priority order:
1. **Memories** - Curated, high-signal knowledge
2. **Documentation** - Indexed markdown files
3. **Code** - Function signatures and symbols

### Hybrid Search

For explicit keyword filtering, use `require_terms`:

```bash
# CLI
ragtime search "error handling" -r mobile -r dart

# MCP
search(query="error handling", require_terms=["mobile", "dart"])
```

This combines semantic similarity (finds conceptually related content) with keyword filtering (ensures qualifiers aren't ignored).

### Hierarchical Doc Chunking

Long markdown files are automatically chunked by headers for better search accuracy:

- Each section becomes a separate searchable chunk
- Parent headers are preserved as context in the embedding
- Short docs (<500 chars) remain as single chunks
- Section path is stored (e.g., "Installation > Configuration > Environment Variables")

### Feedback Loop

Search quality improves over time based on usage patterns:

```bash
# Record when a result is useful (via MCP)
record_feedback(query="auth flow", result_file="src/auth.py", action="used")

# View usage statistics
feedback_stats()
```

Frequently-used files receive a boost in future search rankings.

## Code Indexing

The code indexer extracts meaningful symbols from your codebase:

| Language | What Gets Indexed |
|----------|-------------------|
| Python | Classes, methods, functions (with docstrings) |
| TypeScript/JS | Functions, classes, interfaces, types (exported and non-exported) |
| Vue | Components, composable usage (useXxx calls) |
| Dart | Classes, functions, mixins, extensions |

Each symbol is indexed with:
- **content**: The code snippet with signature and docstring
- **file**: Full path to the source file
- **line**: Line number for quick navigation
- **symbol_name**: Searchable name (e.g., `useAsyncState`, `JWTManager.validate`)
- **symbol_type**: `function`, `class`, `method`, `interface`, `composable`, etc.

Example search results:
```
ragtime search "useAsyncState" --type code

[1] /apps/web/components/agency/payers.vue
    Type: code | Symbol: payers:useAsyncState
    Score: 0.892
    Uses composable: useAsyncState...
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
- `search` - Semantic search (supports tiered mode and auto-extraction)
- `list_memories` - List with filters
- `get_memory` - Get by ID
- `store_doc` - Store document verbatim
- `forget` - Delete memory
- `graduate` - Promote branch → app
- `update_status` - Change memory status
- `record_feedback` - Record when search results are used (improves future rankings)
- `feedback_stats` - View search result usage patterns

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
