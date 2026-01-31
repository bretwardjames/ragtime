---
description: Search the local memory system for knowledge
allowed-arguments: search query and optional filters (e.g., /recall auth patterns, /recall --namespace app)
allowed-tools: mcp__ragtime__search, mcp__ragtime__list_memories, AskUserQuestion
---

# Recall

Search the local memory system for stored knowledge.

**Usage:**
- `/recall auth patterns` - Search for auth-related patterns
- `/recall #301 decisions` - Decisions related to issue #301
- `/recall --namespace team` - All team conventions
- `/recall` - Interactive search mode

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Modify query parsing, add project-specific filters,
     adjust result formatting, etc.
     ═══════════════════════════════════════════════════════════════════════════ -->

## Query Syntax

```
/recall [modifiers] <search terms> [filters]
```

**Modifiers:**
- `all:` - Include all confidence levels (default: high + medium only)
- `pre-merge:` - Include unmerged branch memories from teammates

**Filters:**
- `#301` or `--issue 301` - Filter by issue number
- `@auth` or `--component auth` - Filter by component
- `--type decision` - Filter by memory type
- `--namespace app` - Filter by namespace
- `--status active` - Filter by status

<!-- CUSTOMIZABLE: Add your own filter shortcuts -->

## Step 1: Parse the Query

**If `$ARGUMENTS` provided:**
- Parse modifiers, search terms, and filters

**If no arguments:**
- Ask: "What are you looking for?"
- Optionally ask for filters

## Step 2: Build the Search

Construct filters based on parsed query:

- Default: exclude `status: "abandoned"` and `confidence: "low"`
- If `all:` modifier: include everything
- If `pre-merge:` modifier: include `status: "pre-merge"`

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     These commands must match ragtime's expected format.
     ═══════════════════════════════════════════════════════════════════════════ -->

Execute the search:

```
mcp__ragtime__search:
  query: "{search terms}"
  namespace: "{namespace if specified}"
  type: "{type if specified}"
  component: "{component if specified}"
  limit: 20
```

For listing without semantic search:

```
mcp__ragtime__list_memories:
  namespace: "{namespace}"
  type: "{type}"
  status: "{status}"
  component: "{component}"
  limit: 20
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Step 3: Present Results

Format results clearly:

```
## 🧠 Recall: "{search terms}"

Found {count} memories:

### High Confidence
───────────────────────────────────────────

**Auth uses JWT with 15-min expiry**
- Type: architecture | Component: auth
- Added: 2026-01-15 | Source: pr-graduate

**Sessions stored in Redis, not cookies**
- Type: decision | Component: auth
- Added: 2026-01-10 | Source: meeting

### Medium Confidence
───────────────────────────────────────────

**Consider rate limiting on auth endpoints**
- Type: pattern | Component: auth
- Added: 2026-01-20 | Source: remember

───────────────────────────────────────────

Showing {count} of {total} results.
Refine with: /recall auth --type decision
Include low confidence: /recall all: auth
```

### Pre-Merge Results

If pre-merge memories are found, show them separately with a warning:

```
### ⚠️ Pre-Merge (from teammate branches)
───────────────────────────────────────────

**New auth middleware pattern** (pre-merge)
- From: origin/jm/feature-auth
- Type: architecture | Component: auth
- ⚠️ Not yet merged - may change

```

## Step 4: Offer Actions

```
What would you like to do?

1. **Use this context** - I'll incorporate these into our work
2. **Refine search** - Try different filters
3. **View details** - See full content of a specific memory
4. **Done** - Just wanted to see what we know
```

## Example Queries

| Query | What it finds |
|-------|---------------|
| `/recall auth` | All auth-related memories (high/medium confidence) |
| `/recall #301 decisions` | Decisions made during issue #301 |
| `/recall --namespace team` | All team conventions |
| `/recall --component shifts --type architecture` | Shifts architecture |
| `/recall pre-merge: auth` | Include teammate WIP auth knowledge |
| `/recall all: validation` | Include low-confidence validation memories |

## No Results Handling

```
No memories found for "{query}" with current filters.

Suggestions:
- Try broader terms: /recall {simpler query}
- Include all confidence: /recall all: {query}
- Check different namespace: /recall --namespace team {query}
- Include pre-merge: /recall pre-merge: {query}
```

## Notes

- Results are sorted by relevance (semantic similarity)
- Default excludes low confidence and abandoned memories
- Use `all:` modifier to see everything
- Pre-merge memories are from `(unmerged)` folders synced from teammate branches
