---
description: Audit local memories for duplicates, conflicts, and stale data
allowed-tools: Bash, mcp__ragtime__search, mcp__ragtime__list_memories, mcp__ragtime__forget, mcp__ragtime__update_status, AskUserQuestion
---

# Memory Audit

Periodic cleanup of local memories. This is a **human-in-the-loop** review - no automatic deletions.

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Add project-specific audit rules, adjust the report format,
     add custom namespace checks, etc.
     ═══════════════════════════════════════════════════════════════════════════ -->

## Namespaces to Audit

- `app` - Codebase knowledge (architecture, decisions)
- `team` - Team conventions (standards, processes)
- `user-*` - Developer preferences
- `branch-*` - Branch-specific context and decisions

## Step 1: Gather All Memories

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

```
mcp__ragtime__list_memories:
  limit: 100
```

For each namespace:
```
mcp__ragtime__list_memories:
  namespace: "{namespace}"
  limit: 50
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Step 2: Identify Issues

Group memories by topic and identify:

| Issue | Description |
|-------|-------------|
| **DUPLICATES** | Memories saying essentially the same thing |
| **CONFLICTS** | Memories that contradict each other |
| **STALE** | References to code/features that no longer exist |
| **ORPHANED** | Branch memories for deleted/merged branches |
| **LOW_VALUE** | Vague memories that aren't useful |

## Step 3: Check for Stale Branches

```bash
# List branch memory folders
ls -la .claude/memory/branches/

# For each, check if branch still exists
for dir in .claude/memory/branches/*/; do
  branch_slug=$(basename "$dir")
  # Check if branch exists on remote
  if ! git branch -a | grep -q "$branch_slug"; then
    echo "⚠️  Potentially stale: $branch_slug"
  fi
done
```

Also run:
```bash
ragtime prune --dry-run
```

## Step 4: Present Report

```
## Memory Audit Report

### App Namespace ({count} memories)
- Potential duplicates: {n}
- Potential conflicts: {n}
- Possibly stale: {n}

### Team Namespace ({count} memories)
- Potential duplicates: {n}

### Branch Namespaces
- Active branches: {n}
- Stale (unmerged) folders: {n}
- Ready to prune: {n}

───────────────────────────────────────────

### Issues Found:

**1. Possible Duplicate:**
- "Auth uses JWT tokens" (app, abc123)
- "JWT auth with 15-min expiry" (app, def456)
→ Action: Merge into one?

**2. Possible Conflict:**
- "Use tabs for indentation" (team, ghi789)
- "Use 2-space indentation" (team, jkl012)
→ Action: Which is correct?

**3. Stale Branch:**
- branches/old-feature/ (branch deleted)
→ Action: Prune with `ragtime prune`?
```

## Step 5: Get User Approval

For each issue found, ask:

- "Should I merge these duplicates?"
- "Which of these conflicting memories is correct?"
- "Should I mark this as abandoned?"
- "Run `ragtime prune` to clean stale branches?"

Only make changes the user explicitly approves.

## Step 6: Execute Approved Actions

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

**Delete a memory:**
```
mcp__ragtime__forget:
  memory_id: "{id}"
```

**Mark as abandoned:**
```
mcp__ragtime__update_status:
  memory_id: "{id}"
  status: "abandoned"
```

**Prune stale branches:**
```bash
ragtime prune
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Suggested Cadence

Run monthly or when memories feel cluttered.
