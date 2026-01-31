---
description: Graduate branch knowledge to app after PR merge (fallback)
allowed-tools: Bash, mcp__ragtime__search, mcp__ragtime__graduate, mcp__ragtime__update_status, mcp__ragtime__remember, AskUserQuestion
---

# PR Graduate: Curate Branch Knowledge (Post-Merge)

> **Preferred workflow:** Use `/create-pr` instead - it graduates memories *before*
> creating the PR so knowledge is committed alongside code.
>
> Use this command only if you already merged without graduating.

After a PR is merged, review branch memories and decide what becomes permanent app knowledge.

**This is a human-in-the-loop process** - you curate which memories graduate.

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Adjust the curation workflow, add project-specific
     graduation criteria, modify the summary format, etc.
     ═══════════════════════════════════════════════════════════════════════════ -->

## Process Overview

```
For EACH branch memory:

✅ Graduate → Copy to app namespace with high confidence
📚 Keep     → Leave in branch (reference/history)
❌ Abandon  → Mark as abandoned (noise, superseded)

Branch memories are preserved - nothing is deleted.
```

## Step 1: Get the Branch

```bash
BRANCH=$(git branch --show-current)
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '-')
ISSUE_NUM=$(echo "$BRANCH" | grep -oE '[0-9]+' | head -1)

echo "Branch: $BRANCH"
echo "Issue: #$ISSUE_NUM"

# Check if PR was merged
gh pr list --head "$BRANCH" --state merged --json number,title
```

## Step 2: Gather Branch Memories

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

```
mcp__ragtime__search:
  query: "decisions patterns architecture"
  namespace: "branch-{branch}"
  limit: 50
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Step 3: Present Memories for Curation

Display each memory with options:

```
───────────────────────────────────────────
📋 BRANCH MEMORY CURATION
───────────────────────────────────────────

Branch: {branch}
Total memories found: {count}

For each memory, choose:
  ✅ Graduate - Promote to app knowledge (high confidence)
  📚 Keep - Leave in branch for reference
  ❌ Abandon - Mark as noise/superseded

───────────────────────────────────────────

**Memory 1 of {count}:**

"{memory content preview - first 200 chars}..."

Type: {type} | Added: {date}

What should happen to this memory?
1. ✅ Graduate to app
2. 📚 Keep in branch
3. ❌ Mark as abandoned
4. 👀 Show full content
```

## Step 4: Process Each Memory

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

### If ✅ Graduate:

```
mcp__ragtime__graduate:
  memory_id: "{id}"
  confidence: "high"
```

This creates a copy in `app/` namespace and marks the original as graduated.

### If 📚 Keep:

No action needed - memory stays in branch namespace for reference.

### If ❌ Abandon:

```
mcp__ragtime__update_status:
  memory_id: "{id}"
  status: "abandoned"
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Step 5: Handle Context Document

The branch's `context.md` is a full document, not individual memories:

```
The context.md contains the full development context.

Options:
1. **Extract key insights** - I'll identify valuable patterns to graduate
2. **Keep as reference** - Leave it in branch history
3. **Skip** - Context was just for session continuity
```

If extracting: Identify the most valuable insights and present for approval before graduating.

## Step 6: Summary

```
───────────────────────────────────────────
✅ PR GRADUATION COMPLETE
───────────────────────────────────────────

Branch: {branch}

Memories processed: {total}
  ✅ Graduated to app: {count}
  📚 Kept in branch: {count}
  ❌ Marked abandoned: {count}

Graduated knowledge now searchable via:
  /recall {topic} --namespace app
```

## Quick Mode

For simpler PRs, offer quick mode:

```
Found {count} branch memories.

Options:
1. **Review each** - Curate one by one (recommended)
2. **Quick mode** - I'll propose which to graduate
3. **Graduate all** - Promote everything
4. **Keep all** - Leave everything in branch
```

Quick mode generates a proposal:

```
## Quick Mode Proposal

**Recommend Graduate:**
- "Auth uses JWT with 15-min expiry" ← architecture insight
- "Redis chosen for session storage" ← key decision

**Recommend Keep:**
- "Debugging: token refresh issue" ← development context

Approve? (yes/edit/review-each)
```

## Notes

- Branch memories stay forever (history) - only status changes
- Graduated memories get `source: "pr-graduate"` and high confidence
- Abandoned memories are excluded from default searches
- Use `/recall --namespace branch-{name}` to see branch history
