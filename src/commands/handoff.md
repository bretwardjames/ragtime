---
description: Save current context to branch memory for session continuity
allowed-tools: Bash, Read, Write, mcp__ragtime__remember
---

# Save Context for Handoff

Capture the current session's context into the branch's `context.md` so the next session (or another developer) can pick up seamlessly.

The context lives permanently in `.claude/memory/branches/{branch}/context.md` - no cleanup needed.

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Adjust the context template, add project-specific sections,
     modify the commit message format, etc.
     ═══════════════════════════════════════════════════════════════════════════ -->

## Step 1: Gather Current State

```bash
BRANCH=$(git branch --show-current)
ISSUE_NUM=$(echo "$BRANCH" | grep -oE '[0-9]+' | head -1)
DEVNAME=$(gh api user --jq '.login' 2>/dev/null || git config user.name | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
TIMESTAMP=$(date +%Y-%m-%d)

echo "Branch: $BRANCH"
echo "Issue: #$ISSUE_NUM"
echo "Developer: $DEVNAME"

# Get recent commits on this branch
echo ""
echo "=== Recent Commits ==="
git log main..HEAD --oneline 2>/dev/null || git log -10 --oneline

# Get current changes
echo ""
echo "=== Uncommitted Changes ==="
git status --short
```

## Step 2: Summarize Session Context

Based on the conversation, create a summary that includes:

1. **What This Branch Does** - High-level purpose
2. **Current State** - What's been done, recent commits
3. **What's Left To Do** - Remaining tasks, in priority order
4. **Testing Status** - What's been tested, what hasn't
5. **Known Issues / Lessons Learned** - Gotchas for the next session
6. **Architecture Decisions** - Why things were done a certain way
7. **Next Steps** - Specific instructions for resuming

## Step 3: Write to Branch Context

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     The file path and frontmatter format must match ragtime's expectations.
     ═══════════════════════════════════════════════════════════════════════════ -->

Write to `.claude/memory/branches/{branch-slug}/context.md`:

```markdown
---
id: context
namespace: branch-{branch}
type: context
status: active
added: '{date}'
author: {devname}
issue: '#{issue_num}'
---

## Branch: {branch}

### What This Branch Does

{summary}

---

## CURRENT STATE (as of {date})

{current state details}

---

## WHAT'S LEFT TO DO

{prioritized task list}

---

## TESTING STATUS

### Tested and Working:
{list}

### Not Yet Tested:
{list}

---

## KNOWN ISSUES / LESSONS LEARNED

{numbered list}

---

## ARCHITECTURE DECISIONS

{bullet points}

---

## NEXT STEPS FOR NEW SESSION

{specific instructions}
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

<!-- CUSTOMIZABLE: Add project-specific sections to the context template -->

## Step 4: Index the Context

After writing the file, ensure it's indexed:

```
mcp__ragtime__remember:
  content: "{summary of context for search}"
  namespace: "branch-{branch}"
  type: "context"
  issue: "#{issue_num}"
  source: "handoff"
```

## Step 5: Commit and Push

```bash
BRANCH=$(git branch --show-current)
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '-')
ISSUE_NUM=$(echo "$BRANCH" | grep -oE '[0-9]+' | head -1)

# Block pushes to main/master
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "❌ ERROR: Cannot push directly to $BRANCH"
  exit 1
fi

git add ".claude/memory/branches/$BRANCH_SLUG/context.md"
git commit -m "docs(context): update branch context for session continuity

Relates to #$ISSUE_NUM

Co-Authored-By: Claude <noreply@anthropic.com>"

git push -u origin "$BRANCH"
```

## Step 6: Confirm Handoff Complete

```
✅ Handoff Complete

- Context saved to: .claude/memory/branches/{branch}/context.md
- Committed and pushed to remote
- Ready for next session or another developer

To resume: Start a new session and say "continue on this branch"
```

## Notes

- The context.md file lives permanently with the branch - no cleanup needed
- Each `/handoff` overwrites the previous context.md (it's always current state)
- On `/start`, Claude reads the branch's context.md to resume
- When syncing from teammate branches, their context.md comes along too
