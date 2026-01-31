---
description: Start or resume work on a GitHub issue
allowed-arguments: issue number (e.g., /start 230)
allowed-tools: Bash, Read, Write, mcp__ragtime__search, mcp__ragtime__remember, AskUserQuestion
---

# Start or Resume Work

Smart command that starts fresh OR resumes existing work depending on context.

**Usage:**
- `/start 230` - Start or resume work on issue #230
- `/start` - Show your assigned issues and prompt for selection

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Adjust branch naming conventions, issue lookup, workflow
     steps, etc.
     ═══════════════════════════════════════════════════════════════════════════ -->

## Step 1: Determine the Issue

**If an issue number was provided as an argument**, use it directly.

**If no argument was provided**, check current state and prompt:

```bash
BRANCH=$(git branch --show-current)
ISSUE_NUM=$(echo "$BRANCH" | grep -oE '[0-9]+' | head -1)
DEVNAME=$(gh api user --jq '.login' 2>/dev/null || git config user.name | tr ' ' '-' | tr '[:upper:]' '[:lower:]')

echo "Current branch: $BRANCH"
echo "Developer: $DEVNAME"

if [ -n "$ISSUE_NUM" ]; then
  echo "Already on issue branch: #$ISSUE_NUM"
else
  echo ""
  echo "=== Your Assigned Issues ==="
  gh issue list --assignee @me --state open
fi
```

If not on an issue branch and no argument provided, ask: **"Which issue would you like to start?"**

## Step 2: Switch to the Branch

```bash
ISSUE_NUM={issue_number}

# Check if branch exists
EXISTING_BRANCH=$(git branch -a | grep -E "/$ISSUE_NUM-|/$ISSUE_NUM$" | head -1 | tr -d ' *')

if [ -n "$EXISTING_BRANCH" ]; then
  echo "Found existing branch: $EXISTING_BRANCH"
  git checkout "$EXISTING_BRANCH"
else
  # Create new branch
  ISSUE_TITLE=$(gh issue view $ISSUE_NUM --json title --jq '.title' | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-' | head -c 40)
  NEW_BRANCH="$DEVNAME/$ISSUE_NUM-$ISSUE_TITLE"
  echo "Creating new branch: $NEW_BRANCH"
  git checkout -b "$NEW_BRANCH" main
fi
```

<!-- CUSTOMIZABLE: Adjust branch naming convention for your project -->

## Step 3: Check for Existing Context

```bash
BRANCH=$(git branch --show-current)
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '-')
CONTEXT_FILE=".claude/memory/branches/$BRANCH_SLUG/context.md"

if [ -f "$CONTEXT_FILE" ]; then
  echo "✓ Found branch context"
  cat "$CONTEXT_FILE"
else
  echo "No existing context found"
fi
```

### Decision Tree:

**If context.md exists:**
→ **RESUME MODE** - Go to Step 4a

**If NO context.md:**
→ **FRESH START MODE** - Go to Step 4b

## Step 4a: Resume Mode

Load and present the existing context:

1. Read the branch's `context.md`
2. Display the current state and what's left
3. Ask: "Ready to continue? What would you like to work on first?"

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

Also load any branch memories:

```
mcp__ragtime__search:
  query: "decisions patterns architecture"
  namespace: "branch-{branch}"
  limit: 10
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

Present:

```
## Resuming Issue #{issue_num}

### Current State:
{from context.md}

### What's Left:
{from context.md}

### Branch Memories:
{list of decisions/patterns stored}

Ready to continue? What would you like to work on first?
```

## Step 4b: Fresh Start Mode

Create a new implementation plan:

1. Fetch issue details:
   ```bash
   gh issue view "$ISSUE_NUM" --json title,body,labels,assignees
   ```

2. Parse the GitHub issue (title, body, labels)

3. Identify which parts of the app this touches

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     ═══════════════════════════════════════════════════════════════════════════ -->

4. Search for relevant context:

   **App knowledge (architecture for those areas):**
   ```
   mcp__ragtime__search:
     query: "{component or feature area}"
     namespace: "app"
     limit: 10
   ```

   **Team conventions:**
   ```
   mcp__ragtime__search:
     query: "conventions standards"
     namespace: "team"
     limit: 10
   ```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

5. Create an implementation plan based on:
   - Issue requirements
   - Codebase knowledge from memories
   - Team conventions

6. Present the plan and ask if the user wants to proceed or adjust

## Step 5: Save Initial Context (Fresh Start only)

**After user approves the plan**, save it to context.md:

```bash
BRANCH=$(git branch --show-current)
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '-')
mkdir -p ".claude/memory/branches/$BRANCH_SLUG"
```

Write the context.md with the plan (see `/handoff` for format).

## Step 6: Ready to Work

```
Ready to work on Issue #{issue_num}!

What would you like to tackle first?
```

## Syncing Teammate Context

If you need context from a teammate's branch:

```bash
# Fetch their branch
git fetch origin jm/feature-auth

# Sync their memories
ragtime sync origin/jm/feature-auth
```

This creates `branches/jm-feature-auth(unmerged)/` with their context and memories, searchable but gitignored.
