---
description: Create a PR with convention checks and graduated knowledge
allowed-arguments: optional PR title
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion, mcp__ragtime__search, mcp__ragtime__memories
---

# Create PR

Create a pull request with branch knowledge graduated and committed alongside the code.

**Usage:**
- `/create-pr` - Interactive PR creation with memory curation
- `/create-pr "Add authentication flow"` - With suggested title

## Overview

This command ensures code follows team standards and knowledge gets merged with the code:

1. Check code against team conventions and app patterns
2. Review branch memories and decide what graduates to `app/`
3. Check for conflicts with existing app knowledge
4. Commit graduated memories as part of the PR
5. Create the pull request

## Step 1: Gather Context

```bash
BRANCH=$(git branch --show-current)
BRANCH_SLUG=$(echo "$BRANCH" | tr '/' '-')

echo "Branch: $BRANCH"

# Check for uncommitted changes
git status --short

# Get commits on this branch
git log main..HEAD --oneline
```

## Step 2: Check Code Against Team Conventions

Before creating the PR, verify the code follows team standards.

### Load conventions config

Check `.ragtime/config.yaml` for conventions settings:

```yaml
# Default config (if not specified):
conventions:
  files:
    - ".ragtime/CONVENTIONS.md"
  also_search_memories: true
```

### Read conventions files

For each file in `conventions.files`, read it if it exists:

```bash
# Check for conventions files
cat .ragtime/CONVENTIONS.md 2>/dev/null
# Or other configured files like docs/CODING_STANDARDS.md
```

The conventions doc should contain clear, checkable rules like:
- "All API endpoints must use auth middleware"
- "Use async/await, not .then() chains"
- "Never commit .env files"

### Also search memories (if enabled)

If `also_search_memories: true` (default), also search for convention memories:

```
mcp__ragtime__search:
  query: "convention standard pattern rule always never"
  namespace: "team"
  limit: 20
```

```
mcp__ragtime__search:
  query: "pattern architecture convention"
  namespace: "app"
  type: "convention,pattern"
  limit: 20
```

### Get changed files

```bash
# Files changed in this branch
git diff --name-only main...HEAD
```

### Review code against conventions

For each relevant convention/pattern found, check if the changed code follows it:

1. Read the convention/pattern memory
2. Read the relevant changed files
3. Check for compliance

Present findings:

```
───────────────────────────────────────────
🔍 CONVENTION CHECK
───────────────────────────────────────────

Checked {file_count} changed files against {convention_count} team conventions.

✅ PASSING:
- "Use async/await, not .then()" - All async code uses await
- "Error responses use ApiError class" - Verified in api/routes.ts

⚠️ POTENTIAL ISSUES:
- "All API endpoints need auth middleware"
  → src/api/newEndpoint.ts:15 - No auth middleware detected

- "Use logger.error(), not console.error()"
  → src/services/auth.ts:42 - Found console.error()

❌ VIOLATIONS:
- "Never commit .env files"
  → .env.local is staged
───────────────────────────────────────────
```

### Handle issues

If issues found:

```
Found {issue_count} potential issues.

Options:
1. **Fix now** - I'll help fix these before creating PR
2. **Ignore** - Create PR anyway (add justification comment)
3. **Review each** - Go through issues one by one
4. **Cancel** - Fix manually and try again
```

If "Fix now": Help the user fix each issue before proceeding.

If "Ignore": Ask for justification to include in PR description.

**Only proceed to memory curation after issues are resolved or acknowledged.**

## Step 3: Check for Branch Memories

Search for memories in this branch's namespace:

```bash
# Check .ragtime/branches/{branch-slug}/ for memory files
ls -la .ragtime/branches/$BRANCH_SLUG/ 2>/dev/null || echo "No branch memories found"
```

Also search the index:

```
mcp__ragtime__search:
  query: "*"
  namespace: "branch-{branch-slug}"
  limit: 50
```

**If no branch memories found:** Skip to Step 8 (Create PR).

**If memories found:** Continue to Step 4.

## Step 4: Present Memories for Curation

```
───────────────────────────────────────────
📋 BRANCH KNOWLEDGE REVIEW
───────────────────────────────────────────

Before creating the PR, let's review knowledge from this branch.

Graduated memories will be:
- Moved to app/ or team/ namespace
- Committed as part of this PR
- Merged with your code changes

Found {count} memories in branch-{branch}:

───────────────────────────────────────────
```

For each memory, show:

```
**Memory {n} of {count}:**

"{content preview - first 150 chars}..."

Type: {type} | Component: {component}
File: {relative_path}

Action?
1. ✅ Graduate to app/
2. 🏛️ Graduate to team/ (conventions/standards)
3. 📚 Keep in branch (don't include in PR)
4. ❌ Abandon (mark as noise)
5. 👀 Show full content
```

## Step 5: Check for Conflicts

For each memory being graduated, search for similar content in app/:

```
mcp__ragtime__search:
  query: "{memory content keywords}"
  namespace: "app"
  limit: 5
```

If similar memories exist (similarity > 0.8), present:

```
⚠️ POTENTIAL CONFLICT

Graduating: "{new memory preview}..."

Similar existing memory in app/:
"{existing memory preview}..."
File: {existing_file}

Options:
1. **Keep both** - They're different enough
2. **Replace existing** - New one is better/more current
3. **Merge** - Combine into one comprehensive memory
4. **Skip graduation** - Existing one is sufficient
```

### If merging:

Ask the user to provide the merged content, or offer to draft it:

```
I can draft a merged version combining both. Want me to try?
```

If yes, present the draft for approval before saving.

## Step 6: Execute Graduation

For each memory to graduate:

### Move the file

```bash
# From: .ragtime/branches/{branch-slug}/{id}-{slug}.md
# To:   .ragtime/app/{component}/{id}-{slug}.md (or .ragtime/team/)

mkdir -p .ragtime/app/{component}
mv .ragtime/branches/{branch-slug}/{id}-{slug}.md .ragtime/app/{component}/
```

### Update the frontmatter

Edit the file to update:
- `namespace: app` (or `team`)
- `confidence: high`
- `confidence_reason: pr-graduate`
- `source: pr-graduate`
- Add `graduated_from: branch-{branch}`

### Stage the changes

```bash
git add .ragtime/app/
git add .ragtime/team/
# Don't stage branch/ folder - it stays local or gets cleaned up later
```

## Step 7: Commit Knowledge (if any graduated)

If memories were graduated:

```bash
git add .ragtime/app/ .ragtime/team/

git commit -m "docs: graduate branch knowledge to app

Graduated {count} memories from branch-{branch}:
- {memory1 summary}
- {memory2 summary}
..."
```

## Step 8: Create the PR

```bash
# Check if we need to push
git push -u origin $BRANCH 2>/dev/null || git push

# Create PR
gh pr create --title "{title}" --body "$(cat <<'EOF'
## Summary

{summary from commits and context}

## Convention Compliance

{if all passed}
✅ All team conventions verified

{if issues were acknowledged}
⚠️ Known deviations:
- {issue}: {justification}

## Knowledge Added

{if memories were graduated}
This PR includes {count} graduated memories:
- {list of graduated memories with types}

## Test Plan

- [ ] {test items}

---
🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

## Step 9: Summary

```
───────────────────────────────────────────
✅ PR CREATED
───────────────────────────────────────────

PR: {url}
Branch: {branch}

Knowledge graduated: {count}
  → app/: {app_count}
  → team/: {team_count}

Remaining in branch/: {kept_count}
Abandoned: {abandoned_count}

The graduated knowledge is now part of your PR.
Reviewers will see it alongside your code changes.
```

## Quick Mode

For branches with few memories, offer quick mode:

```
Found {count} branch memories.

1. **Review each** - Curate one by one
2. **Quick mode** - I'll propose what to graduate
3. **Graduate all to app/** - Promote everything
4. **Skip knowledge** - Create PR without graduating
```

Quick mode analyzes content and proposes:

```
## Quick Mode Proposal

**Graduate to app/:**
- "Auth uses JWT with 15-min expiry" (architecture)
- "Redis chosen for session storage" (decision)

**Keep in branch:**
- "Debug notes: token refresh" (task-state)

**Abandon:**
- "TODO: fix later" (noise)

Approve this? (yes / review each / edit)
```

## Notes

- Graduated memories are committed as part of the PR, so reviewers see them
- Branch memories that aren't graduated stay in `.ragtime/branches/` for reference
- After PR merges, run `ragtime prune` to clean up synced branch folders
- The old `/pr-graduate` command can still be used if you forget to graduate before PR
