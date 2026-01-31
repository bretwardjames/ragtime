---
description: Migrate existing docs into ragtime memory structure with AI-assisted classification
allowed-arguments: path to docs folder (e.g., /import-docs docs/)
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion
---

# Import Docs

Analyze an existing docs folder and migrate content into the ragtime memory structure.

**Usage:**
- `/import-docs docs/` - Analyze docs folder and create memories
- `/import-docs` - Interactive mode, asks for path

## Overview

This command helps teams that have existing documentation migrate into ragtime's structured memory system. It:

1. Scans all markdown files using `ragtime audit --json`
2. Analyzes each document's content to classify properly
3. Determines what should become memories vs. stay as indexed docs
4. Creates memories in `.ragtime/app/` or `.ragtime/team/` as appropriate

## Step 1: Get the Docs Path

**If `$ARGUMENTS` provided:**
- Use it as the docs path

**If no arguments:**
- Ask: "What docs folder should I analyze? (e.g., docs/, documentation/)"

## Step 2: Run Audit

Run the ragtime audit command to get a structured view of all docs:

```bash
ragtime audit {docs_path} --json
```

Parse the JSON output to get the list of files and initial suggestions.

## Step 3: Analyze Documents

For each document (or batch), read the content and classify:

### Classification Questions

For each doc, determine:

1. **Is this memory-worthy or just reference?**
   - Memory-worthy: Contains decisions, patterns, architecture insights, conventions
   - Reference: API docs, changelogs, auto-generated content, raw specs

2. **What type of knowledge is it?**
   - `architecture` - How systems/components work
   - `decision` - Why we chose X over Y (look for "we decided", "because", trade-off discussion)
   - `convention` - Team rules, coding standards, process docs
   - `pattern` - Reusable solutions, "how to do X"
   - `integration` - How external services connect
   - `feature` - Feature documentation

3. **What namespace?**
   - `app` - Technical knowledge about this codebase
   - `team` - Team conventions, process, standards

4. **What component?** (infer from path and content)

5. **Should this doc be split?**
   - Large docs with multiple distinct sections may become multiple memories
   - Each memory should be focused on ONE concept

### Classification Hints

Look for signals in the content:

| Signal | Likely Type |
|--------|-------------|
| "We decided to..." | decision |
| "Always do X when Y" | convention |
| "The {system} works by..." | architecture |
| "To implement X, follow these steps..." | pattern |
| "Integrates with {service} via..." | integration |
| ADR format, numbered decisions | decision |
| API endpoints, request/response | architecture |
| Setup instructions, onboarding | convention |

## Step 4: Choose Migration Strategy

Ask the user:

```
How should I handle the original docs?

1. **Memories only** - Create memories in .ragtime/, leave original docs unchanged
2. **Frontmatter only** - Add frontmatter to original docs for better indexing, no memories
3. **Both** - Add frontmatter to originals AND create memories for key insights (recommended)
```

Based on their choice, adjust what the migration plan includes.

## Step 5: Generate Migration Plan

Create a migration plan based on the user's strategy choice:

```
## Migration Plan

### Will Add Frontmatter (if strategy includes frontmatter)

| File | Namespace | Type | Component |
|------|-----------|------|-----------|
| docs/auth/JWT_DESIGN.md | app | architecture | auth |
| docs/CODING_STANDARDS.md | team | convention | - |
| ... | | | |

### Will Create Memories (if strategy includes memories)

| File | Type | Namespace | Component | Notes |
|------|------|-----------|-----------|-------|
| docs/auth/JWT_DESIGN.md | architecture | app | auth | Split into 2 memories |
| docs/CODING_STANDARDS.md | convention | team | - | Full doc |
| ... | | | | |

### Will Index Only (no memory, just frontmatter or skip)

These stay as searchable docs but don't become memories:
- docs/CHANGELOG.md (reference)
- docs/api/endpoints.md (reference, auto-generated)
- ...

### Will Skip (Z files)

- docs/archive/old-stuff.md (outdated)
- ...
```

## Step 6: Get Approval

Present the plan and ask:

```
I've analyzed {total} docs:
- {X} will become memories in .ragtime/
- {Y} will be indexed as reference docs
- {Z} will be skipped

Review the plan above. Should I:
1. Proceed with all
2. Let me adjust some classifications
3. Show me a specific file's analysis
4. Cancel
```

## Step 7: Execute Migration

### If adding frontmatter to originals:

For each doc that needs frontmatter, prepend:

```yaml
---
namespace: {app|team}
type: {type}
component: {if applicable}
---
```

This makes the original docs index better with `ragtime index`.

### If creating memories:

1. **Extract the key content** - Don't copy the whole doc verbatim unless it's focused. Extract the essential knowledge.

2. **Write the memory file** to `.ragtime/app/{component}/{id}-{slug}.md` or `.ragtime/team/{id}-{slug}.md`:

```yaml
---
id: {8-char-uuid}
namespace: {app|team}
type: {type}
component: {if applicable}
confidence: medium
confidence_reason: import-docs
source: import-docs
status: active
added: {today}
migrated_from: {original-file-path}
---

{extracted content}
```

3. **For split documents**, create multiple focused memories with related IDs.

## Step 8: Reindex

After all memories are created:

```bash
ragtime reindex
```

## Step 9: Summary

Summarize based on what was done:

```
## Migration Complete

### Frontmatter Added (if applicable)
Updated {X} docs with frontmatter for better indexing.

### Memories Created (if applicable)
Created {Y} memories:
- {N} architecture in app/
- {N} conventions in team/
- {N} decisions in app/
- ...

Next steps:
- Run 'ragtime index' to update the search index
- Review memories with: ragtime memories --namespace app
- Search with: ragtime search "your query"
- Edit any misclassified content in .ragtime/ or docs/
```

## Tips

- **Don't over-migrate**: Not every doc needs to be a memory. Reference docs are fine as indexed content.
- **Preserve originals**: This creates memories FROM docs, doesn't delete originals.
- **Review component inference**: The path-based component detection may need adjustment.
- **Batch by folder**: For large doc sets, consider migrating one folder at a time.

## Example Session

```
User: /import-docs docs/

Claude: I'll analyze your docs folder...

Running: ragtime audit docs/ --json

Found 45 markdown files. Let me analyze each one...

[Analyzes docs/auth/JWT_DESIGN.md]
This is an architecture doc about JWT implementation. Key insights:
- 15-minute access token expiry
- Refresh token rotation strategy
- Why we chose asymmetric keys

I'll create 1 memory in app/auth/ for this.

[Continues for other files...]

## Migration Plan
...

Proceed? (yes/adjust/cancel)
```
