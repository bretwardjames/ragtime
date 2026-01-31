---
description: Capture ad-hoc knowledge to local memory mid-session
allowed-arguments: the thing to remember (e.g., /remember auth uses 15-min JWT expiry)
allowed-tools: Bash, AskUserQuestion, mcp__ragtime__remember
---

# Remember

Capture ad-hoc knowledge to the local memory system during a session.

**Usage:**
- `/remember auth uses 15-minute JWT expiry` - Quick capture with content
- `/remember` - Interactive mode, asks what to remember

<!-- ═══════════════════════════════════════════════════════════════════════════
     CUSTOMIZABLE: Modify prompts, add project-specific types, adjust the
     approval flow, etc. Everything above the RAGTIME CORE section can be
     tailored to your project's needs.
     ═══════════════════════════════════════════════════════════════════════════ -->

## When to Use

Use `/remember` when you discover something worth preserving:
- Architecture patterns ("the validation pipeline works like X")
- Gotchas ("don't use Y because Z")
- Decisions made during discussion
- Team preferences discovered
- Anything you'd want future sessions to know

**Don't use for:**
- Temporary debug notes (those belong in handoff.md)
- Personal todos (use GitHub issues)
- Things that will be captured by `/pr-graduate` anyway

## Step 1: Capture the Knowledge

**If `$ARGUMENTS` provided:**
- Use it as the memory content

**If no arguments:**
- Ask: "What would you like me to remember?"

## Step 2: Determine Memory Type

Ask the user:

```
What type of knowledge is this?

1. 🏗️ **Architecture** - How something works in the codebase
2. 📋 **Convention** - Team rule or standard practice
3. ⚠️ **Gotcha** - Warning or "don't do this"
4. 🎯 **Decision** - Why we chose X over Y
5. 💡 **Pattern** - Reusable approach for common problem
```

<!-- CUSTOMIZABLE: Add your own types here if needed -->

## Step 3: Determine Scope

```
Where does this knowledge apply?

1. 🌐 **Global** - Applies everywhere
2. 📦 **Component** - Specific area (auth, claims, shifts, etc.)
3. 📄 **File** - Specific file or folder
```

If Component or File, ask which one.

<!-- CUSTOMIZABLE: Add your own components here -->

## Step 4: Determine Confidence

```
How confident are you in this knowledge?

1. ✅ **High** - Verified, tested, or from authoritative source
2. ⚡ **Medium** - Observed it working, but not extensively tested
3. 🤔 **Low** - Best guess or inference, should be validated
```

## Step 5: Determine Namespace

Based on the memory type, suggest the appropriate namespace:

| Type | Default Namespace |
|------|------------------|
| Architecture | `app` |
| Convention | `team` |
| Gotcha | `app` |
| Decision | `app` (or `branch-{branch}` if WIP) |
| Pattern | `app` |

Ask: "Store in `{suggested namespace}`? Or somewhere else?"

<!-- CUSTOMIZABLE: Adjust routing rules for your project -->

## Step 6: Confirm and Store

Present the memory for confirmation:

```
## Memory to Store

**Content:** {the knowledge}
**Type:** {type}
**Namespace:** {namespace}
**Confidence:** {confidence}
**Scope:** {scope}
**Component:** {if applicable}

Store this? (yes/edit/cancel)
```

<!-- ═══════════════════════════════════════════════════════════════════════════
     RAGTIME CORE - DO NOT MODIFY
     These commands must match ragtime's expected format.
     Changing these may break storage/retrieval.
     ═══════════════════════════════════════════════════════════════════════════ -->

Once confirmed, store using ragtime MCP:

```
mcp__ragtime__remember:
  content: "{the knowledge}"
  namespace: "{namespace}"
  type: "{architecture|convention|decision|pattern}"
  component: "{component if applicable}"
  confidence: "{high|medium|low}"
  confidence_reason: "manual"
  source: "remember"
```

<!-- ═══════════════════════════════════════════════════════════════════════════ -->

## Step 7: Confirm Storage

```
✅ Remembered!

"{abbreviated content}..."

Stored to: {namespace}
Confidence: {confidence}
Query with: ragtime search "{topic}" --namespace {namespace}
```

## Quick Mode

For rapid capture, you can specify everything inline:

```
/remember [high] auth uses 15-min JWT expiry #auth #architecture
```

Parsing:
- `[high]` → confidence
- `#auth` → component
- `#architecture` → type
- Rest → content

## Notes

- Memories stored via `/remember` get `source: "remember"` and default to medium confidence unless specified
- If on a feature branch, consider whether this is branch-specific or general knowledge
- For decisions that might be reversed, use low confidence
- Team conventions should be discussed before storing to `team` namespace
