---
description: Generate documentation from code with AI analysis
allowed-arguments: path to code (e.g., /generate-docs src/)
allowed-tools: Bash, Read, Write, Edit, AskUserQuestion, Glob, Grep
---

# Generate Docs

Analyze code and generate comprehensive documentation.

**Usage:**
- `/generate-docs src/` - Generate docs for src folder
- `/generate-docs src/auth/` - Document specific module
- `/generate-docs` - Interactive mode

## Overview

You (Claude) will:
1. Read and understand the code
2. Write clear, helpful documentation
3. Include examples and usage notes
4. Output as markdown files

This is the AI-powered version. For quick stubs without AI, use `ragtime generate --stubs`.

## Step 1: Get the Code Path

**If `$ARGUMENTS` provided:**
- Use it as the code path

**If no arguments:**
- Ask: "What code should I document? (e.g., src/, src/auth/, specific file)"

## Step 2: Discover Code Files

```bash
# Find code files
find {path} -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" \) \
  -not -path "*/__pycache__/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/.venv/*"
```

If many files found, ask:
```
Found {count} files. Options:

1. Document all
2. Document specific folder
3. Document specific files (I'll list them)
```

## Step 3: Choose Output Location

```
Where should I save the documentation?

1. **docs/api/** - API reference docs
2. **docs/code/** - Code documentation
3. **.ragtime/app/** - As searchable memories
4. **Custom path**
```

## Step 4: Analyze and Document Each File

For each code file:

1. **Read the full file**
2. **Understand what it does** - purpose, patterns, relationships
3. **Write documentation** that explains:
   - What the module/class/function does
   - Why it exists (context)
   - How to use it (examples)
   - Important details (edge cases, requirements)

### Documentation Quality Guidelines

**Good documentation answers:**
- What does this do?
- Why would I use it?
- How do I use it?
- What should I watch out for?

**Avoid:**
- Just restating the function name ("getUserById gets a user by ID")
- Obvious descriptions
- Missing the "why"

### Documentation Template

```markdown
# {Module Name}

> **File:** `{path}`

## Overview

{2-3 sentences explaining what this module does and why it exists}

## Quick Start

\`\`\`{language}
{Simple usage example}
\`\`\`

---

## API Reference

### `ClassName`

{Description of the class - what it represents, when to use it}

#### Constructor

\`\`\`{language}
{constructor signature}
\`\`\`

| Parameter | Type | Description |
|-----------|------|-------------|
| `param` | `Type` | {Actual description} |

#### Methods

##### `methodName(params) -> ReturnType`

{What this method does, not just restating the name}

**Parameters:**
- `param` (`Type`): {Description}

**Returns:** {What it returns and when}

**Example:**
\`\`\`{language}
{Practical example}
\`\`\`

**Notes:**
- {Important edge cases}
- {Performance considerations}
- {Related methods}

---

## Functions

### `functionName(params) -> ReturnType`

{Description}

{Parameters, returns, example - same format as methods}

---

## Constants / Configuration

| Name | Value | Description |
|------|-------|-------------|
| `CONSTANT` | `value` | {What it's for} |

---

## See Also

- [{Related Module}]({link})
- [{External Docs}]({link})
```

## Step 5: Add Frontmatter (if .ragtime/ output)

```yaml
---
namespace: app
type: architecture
component: {from path}
source: generate-docs
confidence: medium
confidence_reason: ai-generated
---
```

## Step 6: Write and Report

```
Generating documentation...

  ✓ src/auth/jwt.py → docs/api/auth/jwt.md
    - JWTManager class
    - create_token, validate_token functions

  ✓ src/auth/sessions.py → docs/api/auth/sessions.md
    - SessionStore class
    - Redis integration details

  ✓ src/db/models.py → docs/api/db/models.md
    - User, Claim, Shift models
    - Relationships documented
```

## Step 7: Summary

```
───────────────────────────────────────────
✅ DOCUMENTATION COMPLETE
───────────────────────────────────────────

Generated {count} documentation files

Coverage:
  - {n} classes documented
  - {n} functions documented
  - {n} with examples

Output: {output_path}/

Next steps:
  - Review the generated docs
  - Add any missing context
  - Run 'ragtime index' to make searchable
```

## Best Practices

When writing docs:

1. **Start with the "why"** - Don't just describe what, explain purpose
2. **Include real examples** - Show actual usage, not abstract patterns
3. **Document edge cases** - What happens with null? Empty arrays?
4. **Link related concepts** - Help readers navigate
5. **Keep it scannable** - Use headers, tables, code blocks

## Example

For this code:

```python
class RateLimiter:
    """Rate limiting using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()

    def allow(self, cost: int = 1) -> bool:
        """Check if request is allowed, consuming tokens if so."""
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
```

Generate:

```markdown
# RateLimiter

> **File:** `src/middleware/rate_limit.py`

## Overview

Implements rate limiting using the token bucket algorithm. Use this to protect
APIs from abuse by limiting how many requests a client can make per minute.

## Quick Start

\`\`\`python
limiter = RateLimiter(requests_per_minute=100)

@app.before_request
def check_rate_limit():
    if not limiter.allow():
        return {"error": "Rate limit exceeded"}, 429
\`\`\`

---

## API Reference

### `RateLimiter`

A token bucket rate limiter. Tokens refill continuously over time, allowing
for burst traffic while maintaining an average rate limit.

#### Constructor

\`\`\`python
RateLimiter(requests_per_minute: int = 60)
\`\`\`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `requests_per_minute` | `int` | `60` | Maximum requests allowed per minute |

#### Methods

##### `allow(cost: int = 1) -> bool`

Check if a request should be allowed. If allowed, consumes tokens from the bucket.

**Parameters:**
- `cost` (`int`): Number of tokens this request costs. Use higher values for
  expensive operations.

**Returns:** `True` if the request is allowed, `False` if rate limited.

**Example:**
\`\`\`python
# Normal request
if limiter.allow():
    process_request()

# Expensive operation (costs 5 tokens)
if limiter.allow(cost=5):
    run_expensive_query()
\`\`\`

**Notes:**
- Tokens refill continuously, not all at once
- Thread-safe for single-process use
- For distributed systems, use Redis-backed rate limiting instead
```
