"""
Code indexer - extracts functions, classes, and types from source files.

Parses code to create searchable chunks for each meaningful unit (function, class, etc).
This allows searching for specific code constructs like "useAsyncState" or "JWTManager".
"""

import ast
import re
from fnmatch import fnmatch
from pathlib import Path
from dataclasses import dataclass


# Language file extensions
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "vue": [".vue"],
    "dart": [".dart"],
}


@dataclass
class CodeEntry:
    """A parsed code symbol ready for indexing."""
    content: str           # The actual code + context
    file_path: str         # Full path to file
    language: str          # python, typescript, etc.
    symbol_name: str       # Function/class/component name
    symbol_type: str       # function, class, interface, component, etc.
    line_number: int       # Line where symbol starts
    docstring: str | None = None  # Extracted docstring/JSDoc

    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict."""
        return {
            "type": "code",
            "file": self.file_path,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "line": self.line_number,
        }


def get_extensions_for_languages(languages: list[str]) -> list[str]:
    """Get file extensions for the specified languages."""
    extensions = []
    for lang in languages:
        extensions.extend(LANGUAGE_EXTENSIONS.get(lang, []))
    return extensions


def discover_code_files(
    root: Path,
    languages: list[str],
    exclude: list[str] | None = None,
) -> list[Path]:
    """
    Find all code files to index.

    Args:
        root: Directory to search
        languages: List of languages to include
        exclude: Patterns to exclude
    """
    exclude = exclude or [
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/venv/**",
        "**/.dart_tool/**",
    ]

    extensions = get_extensions_for_languages(languages)
    files = []

    for ext in extensions:
        for path in root.rglob(f"*{ext}"):
            if path.is_file():
                # Check exclusions using proper glob matching
                skip = False
                # Use relative path for matching to avoid absolute path issues
                try:
                    rel_path = str(path.relative_to(root))
                except ValueError:
                    rel_path = str(path)

                for ex in exclude:
                    # Handle ** patterns by checking if pattern appears in path
                    if "**" in ex:
                        # Convert glob to a simpler check: **/node_modules/** means
                        # any path containing /node_modules/ segment
                        core_pattern = ex.replace("**", "").strip("/")
                        if core_pattern and f"/{core_pattern}/" in f"/{rel_path}/":
                            skip = True
                            break
                    elif fnmatch(rel_path, ex) or fnmatch(path.name, ex):
                        skip = True
                        break
                if not skip:
                    files.append(path)

    return files


def index_python_file(file_path: Path, content: str) -> list[CodeEntry]:
    """Extract code entries from a Python file."""
    entries = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return entries

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get class code (signature + docstring + method signatures)
            start_line = node.lineno
            docstring = ast.get_docstring(node) or ""

            # Build a summary of the class
            method_names = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(item.name)

            class_summary = f"class {node.name}:\n"
            if docstring:
                class_summary += f'    """{docstring}"""\n'
            if method_names:
                class_summary += f"\n    # Methods: {', '.join(method_names)}\n"

            entries.append(CodeEntry(
                content=class_summary,
                file_path=str(file_path),
                language="python",
                symbol_name=node.name,
                symbol_type="class",
                line_number=start_line,
                docstring=docstring,
            ))

            # Also index public methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue

                    method_doc = ast.get_docstring(item) or ""
                    async_prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""

                    # Get signature
                    args = []
                    for arg in item.args.args:
                        if arg.arg == "self":
                            continue
                        type_hint = ""
                        if arg.annotation:
                            try:
                                type_hint = f": {ast.unparse(arg.annotation)}"
                            except Exception:
                                pass
                        args.append(f"{arg.arg}{type_hint}")

                    ret_type = ""
                    if item.returns:
                        try:
                            ret_type = f" -> {ast.unparse(item.returns)}"
                        except Exception:
                            pass

                    method_sig = f"{async_prefix}def {item.name}({', '.join(args)}){ret_type}"
                    method_content = f"class {node.name}:\n    {method_sig}:\n"
                    if method_doc:
                        method_content += f'        """{method_doc}"""\n'

                    entries.append(CodeEntry(
                        content=method_content,
                        file_path=str(file_path),
                        language="python",
                        symbol_name=f"{node.name}.{item.name}",
                        symbol_type="method",
                        line_number=item.lineno,
                        docstring=method_doc,
                    ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level function
            if hasattr(node, 'col_offset') and node.col_offset > 0:
                continue  # Skip nested functions

            docstring = ast.get_docstring(node) or ""
            async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""

            # Get signature
            args = []
            for arg in node.args.args:
                type_hint = ""
                if arg.annotation:
                    try:
                        type_hint = f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        pass
                args.append(f"{arg.arg}{type_hint}")

            ret_type = ""
            if node.returns:
                try:
                    ret_type = f" -> {ast.unparse(node.returns)}"
                except Exception:
                    pass

            func_sig = f"{async_prefix}def {node.name}({', '.join(args)}){ret_type}"
            func_content = f"{func_sig}:\n"
            if docstring:
                func_content += f'    """{docstring}"""\n'

            entries.append(CodeEntry(
                content=func_content,
                file_path=str(file_path),
                language="python",
                symbol_name=node.name,
                symbol_type="function",
                line_number=node.lineno,
                docstring=docstring,
            ))

    return entries


def index_typescript_file(file_path: Path, content: str) -> list[CodeEntry]:
    """Extract code entries from a TypeScript/JavaScript file."""
    entries = []
    lines = content.split("\n")

    # Patterns for different constructs
    patterns = [
        # Exported functions
        (r'export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*:\s*([^\{]+))?',
         "function"),
        # Arrow function exports
        (r'export\s+const\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>',
         "function"),
        # Class exports
        (r'export\s+(?:default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?',
         "class"),
        # Interface exports
        (r'export\s+(?:default\s+)?interface\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+([^{]+))?',
         "interface"),
        # Type exports
        (r'export\s+type\s+(\w+)(?:<[^>]+>)?\s*=',
         "type"),
        # Const exports (useful for config objects, composables, etc.)
        (r'export\s+const\s+(\w+)\s*(?::\s*([^=]+))?\s*=\s*(?!.*=>)',
         "constant"),
    ]

    for i, line in enumerate(lines):
        for pattern, symbol_type in patterns:
            match = re.match(pattern, line.strip())
            if match:
                symbol_name = match.group(1)

                # Get context (a few lines around the definition)
                start = max(0, i - 1)
                end = min(len(lines), i + 10)
                context_lines = lines[start:end]

                # Extract JSDoc if present
                jsdoc = ""
                if i > 0 and lines[i - 1].strip().endswith("*/"):
                    # Look backward for JSDoc start
                    for j in range(i - 1, max(0, i - 20), -1):
                        if "/**" in lines[j]:
                            jsdoc_lines = lines[j:i]
                            jsdoc = "\n".join(jsdoc_lines)
                            break

                entries.append(CodeEntry(
                    content="\n".join(context_lines),
                    file_path=str(file_path),
                    language="typescript" if file_path.suffix in [".ts", ".tsx"] else "javascript",
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    line_number=i + 1,
                    docstring=jsdoc if jsdoc else None,
                ))
                break

    # Also look for Vue composables pattern (useXxx functions)
    composable_pattern = r'(?:export\s+)?(?:const|function)\s+(use[A-Z]\w*)'
    for i, line in enumerate(lines):
        match = re.search(composable_pattern, line)
        if match:
            symbol_name = match.group(1)
            # Check if we already indexed this
            if not any(e.symbol_name == symbol_name for e in entries):
                start = max(0, i - 1)
                end = min(len(lines), i + 15)

                entries.append(CodeEntry(
                    content="\n".join(lines[start:end]),
                    file_path=str(file_path),
                    language="typescript" if file_path.suffix in [".ts", ".tsx"] else "javascript",
                    symbol_name=symbol_name,
                    symbol_type="composable",
                    line_number=i + 1,
                ))

    return entries


def index_vue_file(file_path: Path, content: str) -> list[CodeEntry]:
    """Extract code entries from a Vue SFC file."""
    entries = []

    # Get component name from filename
    component_name = file_path.stem

    # Extract script section
    script_match = re.search(
        r'<script[^>]*(?:setup)?[^>]*>(.*?)</script>',
        content,
        re.DOTALL | re.IGNORECASE
    )

    script_content = script_match.group(1) if script_match else ""

    # Add the component itself
    entries.append(CodeEntry(
        content=f"Vue Component: {component_name}\n\n{script_content[:500]}",
        file_path=str(file_path),
        language="vue",
        symbol_name=component_name,
        symbol_type="component",
        line_number=1,
    ))

    # If there's script content, parse it for composables and functions
    if script_content:
        # Look for composable usage (useXxx calls)
        composable_usages = re.findall(r'(use[A-Z]\w*)\s*\(', script_content)
        for composable in set(composable_usages):
            # Find the line
            for i, line in enumerate(content.split("\n")):
                if composable in line:
                    entries.append(CodeEntry(
                        content=f"Uses composable: {composable}\n{line.strip()}",
                        file_path=str(file_path),
                        language="vue",
                        symbol_name=f"{component_name}:{composable}",
                        symbol_type="composable_usage",
                        line_number=i + 1,
                    ))
                    break

        # Parse script for functions
        ts_entries = index_typescript_file(file_path, script_content)
        for entry in ts_entries:
            entry.language = "vue"
            entry.symbol_name = f"{component_name}.{entry.symbol_name}"
            entries.append(entry)

    return entries


def index_dart_file(file_path: Path, content: str) -> list[CodeEntry]:
    """Extract code entries from a Dart file."""
    entries = []
    lines = content.split("\n")

    # Patterns for Dart constructs
    patterns = [
        # Class definitions
        (r'(?:abstract\s+)?class\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+(\w+))?(?:\s+with\s+([^{]+))?(?:\s+implements\s+([^{]+))?',
         "class"),
        # Function definitions
        (r'(?:Future<[^>]+>|void|int|String|bool|double|dynamic|\w+)\s+(\w+)\s*(?:<[^>]+>)?\s*\(',
         "function"),
        # Mixins
        (r'mixin\s+(\w+)(?:\s+on\s+(\w+))?',
         "mixin"),
        # Extensions
        (r'extension\s+(\w+)\s+on\s+(\w+)',
         "extension"),
    ]

    for i, line in enumerate(lines):
        for pattern, symbol_type in patterns:
            match = re.match(r'\s*' + pattern, line)
            if match:
                symbol_name = match.group(1)

                # Get context
                start = max(0, i - 1)
                end = min(len(lines), i + 10)

                # Extract doc comment if present
                doc_comment = ""
                if i > 0:
                    for j in range(i - 1, max(0, i - 20), -1):
                        if lines[j].strip().startswith("///"):
                            doc_comment = lines[j].strip() + "\n" + doc_comment
                        elif lines[j].strip():
                            break

                entries.append(CodeEntry(
                    content="\n".join(lines[start:end]),
                    file_path=str(file_path),
                    language="dart",
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                    line_number=i + 1,
                    docstring=doc_comment if doc_comment else None,
                ))
                break

    return entries


def index_file(file_path: Path) -> list[CodeEntry]:
    """
    Parse a single code file into CodeEntry objects.

    Returns empty list if file can't be parsed.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError):
        return []

    # Skip empty files
    if not content.strip():
        return []

    suffix = file_path.suffix.lower()

    if suffix == ".py":
        return index_python_file(file_path, content)
    elif suffix in [".ts", ".tsx", ".js", ".jsx"]:
        return index_typescript_file(file_path, content)
    elif suffix == ".vue":
        return index_vue_file(file_path, content)
    elif suffix == ".dart":
        return index_dart_file(file_path, content)

    return []


def index_directory(
    root: Path,
    languages: list[str],
    exclude: list[str] | None = None,
) -> list[CodeEntry]:
    """
    Index all code files in a directory.

    Returns list of CodeEntry objects ready for vector DB.
    """
    files = discover_code_files(root, languages, exclude)
    entries = []

    for file_path in files:
        file_entries = index_file(file_path)
        entries.extend(file_entries)

    return entries
