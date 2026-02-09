"""
ChromaDB wrapper for ragtime.

Handles storage and retrieval of indexed documents and code.
"""

import re
from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings


def extract_query_hints(query: str, known_components: list[str] | None = None) -> tuple[str, list[str]]:
    """
    Extract component/scope hints from a query for hybrid search.

    Detects patterns like "X in mobile", "mobile X", "X for auth" and extracts
    the qualifier to use as require_terms. This prevents qualifiers from being
    diluted in semantic search.

    Args:
        query: The natural language search query
        known_components: Optional list of known component names to detect

    Returns:
        (cleaned_query, extracted_terms) - query with hints removed, terms to require
    """
    # Default known components/scopes (common patterns)
    default_components = [
        # Platforms
        "mobile", "web", "desktop", "ios", "android", "flutter", "react", "vue",
        # Languages
        "dart", "python", "typescript", "javascript", "ts", "js", "py",
        # Common components
        "auth", "authentication", "api", "database", "db", "ui", "frontend", "backend",
        "server", "client", "admin", "user", "payment", "billing", "notification",
        "email", "cache", "queue", "worker", "scheduler", "logging", "metrics",
    ]

    components = set(c.lower() for c in (known_components or default_components))
    extracted = []
    cleaned = query

    # Pattern 1: "X in/for/on {component}" - extract component
    patterns = [
        r'\b(?:in|for|on|from|using|with)\s+(?:the\s+)?(\w+)\s*(?:app|code|module|service|codebase)?(?:\s|$)',
        r'\b(\w+)\s+(?:app|code|module|service|codebase)\b',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            word = match.group(1).lower()
            if word in components:
                extracted.append(word)
                # Remove the matched phrase from query
                cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]

    # Pattern 2: Check if any known component appears as standalone word
    words = re.findall(r'\b\w+\b', query.lower())
    for word in words:
        if word in components and word not in extracted:
            # Only extract if it looks like a qualifier (not the main subject)
            # Heuristic: if query has other meaningful words, it's likely a qualifier
            other_words = [w for w in words if w != word and len(w) > 3]
            if len(other_words) >= 2:
                extracted.append(word)

    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned, list(set(extracted))


def _vector_summary(result: dict) -> str:
    """Summarize a vector search result as a pointer instead of full content.

    For code: "function handleCheckIn — shift check-in handler with GPS validation"
    For docs: "Installation > Configuration — environment variables and setup"
    """
    meta = result.get("metadata", {})
    result_type = meta.get("type", "")
    content = result.get("content", "")

    if result_type == "code":
        symbol_type = meta.get("symbol_type", "symbol")
        symbol_name = meta.get("symbol_name", "")
        # Use first line of content as context (usually the signature)
        first_line = content.split("\n", 1)[0].strip()
        if symbol_name:
            return f"Matched {symbol_type}: {symbol_name} — {first_line}"
        return f"Matched {symbol_type} — {first_line}"

    elif result_type == "docs":
        section = meta.get("section_path", "")
        title = meta.get("title", "")
        # First sentence of content as context
        first_sentence = content.split("\n", 1)[0].strip()
        if not first_sentence or first_sentence.startswith("#"):
            # Skip headers, grab next non-empty line
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    first_sentence = line
                    break
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117].rsplit(" ", 1)[0] + "..."
        label = section or title or "doc"
        return f"Matched doc: {label} — {first_sentence}"

    else:
        # Memory or other type
        preview = content[:120].replace("\n", " ").strip()
        if len(preview) == 120:
            preview = preview.rsplit(" ", 1)[0] + "..."
        return f"Matched {result_type}: {preview}"


class RagtimeDB:
    """Vector database for ragtime indexes."""

    def __init__(self, path: Path):
        """
        Initialize the database.

        Args:
            path: Directory to store ChromaDB data
        """
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="ragtime",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add documents to the index.

        Args:
            ids: Unique identifiers for each document
            documents: Text content to embed and index
            metadatas: Metadata dicts for filtering
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def update(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Update existing documents."""
        self.collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update documents."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        type_filter: str | None = None,
        namespace: str | None = None,
        status: str | list[str] | None = None,
        require_terms: list[str] | None = None,
        auto_extract: bool = True,
        **filters,
    ) -> list[dict]:
        """
        Hybrid search: semantic similarity + keyword filtering.

        Args:
            query: Natural language search query
            limit: Max results to return
            type_filter: "code" or "docs" (None = both)
            namespace: Filter by namespace (for docs)
            status: Filter by status - "ephemeral", "permanent", or list of both.
                   None means no filtering (returns all). For convention checks,
                   use status="permanent" to only match merged code.
            require_terms: List of terms that MUST appear in results (case-insensitive).
                          Use for scoped queries like "error handling in mobile" with
                          require_terms=["mobile"] to ensure "mobile" isn't ignored.
            auto_extract: If True (default), automatically detect component qualifiers
                         in the query and add them to require_terms. Set to False
                         for raw/literal search.
            **filters: Additional metadata filters (None values are ignored)

        Returns:
            List of dicts with 'content', 'metadata', 'distance'
        """
        # Auto-extract component hints from query if enabled
        search_query = query
        all_require_terms = list(require_terms) if require_terms else []

        if auto_extract:
            cleaned_query, extracted = extract_query_hints(query)
            if extracted:
                # Use cleaned query for embedding (removes noise)
                search_query = cleaned_query
                # Add extracted terms to require_terms
                all_require_terms.extend(extracted)
                all_require_terms = list(set(all_require_terms))  # dedupe
        # Build list of filter conditions, excluding None values
        conditions = []

        if type_filter:
            conditions.append({"type": type_filter})

        if namespace:
            conditions.append({"namespace": namespace})

        if status:
            if isinstance(status, list):
                conditions.append({"status": {"$in": status}})
            else:
                conditions.append({"status": status})

        # Add any additional filters, but skip None values
        for key, value in filters.items():
            if value is not None:
                conditions.append({key: value})

        # ChromaDB requires $and for multiple conditions
        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

        # When using require_terms, fetch more results since we'll filter some out
        fetch_limit = limit * 5 if all_require_terms else limit

        results = self.collection.query(
            query_texts=[search_query],
            n_results=fetch_limit,
            where=where,
        )

        # Flatten results into list of dicts
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # Hybrid filtering: ensure required terms appear
                if all_require_terms:
                    doc_lower = doc.lower()
                    # Also check file path in metadata for code/file matches
                    file_path = (results["metadatas"][0][i].get("file", "") or "").lower()
                    combined_text = f"{doc_lower} {file_path}"

                    if not all(term.lower() in combined_text for term in all_require_terms):
                        continue

                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

                # Stop once we have enough
                if len(output) >= limit:
                    break

        return output

    def search_tiered(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        require_terms: list[str] | None = None,
        auto_extract: bool = True,
        **filters,
    ) -> list[dict]:
        """
        Tiered search: prioritizes memories > docs > code.

        Searches in priority order, filling up to limit:
        1. Memories (curated, high-signal knowledge)
        2. Documentation (indexed markdown)
        3. Code (broadest, implementation details)

        Args:
            query: Natural language search query
            limit: Max total results to return
            namespace: Filter by namespace
            require_terms: Terms that MUST appear in results
            auto_extract: Auto-detect qualifiers from query
            **filters: Additional metadata filters

        Returns:
            List of dicts with 'content', 'metadata', 'distance', 'tier'
        """
        results = []

        # Tier 1: Memories (not docs or code)
        memory_results = self._search_tier(
            query=query,
            tier_name="memory",
            exclude_types=["docs", "code"],
            limit=limit,
            namespace=namespace,
            require_terms=require_terms,
            auto_extract=auto_extract,
            **filters,
        )
        results.extend(memory_results)

        # Tier 2: Documentation
        if len(results) < limit:
            doc_results = self._search_tier(
                query=query,
                tier_name="docs",
                type_filter="docs",
                limit=limit - len(results),
                namespace=namespace,
                require_terms=require_terms,
                auto_extract=auto_extract,
                **filters,
            )
            results.extend(doc_results)

        # Tier 3: Code
        if len(results) < limit:
            code_results = self._search_tier(
                query=query,
                tier_name="code",
                type_filter="code",
                limit=limit - len(results),
                namespace=namespace,
                require_terms=require_terms,
                auto_extract=auto_extract,
                **filters,
            )
            results.extend(code_results)

        return results

    def _search_tier(
        self,
        query: str,
        tier_name: str,
        limit: int,
        type_filter: str | None = None,
        exclude_types: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Search a single tier and tag results."""
        # Build where clause for exclusion if needed
        if exclude_types:
            # Search without type filter, then exclude in post-processing
            results = self.search(
                query=query,
                limit=limit * 2,  # fetch more since we'll filter
                type_filter=None,
                **kwargs,
            )
            # Filter out excluded types
            filtered = []
            for r in results:
                if r["metadata"].get("type") not in exclude_types:
                    r["tier"] = tier_name
                    filtered.append(r)
                    if len(filtered) >= limit:
                        break
            return filtered
        else:
            results = self.search(
                query=query,
                limit=limit,
                type_filter=type_filter,
                **kwargs,
            )
            for r in results:
                r["tier"] = tier_name
            return results

    def search_narrative(
        self,
        query: str,
        project_path: "Path",
        narrative_config: "NarrativeConfig",
        limit: int = 10,
        **vector_kwargs,
    ) -> list[dict]:
        """
        Narrative search: manifest routing first, vector fallback.

        Consults the narrative manifest index for full-document matches,
        then fills remaining slots with vector search if configured.

        Args:
            query: Natural language search query
            project_path: Root of the project (for reading docs)
            narrative_config: Narrative configuration
            limit: Max total results
            **vector_kwargs: Passed through to vector search() for fallback

        Returns:
            List of result dicts with 'retrieval_method' field
        """
        from .narrative import parse_manifest, match_query, read_full_doc

        results = []
        manifest_path = project_path / narrative_config.index_file

        # Layer 1: Manifest matching
        if manifest_path.exists():
            try:
                manifest_content = manifest_path.read_text()
                entries = parse_manifest(manifest_content)
                matches = match_query(query, entries, max_matches=min(limit, 3))

                for entry in matches:
                    if narrative_config.serve_full_docs:
                        content = read_full_doc(
                            project_path, entry.file_path,
                            max_tokens=narrative_config.max_doc_tokens,
                        )
                    else:
                        content = entry.scope

                    if content:
                        results.append({
                            "content": content,
                            "metadata": {
                                "file": entry.file_path,
                                "type": "docs",
                                "narrative_scope": entry.scope,
                            },
                            "distance": 0.0,
                            "retrieval_method": "narrative",
                        })

                    if len(results) >= limit:
                        break
            except (IOError, UnicodeDecodeError):
                pass

        # Layer 2: Vector fallback — return pointers (file path + match summary)
        # instead of full chunks, letting the caller decide what to read
        remaining = limit - len(results)
        if remaining > 0 and narrative_config.fallback_to_vector:
            vector_results = self.search(query=query, limit=remaining, **vector_kwargs)
            # Dedupe: skip vector results for files already in narrative results
            narrative_files = {r["metadata"].get("file") for r in results}
            for r in vector_results:
                if r["metadata"].get("file") not in narrative_files:
                    r["retrieval_method"] = "vector"
                    r["content"] = _vector_summary(r)
                    results.append(r)
                    if len(results) >= limit:
                        break

        return results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    def clear(self, type_filter: str | None = None) -> None:
        """
        Clear the index.

        Args:
            type_filter: Only clear "code" or "docs" (None = everything)
        """
        if type_filter:
            # Get all IDs matching the type
            results = self.collection.get(where={"type": type_filter})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        else:
            # Nuclear option - recreate collection
            self.client.delete_collection("ragtime")
            self.collection = self.client.get_or_create_collection(
                name="ragtime",
                metadata={"hnsw:space": "cosine"},
            )

    def stats(self) -> dict:
        """Get index statistics."""
        try:
            count = self.collection.count()

            # Count by type - only retrieve IDs, not full documents
            docs_result = self.collection.get(where={"type": "docs"}, include=[])
            code_result = self.collection.get(where={"type": "code"}, include=[])

            docs_count = len(docs_result["ids"])
            code_count = len(code_result["ids"])

            return {
                "total": count,
                "docs": docs_count,
                "code": code_count,
            }
        except Exception:
            # Return zeros if collection is corrupted or unavailable
            return {
                "total": 0,
                "docs": 0,
                "code": 0,
            }

    def get_indexed_files(self, type_filter: str | None = None) -> dict[str, float]:
        """
        Get all indexed files and their modification times.

        Args:
            type_filter: "code" or "docs" (None = both)

        Returns:
            Dict mapping file paths to their indexed mtime
        """
        where = {"type": type_filter} if type_filter else None
        results = self.collection.get(where=where, include=["metadatas"])

        files: dict[str, float] = {}
        for meta in results["metadatas"]:
            file_path = meta.get("file", "")
            mtime = meta.get("mtime", 0.0)
            # For code files, multiple entries per file - keep max mtime
            if file_path not in files or mtime > files[file_path]:
                files[file_path] = mtime

        return files

    def delete_by_file(self, file_paths: list[str], type_filter: str | None = None) -> int:
        """
        Delete all entries for the given file paths.

        Args:
            file_paths: List of file paths to remove
            type_filter: "code" or "docs" (None = both)

        Returns:
            Number of entries deleted
        """
        if not file_paths:
            return 0

        # Build where clause
        where = {"file": {"$in": file_paths}}
        if type_filter:
            where = {"$and": [{"file": {"$in": file_paths}}, {"type": type_filter}]}

        # Get IDs to delete
        results = self.collection.get(where=where)
        ids = results["ids"]

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)

    def delete_by_branch(self, branch: str) -> int:
        """
        Delete all ephemeral entries for a branch.

        Used when a branch is deleted or merged.

        Args:
            branch: Branch name to remove entries for

        Returns:
            Number of entries deleted
        """
        where = {"$and": [{"status": "ephemeral"}, {"branch": branch}]}
        results = self.collection.get(where=where)
        ids = results["ids"]

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)

    def graduate_branch(self, branch: str) -> int:
        """
        Graduate ephemeral entries from a branch to permanent.

        Called after PR merge. Updates status without re-embedding.

        Args:
            branch: Branch name to graduate

        Returns:
            Number of entries graduated
        """
        where = {"$and": [{"status": "ephemeral"}, {"branch": branch}]}
        results = self.collection.get(where=where, include=["metadatas", "documents"])

        if not results["ids"]:
            return 0

        # Update metadata to permanent
        new_metadatas = []
        for meta in results["metadatas"]:
            updated = dict(meta)
            updated["status"] = "permanent"
            # Remove branch key entirely (ChromaDB doesn't support None values)
            if "branch" in updated:
                del updated["branch"]
            new_metadatas.append(updated)

        self.collection.update(
            ids=results["ids"],
            metadatas=new_metadatas,
        )

        return len(results["ids"])

    def clear_permanent(self, type_filter: str | None = None) -> int:
        """
        Clear all permanent entries (before reindexing from main).

        Args:
            type_filter: Only clear "code" or "docs" (None = both)

        Returns:
            Number of entries cleared
        """
        conditions = [{"status": "permanent"}]
        if type_filter:
            conditions.append({"type": type_filter})

        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        results = self.collection.get(where=where)
        ids = results["ids"]

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)

    def get_ephemeral_stats(self, branch: str | None = None) -> dict:
        """
        Get statistics about ephemeral entries.

        Args:
            branch: Optional branch to filter by

        Returns:
            Dict with counts by branch
        """
        where = {"status": "ephemeral"}
        if branch:
            where = {"$and": [{"status": "ephemeral"}, {"branch": branch}]}

        results = self.collection.get(where=where, include=["metadatas"])

        # Count by branch
        by_branch: dict[str, int] = {}
        for meta in results["metadatas"]:
            b = meta.get("branch", "unknown")
            by_branch[b] = by_branch.get(b, 0) + 1

        return {
            "total": len(results["ids"]),
            "by_branch": by_branch,
        }
