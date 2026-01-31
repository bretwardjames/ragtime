"""
ChromaDB wrapper for ragtime.

Handles storage and retrieval of indexed documents and code.
"""

from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings


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
        **filters,
    ) -> list[dict]:
        """
        Semantic search over indexed content.

        Args:
            query: Natural language search query
            limit: Max results to return
            type_filter: "code" or "docs" (None = both)
            namespace: Filter by namespace (for docs)
            **filters: Additional metadata filters

        Returns:
            List of dicts with 'content', 'metadata', 'distance'
        """
        where = {}

        if type_filter:
            where["type"] = type_filter

        if namespace:
            where["namespace"] = namespace

        for key, value in filters.items():
            where[key] = value

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where if where else None,
        )

        # Flatten results into list of dicts
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return output

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
        count = self.collection.count()

        # Count by type
        docs_count = len(self.collection.get(where={"type": "docs"})["ids"])
        code_count = len(self.collection.get(where={"type": "code"})["ids"])

        return {
            "total": count,
            "docs": docs_count,
            "code": code_count,
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
