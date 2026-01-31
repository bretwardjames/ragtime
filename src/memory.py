"""
Memory storage for ragtime.

Handles structured memory storage in .ragtime/ directory.
Each memory is a markdown file with YAML frontmatter.
"""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import uuid
import re
import yaml


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    namespace: str
    type: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    component: Optional[str] = None
    confidence: str = "medium"
    confidence_reason: Optional[str] = None
    source: str = "manual"
    status: str = "active"
    added: str = field(default_factory=lambda: date.today().isoformat())
    author: Optional[str] = None
    issue: Optional[str] = None
    epic: Optional[str] = None
    branch: Optional[str] = None
    supersedes: Optional[str] = None

    def to_frontmatter(self) -> dict:
        """Convert to YAML frontmatter dict."""
        data = {
            "id": self.id,
            "namespace": self.namespace,
            "type": self.type,
            "confidence": self.confidence,
            "source": self.source,
            "status": self.status,
            "added": self.added,
        }

        # Add optional fields if present
        if self.component:
            data["component"] = self.component
        if self.confidence_reason:
            data["confidence_reason"] = self.confidence_reason
        if self.author:
            data["author"] = self.author
        if self.issue:
            data["issue"] = self.issue
        if self.epic:
            data["epic"] = self.epic
        if self.branch:
            data["branch"] = self.branch
        if self.supersedes:
            data["supersedes"] = self.supersedes

        return data

    def to_markdown(self) -> str:
        """Convert to markdown with YAML frontmatter."""
        frontmatter = yaml.dump(self.to_frontmatter(), default_flow_style=False, sort_keys=False)
        return f"---\n{frontmatter}---\n\n{self.content}\n"

    def to_metadata(self) -> dict:
        """Convert to metadata dict for ChromaDB."""
        meta = self.to_frontmatter()
        meta["file"] = self.get_relative_path()
        return meta

    def get_relative_path(self) -> str:
        """Get the relative path for this memory's file."""
        slug = self._slugify(self.content[:50])

        if self.namespace == "app":
            if self.component:
                return f"app/{self.component}/{self.id}-{slug}.md"
            return f"app/{self.id}-{slug}.md"
        elif self.namespace == "team":
            return f"team/{self.id}-{slug}.md"
        elif self.namespace.startswith("user-"):
            username = self.namespace.replace("user-", "")
            return f"users/{username}/{self.id}-{slug}.md"
        elif self.namespace.startswith("branch-"):
            branch_slug = self._slugify(self.namespace.replace("branch-", ""))
            return f"branches/{branch_slug}/{self.id}-{slug}.md"
        else:
            return f"other/{self.namespace}/{self.id}-{slug}.md"

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filename-safe slug."""
        # Lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug).strip('-')
        return slug[:40]  # Limit length

    @classmethod
    def from_file(cls, path: Path) -> "Memory":
        """Parse a memory from a markdown file with YAML frontmatter."""
        text = path.read_text()

        if not text.startswith("---"):
            raise ValueError(f"No YAML frontmatter found in {path}")

        # Split frontmatter and content
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid frontmatter format in {path}")

        frontmatter = yaml.safe_load(parts[1])
        content = parts[2].strip()

        return cls(
            id=frontmatter.get("id", str(uuid.uuid4())[:8]),
            content=content,
            namespace=frontmatter.get("namespace", "app"),
            type=frontmatter.get("type", "unknown"),
            component=frontmatter.get("component"),
            confidence=frontmatter.get("confidence", "medium"),
            confidence_reason=frontmatter.get("confidence_reason"),
            source=frontmatter.get("source", "file"),
            status=frontmatter.get("status", "active"),
            added=frontmatter.get("added", date.today().isoformat()),
            author=frontmatter.get("author"),
            issue=frontmatter.get("issue"),
            epic=frontmatter.get("epic"),
            branch=frontmatter.get("branch"),
            supersedes=frontmatter.get("supersedes"),
        )


class MemoryStore:
    """
    File-based memory storage.

    Stores memories as markdown files with YAML frontmatter.
    Also maintains a ChromaDB index for semantic search.
    """

    def __init__(self, project_path: Path, db):
        """
        Initialize the memory store.

        Args:
            project_path: Root of the project
            db: RagtimeDB instance for vector search
        """
        self.project_path = project_path
        self.memory_dir = project_path / ".ragtime"
        self.db = db

    def save(self, memory: Memory) -> Path:
        """
        Save a memory to disk and index it.

        Returns:
            Path to the saved file
        """
        # Determine file path
        relative_path = memory.get_relative_path()
        file_path = self.memory_dir / relative_path

        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(memory.to_markdown())

        # Index in ChromaDB
        self.db.upsert(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[memory.to_metadata()],
        )

        return file_path

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        # Search in ChromaDB to find the file
        results = self.db.collection.get(ids=[memory_id])

        if not results["ids"]:
            return None

        metadata = results["metadatas"][0]
        file_rel_path = metadata.get("file", "")

        if not file_rel_path:
            return None

        file_path = self.memory_dir / file_rel_path

        if file_path.exists():
            return Memory.from_file(file_path)

        return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        memory = self.get(memory_id)
        if not memory:
            return False

        # Delete file
        file_path = self.memory_dir / memory.get_relative_path()
        if file_path.exists():
            file_path.unlink()

            # Clean up empty directories
            self._cleanup_empty_dirs(file_path.parent)

        # Remove from index
        self.db.delete([memory_id])

        return True

    def update_status(self, memory_id: str, status: str) -> bool:
        """Update a memory's status (e.g., graduated, abandoned)."""
        memory = self.get(memory_id)
        if not memory:
            return False

        memory.status = status
        self.save(memory)
        return True

    def graduate(self, memory_id: str, new_confidence: str = "high") -> Optional[Memory]:
        """
        Graduate a branch memory to app namespace.

        Creates a copy in app namespace and marks original as graduated.
        """
        memory = self.get(memory_id)
        if not memory:
            return None

        if not memory.namespace.startswith("branch-"):
            raise ValueError("Can only graduate branch memories")

        # Create graduated copy
        graduated = Memory(
            content=memory.content,
            namespace="app",
            type=memory.type,
            component=memory.component,
            confidence=new_confidence,
            confidence_reason="pr-graduate",
            source="pr-graduate",
            status="active",
            author=memory.author,
            issue=memory.issue,
            epic=memory.epic,
        )

        # Save the graduated memory
        self.save(graduated)

        # Mark original as graduated
        memory.status = "graduated"
        self.save(memory)

        return graduated

    def list_memories(
        self,
        namespace: Optional[str] = None,
        type_filter: Optional[str] = None,
        status: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> list[Memory]:
        """List memories with optional filters."""
        where = {}

        if namespace:
            if namespace.endswith("*"):
                # Prefix match - ChromaDB doesn't support this directly
                # We'll filter in Python
                pass
            else:
                where["namespace"] = namespace

        if type_filter:
            where["type"] = type_filter

        if status:
            where["status"] = status

        if component:
            where["component"] = component

        # Get from ChromaDB
        results = self.db.collection.get(
            where=where if where else None,
            limit=limit,
        )

        memories = []
        for i, mem_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            content = results["documents"][i] if results["documents"] else ""

            # Handle namespace prefix filtering
            if namespace and namespace.endswith("*"):
                prefix = namespace[:-1]
                if not metadata.get("namespace", "").startswith(prefix):
                    continue

            memories.append(Memory(
                id=mem_id,
                content=content,
                namespace=metadata.get("namespace", "unknown"),
                type=metadata.get("type", "unknown"),
                component=metadata.get("component"),
                confidence=metadata.get("confidence", "medium"),
                source=metadata.get("source", "unknown"),
                status=metadata.get("status", "active"),
                added=metadata.get("added", ""),
                author=metadata.get("author"),
            ))

        return memories

    def store_document(self, file_path: Path, namespace: str, doc_type: str = "handoff") -> Memory:
        """
        Store a document verbatim (like handoff.md).

        The file content becomes the memory content without processing.
        """
        content = file_path.read_text()

        memory = Memory(
            content=content,
            namespace=namespace,
            type=doc_type,
            source="store-doc",
            confidence="medium",
            confidence_reason="document",
        )

        self.save(memory)
        return memory

    def reindex(self) -> int:
        """
        Reindex all memory files.

        Scans .ragtime/ and indexes any files not in ChromaDB.
        Returns count of files indexed.
        """
        if not self.memory_dir.exists():
            return 0

        count = 0
        for md_file in self.memory_dir.rglob("*.md"):
            try:
                memory = Memory.from_file(md_file)
                self.db.upsert(
                    ids=[memory.id],
                    documents=[memory.content],
                    metadatas=[memory.to_metadata()],
                )
                count += 1
            except Exception as e:
                print(f"Warning: Could not index {md_file}: {e}")

        return count

    def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """Remove empty directories up to memory_dir."""
        while dir_path != self.memory_dir and dir_path.exists():
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                dir_path = dir_path.parent
            else:
                break
