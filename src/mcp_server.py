"""
MCP Server for ragtime.

Exposes ragtime operations as MCP tools for Claude integration.
Run with: python -m src.mcp_server
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Any

from .db import RagtimeDB
from .memory import Memory, MemoryStore


class RagtimeMCPServer:
    """MCP Server that exposes ragtime operations as tools."""

    def __init__(self, project_path: Path | None = None):
        """
        Initialize the MCP server.

        Args:
            project_path: Root of the project (defaults to cwd)
        """
        self.project_path = project_path or Path.cwd()
        self._db = None
        self._store = None

    @property
    def db(self) -> RagtimeDB:
        """Lazy-load the database."""
        if self._db is None:
            db_path = self.project_path / ".ragtime" / "index"
            self._db = RagtimeDB(db_path)
        return self._db

    @property
    def store(self) -> MemoryStore:
        """Lazy-load the memory store."""
        if self._store is None:
            self._store = MemoryStore(self.project_path, self.db)
        return self._store

    def get_author(self) -> str:
        """Get the current developer's username."""
        try:
            result = subprocess.run(
                ["gh", "api", "user", "--jq", ".login"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["git", "config", "user.name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().lower().replace(" ", "-")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return "unknown"

    def get_tools(self) -> list[dict]:
        """Return the list of available tools."""
        return [
            {
                "name": "remember",
                "description": "Store a memory with structured metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace: app, team, user-{name}, branch-{name}"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["architecture", "feature", "integration", "convention",
                                     "preference", "decision", "pattern", "task-state", "handoff"],
                            "description": "Memory type"
                        },
                        "component": {
                            "type": "string",
                            "description": "Component area (e.g., auth, claims, shifts)"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "default": "medium",
                            "description": "Confidence level"
                        },
                        "confidence_reason": {
                            "type": "string",
                            "description": "Why this confidence level"
                        },
                        "source": {
                            "type": "string",
                            "default": "remember",
                            "description": "Source of this memory"
                        },
                        "issue": {
                            "type": "string",
                            "description": "Related GitHub issue (e.g., #301)"
                        },
                        "epic": {
                            "type": "string",
                            "description": "Parent epic (e.g., #286)"
                        },
                        "branch": {
                            "type": "string",
                            "description": "Related branch name"
                        }
                    },
                    "required": ["content", "namespace", "type"]
                }
            },
            {
                "name": "search",
                "description": "Semantic search over indexed content and memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Filter by namespace"
                        },
                        "type": {
                            "type": "string",
                            "description": "Filter by type (docs, code, architecture, etc.)"
                        },
                        "component": {
                            "type": "string",
                            "description": "Filter by component"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Max results to return"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "store_doc",
                "description": "Store a document verbatim (like handoff.md)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The document content to store"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Namespace for the document"
                        },
                        "doc_type": {
                            "type": "string",
                            "enum": ["handoff", "document", "plan", "notes"],
                            "default": "handoff",
                            "description": "Document type"
                        }
                    },
                    "required": ["content", "namespace"]
                }
            },
            {
                "name": "list_memories",
                "description": "List memories with optional filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Filter by namespace (use * suffix for prefix match)"
                        },
                        "type": {
                            "type": "string",
                            "description": "Filter by type"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "graduated", "abandoned", "ungraduated"],
                            "description": "Filter by status"
                        },
                        "component": {
                            "type": "string",
                            "description": "Filter by component"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max results"
                        }
                    }
                }
            },
            {
                "name": "get_memory",
                "description": "Get a specific memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID"
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "forget",
                "description": "Delete a memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to delete"
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "graduate",
                "description": "Graduate a branch memory to app namespace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to graduate"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "default": "high",
                            "description": "Confidence level for graduated memory"
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "update_status",
                "description": "Update a memory's status (e.g., mark as abandoned)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "graduated", "abandoned", "ungraduated"],
                            "description": "New status"
                        }
                    },
                    "required": ["memory_id", "status"]
                }
            }
        ]

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute a tool call."""
        if name == "remember":
            return self._remember(arguments)
        elif name == "search":
            return self._search(arguments)
        elif name == "store_doc":
            return self._store_doc(arguments)
        elif name == "list_memories":
            return self._list_memories(arguments)
        elif name == "get_memory":
            return self._get_memory(arguments)
        elif name == "forget":
            return self._forget(arguments)
        elif name == "graduate":
            return self._graduate(arguments)
        elif name == "update_status":
            return self._update_status(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _remember(self, args: dict) -> dict:
        """Store a memory."""
        memory = Memory(
            content=args["content"],
            namespace=args["namespace"],
            type=args["type"],
            component=args.get("component"),
            confidence=args.get("confidence", "medium"),
            confidence_reason=args.get("confidence_reason"),
            source=args.get("source", "remember"),
            author=self.get_author(),
            issue=args.get("issue"),
            epic=args.get("epic"),
            branch=args.get("branch"),
        )

        file_path = self.store.save(memory)

        return {
            "success": True,
            "memory_id": memory.id,
            "file": str(file_path.relative_to(self.project_path)),
            "namespace": memory.namespace,
            "type": memory.type,
        }

    def _search(self, args: dict) -> dict:
        """Search indexed content."""
        results = self.db.search(
            query=args["query"],
            limit=args.get("limit", 10),
            namespace=args.get("namespace"),
            type_filter=args.get("type"),
            component=args.get("component"),
        )

        return {
            "count": len(results),
            "results": [
                {
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": 1 - r["distance"] if r["distance"] else None,
                }
                for r in results
            ]
        }

    def _store_doc(self, args: dict) -> dict:
        """Store a document verbatim."""
        memory = Memory(
            content=args["content"],
            namespace=args["namespace"],
            type=args.get("doc_type", "handoff"),
            source="store-doc",
            confidence="medium",
            confidence_reason="document",
            author=self.get_author(),
        )

        file_path = self.store.save(memory)

        return {
            "success": True,
            "memory_id": memory.id,
            "file": str(file_path.relative_to(self.project_path)),
            "namespace": memory.namespace,
            "type": memory.type,
        }

    def _list_memories(self, args: dict) -> dict:
        """List memories with filters."""
        memories = self.store.list_memories(
            namespace=args.get("namespace"),
            type_filter=args.get("type"),
            status=args.get("status"),
            component=args.get("component"),
            limit=args.get("limit", 20),
        )

        return {
            "count": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "namespace": m.namespace,
                    "type": m.type,
                    "component": m.component,
                    "status": m.status,
                    "confidence": m.confidence,
                    "added": m.added,
                    "source": m.source,
                    "preview": m.content[:100],
                }
                for m in memories
            ]
        }

    def _get_memory(self, args: dict) -> dict:
        """Get a specific memory."""
        memory = self.store.get(args["memory_id"])

        if not memory:
            return {"success": False, "error": "Memory not found"}

        return {
            "success": True,
            "memory": {
                "id": memory.id,
                "content": memory.content,
                "namespace": memory.namespace,
                "type": memory.type,
                "component": memory.component,
                "status": memory.status,
                "confidence": memory.confidence,
                "confidence_reason": memory.confidence_reason,
                "added": memory.added,
                "author": memory.author,
                "source": memory.source,
                "issue": memory.issue,
                "epic": memory.epic,
                "branch": memory.branch,
            }
        }

    def _forget(self, args: dict) -> dict:
        """Delete a memory."""
        success = self.store.delete(args["memory_id"])

        return {
            "success": success,
            "memory_id": args["memory_id"],
        }

    def _graduate(self, args: dict) -> dict:
        """Graduate a branch memory."""
        try:
            graduated = self.store.graduate(
                args["memory_id"],
                args.get("confidence", "high"),
            )

            if not graduated:
                return {"success": False, "error": "Memory not found"}

            return {
                "success": True,
                "original_id": args["memory_id"],
                "graduated_id": graduated.id,
                "namespace": graduated.namespace,
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}

    def _update_status(self, args: dict) -> dict:
        """Update a memory's status."""
        success = self.store.update_status(
            args["memory_id"],
            args["status"],
        )

        return {
            "success": success,
            "memory_id": args["memory_id"],
            "status": args["status"],
        }

    def handle_message(self, message: dict) -> dict:
        """Handle an incoming JSON-RPC message."""
        method = message.get("method")
        msg_id = message.get("id")

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "ragtime",
                            "version": "0.2.6",
                        },
                        "capabilities": {
                            "tools": {},
                        },
                    },
                }

            elif method == "notifications/initialized":
                # No response needed for notifications
                return None

            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": self.get_tools(),
                    },
                }

            elif method == "tools/call":
                params = message.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                result = self.call_tool(tool_name, arguments)

                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }

    def run(self):
        """Run the MCP server on stdin/stdout."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                message = json.loads(line)
                response = self.handle_message(message)

                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                break


def main():
    """Entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Ragtime MCP Server")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Project path (defaults to current directory)",
    )
    args = parser.parse_args()

    server = RagtimeMCPServer(args.path)
    server.run()


if __name__ == "__main__":
    main()
