"""
Configuration management for ragtime.

Config lives in .ragtime/config.yaml in the project root.
"""

from pathlib import Path
from dataclasses import dataclass, field
import yaml


@dataclass
class DocsConfig:
    """Configuration for docs indexing."""
    paths: list[str] = field(default_factory=lambda: ["docs", ".ragtime"])
    patterns: list[str] = field(default_factory=lambda: ["**/*.md"])
    exclude: list[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/.ragtime/index/**",
        "**/.ragtime/branches/.*",  # Exclude synced (dot-prefixed) branches
    ])


@dataclass
class CodeConfig:
    """Configuration for code indexing."""
    paths: list[str] = field(default_factory=lambda: ["."])
    languages: list[str] = field(default_factory=lambda: ["dart", "typescript", "python"])
    exclude: list[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/.dart_tool/**",
    ])


@dataclass
class ConventionsConfig:
    """Configuration for convention checking."""
    files: list[str] = field(default_factory=lambda: [".ragtime/CONVENTIONS.md"])
    also_search_memories: bool = True


@dataclass
class RagtimeConfig:
    """Main ragtime configuration."""
    docs: DocsConfig = field(default_factory=DocsConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    conventions: ConventionsConfig = field(default_factory=ConventionsConfig)

    @classmethod
    def load(cls, project_path: Path) -> "RagtimeConfig":
        """Load config from .ragtime/config.yaml or return defaults."""
        config_path = project_path / ".ragtime" / "config.yaml"

        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        except (IOError, yaml.YAMLError):
            return cls()

        docs_data = data.get("docs", {})
        code_data = data.get("code", {})
        conventions_data = data.get("conventions", {})

        return cls(
            docs=DocsConfig(
                paths=docs_data.get("paths", DocsConfig().paths),
                patterns=docs_data.get("patterns", DocsConfig().patterns),
                exclude=docs_data.get("exclude", DocsConfig().exclude),
            ),
            code=CodeConfig(
                paths=code_data.get("paths", CodeConfig().paths),
                languages=code_data.get("languages", CodeConfig().languages),
                exclude=code_data.get("exclude", CodeConfig().exclude),
            ),
            conventions=ConventionsConfig(
                files=conventions_data.get("files", ConventionsConfig().files),
                also_search_memories=conventions_data.get(
                    "also_search_memories", ConventionsConfig().also_search_memories
                ),
            ),
        )

    def save(self, project_path: Path) -> None:
        """Save config to .ragtime/config.yaml."""
        config_dir = project_path / ".ragtime"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"

        data = {
            "docs": {
                "paths": self.docs.paths,
                "patterns": self.docs.patterns,
                "exclude": self.docs.exclude,
            },
            "code": {
                "paths": self.code.paths,
                "languages": self.code.languages,
                "exclude": self.code.exclude,
            },
            "conventions": {
                "files": self.conventions.files,
                "also_search_memories": self.conventions.also_search_memories,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def init_config(project_path: Path) -> RagtimeConfig:
    """Initialize a new config file with defaults."""
    config = RagtimeConfig()
    config.save(project_path)
    return config
