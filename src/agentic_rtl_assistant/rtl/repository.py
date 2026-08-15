from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agentic_rtl_assistant.rtl.types import SourceMatch


class PathConfinementError(ValueError):
    """Raised when a requested path escapes the selected project root."""


class RTLRepositoryError(RuntimeError):
    """Raised for invalid or unavailable RTL repositories."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: str
    modified_ns: int
    content_hash: str


class RTLRepository:
    def __init__(
        self,
        root: Path,
        *,
        extensions: tuple[str, ...] = (".v", ".sv"),
        ignored_paths: tuple[str, ...] = (".git", ".venv", "build"),
        allow_reads: bool = True,
        allow_writes: bool = False,
    ) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise RTLRepositoryError(f"invalid project directory: {self.root}")
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.ignored_paths = frozenset(ignored_paths)
        self.allow_reads = allow_reads
        self.allow_writes = allow_writes

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.root / candidate).resolve()
        )
        if not candidate.is_relative_to(self.root):
            raise PathConfinementError(f"path escapes project root: {path}")
        return candidate

    def _require_reads(self) -> None:
        if not self.allow_reads:
            raise RTLRepositoryError("source reads are disabled by configuration")

    def _require_writes(self) -> None:
        if not self.allow_writes:
            raise RTLRepositoryError("source writes are disabled by configuration")

    def validate_write_target(
        self, path: str | Path, *, overwrite: bool = False
    ) -> Path:
        self._require_writes()
        resolved = self.resolve(path)
        if resolved.suffix.lower() not in self.extensions:
            allowed = ", ".join(self.extensions)
            raise RTLRepositoryError(f"write target must use an RTL extension: {allowed}")
        if resolved.exists():
            if not resolved.is_file():
                raise RTLRepositoryError(f"write target is not a file: {resolved}")
            if not overwrite:
                raise RTLRepositoryError(
                    f"source file already exists; overwrite approval is required: {resolved}"
                )
        elif not resolved.parent.is_dir():
            raise RTLRepositoryError(
                f"write target parent directory does not exist: {resolved.parent}"
            )
        return resolved

    def list_verilog_files(self) -> tuple[Path, ...]:
        self._require_reads()
        files = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            relative = path.relative_to(self.root)
            if any(part in self.ignored_paths for part in relative.parts):
                continue
            files.append(path)
        return tuple(sorted(files, key=lambda item: item.as_posix().lower()))

    def read_source(self, path: str | Path) -> str:
        self._require_reads()
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise RTLRepositoryError(f"source file does not exist: {resolved}")
        return resolved.read_text(encoding="utf-8")

    def write_source(
        self,
        path: str | Path,
        source: str,
        *,
        overwrite: bool = False,
    ) -> Path:
        resolved = self.validate_write_target(path, overwrite=overwrite)
        mode = "w" if overwrite else "x"
        with resolved.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(source)
        return resolved

    def read_lines(self, path: str | Path, start_line: int, end_line: int) -> str:
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid source line range")
        lines = self.read_source(path).splitlines()
        return "\n".join(lines[start_line - 1 : end_line])

    def search_source(self, query: str, *, limit: int = 50) -> tuple[SourceMatch, ...]:
        needle = query.casefold()
        matches: list[SourceMatch] = []
        for path in self.list_verilog_files():
            relative = path.relative_to(self.root).as_posix()
            for line_number, line in enumerate(self.read_source(path).splitlines(), start=1):
                if needle in line.casefold():
                    matches.append(SourceMatch(relative, line_number, line))
                    if len(matches) >= limit:
                        return tuple(matches)
        return tuple(matches)

    def snapshot(self, path: str | Path) -> FileSnapshot:
        resolved = self.resolve(path)
        content = self.read_source(resolved)
        return FileSnapshot(
            path=resolved.relative_to(self.root).as_posix(),
            modified_ns=resolved.stat().st_mtime_ns,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
