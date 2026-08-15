from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.rtl.validator import RTLValidator


@dataclass(frozen=True, slots=True)
class WriteRequest:
    path: str
    content: str
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: str
    bytes_written: int


type WriteConfirmation = Callable[[WriteRequest], Awaitable[bool]]


class WriteToolError(ValueError):
    pass


class WriteDeclinedError(WriteToolError):
    pass


class RTLWriteTool:
    def __init__(
        self,
        repository: RTLRepository,
        validator: RTLValidator,
        *,
        require_confirmation: bool = True,
    ) -> None:
        self.repository = repository
        self.validator = validator
        self.require_confirmation = require_confirmation

    @property
    def enabled(self) -> bool:
        return self.repository.allow_writes

    async def execute(
        self,
        request: WriteRequest,
        confirmation: WriteConfirmation | None = None,
    ) -> WriteResult:
        content = request.content.strip()
        if not content:
            raise WriteToolError("write content cannot be empty")

        target = self.repository.validate_write_target(
            request.path, overwrite=request.overwrite
        )
        normalized = replace(
            request,
            path=target.relative_to(self.repository.root).as_posix(),
            content=content + "\n",
        )
        validation = self.validator.validate_verilog(normalized.content)
        if not validation.valid:
            errors = "; ".join(validation.errors) or "unknown parser error"
            raise WriteToolError(f"refusing to write invalid RTL: {errors}")

        if self.require_confirmation:
            if confirmation is None:
                raise WriteToolError("write requires interactive user confirmation")
            if not await confirmation(normalized):
                raise WriteDeclinedError("write was declined by the user")

        written = self.repository.write_source(
            normalized.path,
            normalized.content,
            overwrite=normalized.overwrite,
        )
        return WriteResult(
            path=written.relative_to(self.repository.root).as_posix(),
            bytes_written=len(normalized.content.encode("utf-8")),
        )
