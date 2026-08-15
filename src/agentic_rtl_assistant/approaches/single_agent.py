from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rtl_assistant.approaches.base import RunContext, RunResult, UserRequest
from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    RetrievalMetrics,
    SourceEvidence,
)
from agentic_rtl_assistant.models.base import ModelProvider
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest, TokenUsage
from agentic_rtl_assistant.rtl.repository import RTLRepository, RTLRepositoryError
from agentic_rtl_assistant.rtl.tools import RTLWriteTool, WriteRequest
from agentic_rtl_assistant.session.models import as_model_messages
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.context import ContextWindowMetrics
from agentic_rtl_assistant.telemetry.timing import TimingMetrics
from agentic_rtl_assistant.telemetry.tokens import aggregate_usage
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    evidence: tuple[SourceEvidence, ...] = ()
    written_path: str | None = None


class SingleAgentApproach:
    name = "single_agent"

    def __init__(
        self,
        provider: ModelProvider,
        repository: RTLRepository,
        *,
        model: str,
        provider_name: str,
        prompt: str,
        telemetry: TelemetryCollector,
        temperature: float = 0.0,
        max_steps: int = 12,
        max_evidence_items: int = 12,
        max_evidence_tokens: int = 4000,
        max_output_tokens: int | None = None,
        write_tool: RTLWriteTool | None = None,
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.model = model
        self.provider_name = provider_name
        self.prompt = prompt
        self.telemetry = telemetry
        self.temperature = temperature
        self.max_steps = max_steps
        self.max_evidence_items = max_evidence_items
        self.max_evidence_tokens = max_evidence_tokens
        self.max_output_tokens = max_output_tokens
        self.write_tool = write_tool

    @staticmethod
    def _json_object(content: str) -> dict[str, Any] | None:
        candidate = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            decoder = json.JSONDecoder()
            for start, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate, start)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and ("tool" in value or "answer" in value):
                    return value
            return None
        return (
            value
            if isinstance(value, dict) and ("tool" in value or "answer" in value)
            else None
        )

    def _allowed_path(self, raw_path: object) -> tuple[Path, str]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("path must be a non-empty string")
        resolved = self.repository.resolve(raw_path.strip())
        allowed = set(self.repository.list_verilog_files())
        if resolved not in allowed:
            raise ValueError("path is not a readable Verilog/SystemVerilog project file")
        return resolved, resolved.relative_to(self.repository.root).as_posix()

    def _list_files(self) -> ToolResult:
        paths = [
            path.relative_to(self.repository.root).as_posix()
            for path in self.repository.list_verilog_files()
        ]
        return ToolResult(json.dumps({"files": paths}))

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        resolved, relative = self._allowed_path(arguments.get("path"))
        lines = self.repository.read_source(resolved).splitlines()
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", max(1, len(lines)))
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError("start_line must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError("end_line must be an integer")
        if start < 1 or end < start:
            raise ValueError("invalid source line range")
        if lines and start > len(lines):
            raise ValueError(f"start_line exceeds the file length ({len(lines)} lines)")
        end = min(end, max(1, len(lines)))
        content = "\n".join(lines[start - 1 : end])
        evidence = SourceEvidence(relative, start, end, content, "agent_read_file")
        return ToolResult(
            f"[{relative}:{start}-{end}]\n{content}",
            (evidence,),
        )

    def _search_source(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        raw_limit = arguments.get("limit", 20)
        if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
            raise ValueError("limit must be an integer")
        limit = min(max(raw_limit, 1), 50)
        matches = self.repository.search_source(query.strip(), limit=limit)
        evidence = tuple(
            SourceEvidence(
                match.path,
                match.line,
                match.line,
                match.content,
                "agent_search_source",
            )
            for match in matches
        )
        return ToolResult(
            json.dumps(
                {
                    "matches": [
                        {
                            "path": match.path,
                            "line": match.line,
                            "content": match.content,
                        }
                        for match in matches
                    ]
                }
            ),
            evidence,
        )

    async def _write_file(
        self,
        arguments: dict[str, Any],
        context: RunContext,
    ) -> ToolResult:
        if self.write_tool is None:
            raise ValueError("write_file tool is unavailable")
        path = arguments.get("path")
        content = arguments.get("content")
        overwrite = arguments.get("overwrite", False)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")
        result = await self.write_tool.execute(
            WriteRequest(path.strip(), content, overwrite),
            context.write_confirmation,
        )
        return ToolResult(
            json.dumps(
                {"written": result.path, "bytes_written": result.bytes_written}
            ),
            (),
            result.path,
        )

    async def _execute_tool(
        self,
        name: object,
        arguments: object,
        context: RunContext,
    ) -> ToolResult:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        if name == "list_files":
            return self._list_files()
        if name == "read_file":
            return self._read_file(arguments)
        if name == "search_source":
            return self._search_source(arguments)
        if name == "write_file":
            return await self._write_file(arguments, context)
        raise ValueError(f"unknown tool: {name}")

    def _accept_evidence(
        self,
        current: list[SourceEvidence],
        additions: tuple[SourceEvidence, ...],
    ) -> tuple[SourceEvidence, ...]:
        seen = {(item.path, item.start_line, item.end_line) for item in current}
        token_count = sum(len(item.content.split()) for item in current)
        visible = []
        for item in additions:
            key = (item.path, item.start_line, item.end_line)
            item_tokens = len(item.content.split())
            if key in seen:
                visible.append(item)
                continue
            if len(current) >= self.max_evidence_items:
                continue
            if token_count + item_tokens > self.max_evidence_tokens:
                continue
            current.append(item)
            visible.append(item)
            seen.add(key)
            token_count += item_tokens
        return tuple(visible)

    @staticmethod
    def _format_source_results(items: tuple[SourceEvidence, ...]) -> str:
        return "\n\n".join(
            f"[{item.path}:{item.start_line}-{item.end_line}]\n{item.content}"
            for item in items
        )

    async def run(self, request: UserRequest, context: RunContext) -> RunResult:
        started = time.perf_counter()
        start_trace = ExecutionTrace(
            request.request_id, self.name, EventType.REQUEST_STARTED
        )
        self.telemetry.record(start_trace)
        messages = [
            ModelMessage("system", self.prompt),
            *as_model_messages(context.recent_messages),
            ModelMessage("user", request.text),
        ]
        usages: list[TokenUsage] = []
        evidence_items: list[SourceEvidence] = []
        traces = [start_trace]
        tool_calls = 0
        successful_tool_calls = 0
        source_tool_calls = 0
        retrieval_seconds = 0.0
        written_files: list[str] = []
        final_answer: str | None = None
        error: str | None = None

        for step in range(1, self.max_steps + 1):
            response = await self.provider.generate(
                ModelRequest(
                    model=self.model,
                    messages=tuple(messages),
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    metadata=context.model_metadata,
                )
            )
            usages.append(response.usage)
            action = self._json_object(response.content)
            if (
                action is not None
                and "tool" not in action
                and isinstance(action.get("answer"), str)
            ):
                if source_tool_calls:
                    final_answer = action["answer"].strip()
                    break
                messages.extend(
                    (
                        ModelMessage("assistant", response.content),
                        ModelMessage(
                            "user",
                            "You must inspect source with read_file or search_source before "
                            "answering. list_files alone is not source evidence.",
                        ),
                    )
                )
                continue

            if action is None or "tool" not in action:
                messages.extend(
                    (
                        ModelMessage("assistant", response.content),
                        ModelMessage(
                            "user",
                            "Return one JSON object using either `tool` and `arguments`, or "
                            "`answer`. Inspect the project before answering.",
                        ),
                    )
                )
                continue

            tool_name = action.get("tool")
            tool_started = time.perf_counter()
            started_trace = ExecutionTrace(
                request.request_id,
                str(tool_name),
                EventType.TOOL_STARTED,
                metadata={"step": step},
            )
            self.telemetry.record(started_trace)
            traces.append(started_trace)
            result_truncated = False
            try:
                tool_result = await self._execute_tool(
                    tool_name, action.get("arguments", {}), context
                )
                visible_evidence = self._accept_evidence(
                    evidence_items, tool_result.evidence
                )
                result_truncated = len(visible_evidence) < len(tool_result.evidence)
                if result_truncated and not visible_evidence:
                    tool_error = (
                        "tool result exceeds the configured evidence budget; request a "
                        "smaller line range or a narrower search"
                    )
                    result_content = json.dumps({"error": tool_error})
                elif result_truncated:
                    tool_error = None
                    result_content = (
                        self._format_source_results(visible_evidence)
                        + "\n\n[Additional matches omitted: evidence budget reached]"
                    )
                    successful_tool_calls += 1
                else:
                    tool_error = None
                    result_content = tool_result.content
                    successful_tool_calls += 1
                if tool_error is None and tool_name in {"read_file", "search_source"}:
                    source_tool_calls += 1
                if tool_result.written_path is not None:
                    written_files.append(tool_result.written_path)
            except (OSError, RTLRepositoryError, ValueError) as exc:
                result_content = json.dumps({"error": str(exc)})
                tool_error = str(exc)
            tool_duration = time.perf_counter() - tool_started
            retrieval_seconds += tool_duration
            completed_trace = ExecutionTrace(
                request.request_id,
                str(tool_name),
                EventType.TOOL_COMPLETED,
                duration_seconds=tool_duration,
                metadata={
                    "step": step,
                    "error": tool_error,
                    "result_truncated": result_truncated,
                },
            )
            self.telemetry.record(completed_trace)
            traces.append(completed_trace)
            tool_calls += 1
            messages.extend(
                (
                    ModelMessage("assistant", json.dumps(action)),
                    ModelMessage(
                        "user",
                        f"Tool result for {tool_name}:\n{result_content}\n\n"
                        "Continue by returning exactly one JSON object.",
                    ),
                )
            )

        if not final_answer:
            error = f"single agent did not produce a grounded answer within {self.max_steps} steps"

        duration = time.perf_counter() - started
        completed = ExecutionTrace(
            request.request_id,
            self.name,
            EventType.REQUEST_COMPLETED if error is None else EventType.REQUEST_FAILED,
            duration_seconds=duration,
            metadata={
                "tool_calls": tool_calls,
                "successful_tool_calls": successful_tool_calls,
                "source_tool_calls": source_tool_calls,
            },
        )
        self.telemetry.record(completed)
        traces.append(completed)
        evidence = EvidencePack(
            source_evidence=tuple(evidence_items),
            metrics=RetrievalMetrics(
                source_chunks_retrieved=len(evidence_items),
                source_tokens_retrieved=sum(
                    len(item.content.split()) for item in evidence_items
                ),
                retrieval_latency_seconds=retrieval_seconds,
            ),
        )
        return RunResult(
            request_id=request.request_id,
            approach=self.name,
            answer=final_answer,
            written_files=tuple(dict.fromkeys(written_files)),
            evidence=evidence,
            usage=aggregate_usage(usages),
            context_window=ContextWindowMetrics.from_usage_events(
                usages, history_messages=len(context.recent_messages)
            ),
            timing=TimingMetrics(
                total_seconds=duration,
                model_seconds=max(0.0, duration - retrieval_seconds),
                retrieval_seconds=retrieval_seconds,
            ),
            traces=tuple(traces),
            provider=self.provider_name,
            model=self.model,
            error=error,
        )
