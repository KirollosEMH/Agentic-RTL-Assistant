import asyncio
from pathlib import Path

import pytest

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.approaches.base import RunResult
from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.knowledge.evidence import EvidencePack


class RecordingApproach:
    name = "recording"

    def __init__(self) -> None:
        self.contexts = []

    async def run(self, request, context) -> RunResult:
        self.contexts.append(context)
        return RunResult(
            request_id=request.request_id,
            approach=self.name,
            answer=f"answer to {request.text}",
            evidence=EvidencePack(entities=("CounterProducer",)),
        )


class RecordingApproachFactory:
    def __init__(self, approach: RecordingApproach) -> None:
        self.approach = approach

    def create(self) -> RecordingApproach:
        return self.approach


class SlowApproach:
    name = "slow"

    async def run(self, request, context) -> RunResult:
        del request, context
        await asyncio.sleep(10)
        raise AssertionError("timeout should cancel the approach")


@pytest.mark.asyncio
async def test_same_session_receives_bounded_conversation_history(
    repository_root: Path,
) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    config = config.model_copy(
        update={
            "context": config.context.model_copy(
                update={"max_conversation_messages": 2}
            )
        }
    )
    approach = RecordingApproach()
    service = ApplicationService(config, approach_factory=RecordingApproachFactory(approach))
    session_id = service.create_session()

    await service.ask("first question", session_id=session_id)
    await service.ask("follow-up question", session_id=session_id)
    await service.ask("third question", session_id=session_id)

    assert approach.contexts[0].recent_messages == ()
    assert [message.content for message in approach.contexts[1].recent_messages] == [
        "first question",
        "answer to first question",
    ]
    assert [message.content for message in approach.contexts[2].recent_messages] == [
        "follow-up question",
        "answer to follow-up question",
    ]
    assert approach.contexts[1].session_id == session_id
    assert approach.contexts[1].resolved_entities == ("CounterProducer",)


@pytest.mark.asyncio
async def test_sessions_do_not_share_conversation_history(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    approach = RecordingApproach()
    service = ApplicationService(config, approach_factory=RecordingApproachFactory(approach))

    first_session = service.create_session()
    second_session = service.create_session()
    await service.ask("first session question", session_id=first_session)
    await service.ask("second session question", session_id=second_session)

    assert approach.contexts[1].recent_messages == ()
    assert approach.contexts[1].resolved_entities == ()


@pytest.mark.asyncio
async def test_application_service_enforces_workflow_timeout(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    config = config.model_copy(
        update={
            "orchestration": config.orchestration.model_copy(
                update={"timeout_seconds": 0.01}
            )
        }
    )
    service = ApplicationService(
        config,
        approach_factory=RecordingApproachFactory(SlowApproach()),
    )

    result = await service.ask("take too long")

    assert not result.succeeded
    assert result.error == "request timed out after 0.01 seconds"
