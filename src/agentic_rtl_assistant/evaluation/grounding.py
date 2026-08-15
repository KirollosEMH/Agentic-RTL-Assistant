from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean

from agentic_rtl_assistant.knowledge.evidence import EvidencePack, SourceEvidence

_CITATION = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.(?:sv|v)):(?P<start>\d+)(?:-(?P<end>\d+))?",
    re.IGNORECASE,
)


def _normalized_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().casefold()


@dataclass(frozen=True, slots=True)
class Citation:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class GroundingScores:
    grounding_accuracy: float | None
    citation_precision: float | None
    source_citation_recall: float | None
    answer_entity_support: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def extract_citations(answer: str | None) -> tuple[Citation, ...]:
    if not answer:
        return ()
    citations = []
    for match in _CITATION.finditer(answer):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        citations.append(Citation(match.group("path"), start, end))
    return tuple(citations)


def _inside_evidence(citation: Citation, evidence: SourceEvidence) -> bool:
    if _normalized_path(citation.path) != _normalized_path(evidence.path):
        return False
    if evidence.start_line is None or evidence.end_line is None:
        return True
    return (
        citation.start_line >= evidence.start_line
        and citation.end_line <= evidence.end_line
        and citation.end_line >= citation.start_line
    )


def _inside_project(citation: Citation, project_root: Path) -> bool:
    try:
        root = project_root.resolve()
        relative = PurePosixPath(citation.path.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        candidate = root.joinpath(*relative.parts).resolve()
        candidate.relative_to(root)
        line_count = len(candidate.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError, ValueError):
        return False
    return 1 <= citation.start_line <= citation.end_line <= max(1, line_count)


def score_grounding(
    answer: str | None,
    evidence: EvidencePack,
    *,
    project_root: Path,
    expected_source_paths: list[str] | tuple[str, ...] = (),
    expected_answer_entities: list[str] | tuple[str, ...] = (),
) -> GroundingScores:
    citations = extract_citations(answer)
    valid = tuple(
        citation
        for citation in citations
        if _inside_project(citation, project_root)
        and any(_inside_evidence(citation, item) for item in evidence.source_evidence)
    )

    expected_paths = {_normalized_path(path) for path in expected_source_paths}
    valid_paths = {_normalized_path(citation.path) for citation in valid}
    citation_precision = (
        len(valid) / len(citations) if citations else (0.0 if expected_paths else None)
    )
    source_recall = (
        len(expected_paths & valid_paths) / len(expected_paths) if expected_paths else None
    )

    answer_text = (answer or "").casefold()
    evidence_text = evidence.to_prompt().casefold()
    expected_entities = {entity.casefold() for entity in expected_answer_entities if entity.strip()}
    mentioned = {entity for entity in expected_entities if entity in answer_text}
    entity_support = (
        sum(entity in evidence_text for entity in mentioned) / len(mentioned)
        if mentioned
        else (0.0 if expected_entities else None)
    )

    components = [
        score for score in (citation_precision, source_recall, entity_support) if score is not None
    ]
    return GroundingScores(
        grounding_accuracy=fmean(components) if components else None,
        citation_precision=citation_precision,
        source_citation_recall=source_recall,
        answer_entity_support=entity_support,
    )


def aggregate_grounding_scores(
    scores: list[GroundingScores],
) -> dict[str, float | int | None]:
    names = tuple(GroundingScores.__dataclass_fields__)
    aggregated: dict[str, float | int | None] = {
        "evaluated_requests": sum(score.grounding_accuracy is not None for score in scores)
    }
    for name in names:
        values = [value for score in scores if (value := getattr(score, name)) is not None]
        aggregated[name] = fmean(values) if values else None
    return aggregated
