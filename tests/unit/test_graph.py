from pathlib import Path

from agentic_rtl_assistant.knowledge.graph import (
    GraphQuery,
    InMemoryGraphStore,
    KnowledgeGraphBuilder,
)
from agentic_rtl_assistant.knowledge.graph.model import RelationType
from agentic_rtl_assistant.knowledge.rag.graph_retriever import GraphRetriever
from agentic_rtl_assistant.rtl.parser import PyVerilogParser
from agentic_rtl_assistant.rtl.repository import RTLRepository


def _graph(rtl_root: Path):
    repository = RTLRepository(rtl_root)
    modules = PyVerilogParser(repository).parse_project()
    store = InMemoryGraphStore()
    KnowledgeGraphBuilder(store).rebuild(modules)
    return repository, store


def test_graph_builder_creates_pipeline_instantiation_edges(rtl_root: Path) -> None:
    _, store = _graph(rtl_root)
    relationships = {
        (edge.source, edge.relation, edge.target) for edge in store.edges()
    }

    assert (
        "module:DataPipeline",
        RelationType.INSTANTIATES,
        "module:CounterProducer",
    ) in relationships
    assert (
        "module:DataPipeline",
        RelationType.INSTANTIATES,
        "module:DataConsumer",
    ) in relationships


def test_graphrag_maps_relationships_back_to_source(rtl_root: Path) -> None:
    repository, store = _graph(rtl_root)
    evidence = GraphRetriever(repository, GraphQuery(store)).retrieve(
        "How is DataPipeline related to CounterProducer and DataConsumer?"
    )

    facts = {(item.source, item.relation, item.target) for item in evidence.relations}
    assert ("DataPipeline", "INSTANTIATES", "CounterProducer") in facts
    assert ("DataPipeline", "INSTANTIATES", "DataConsumer") in facts
    assert any(item.path == "data_pipeline.v" for item in evidence.source_evidence)
    assert evidence.metrics.graph_nodes_visited > 0
