from agentic_rtl_assistant.knowledge.graph.model import (
    GraphEdge,
    GraphNode,
    NodeType,
    RelationType,
)
from agentic_rtl_assistant.knowledge.graph.store import ProjectKnowledgeGraph
from agentic_rtl_assistant.rtl.types import ModuleInfo


class KnowledgeGraphBuilder:
    def __init__(self, store: ProjectKnowledgeGraph) -> None:
        self.store = store

    def rebuild(self, modules: tuple[ModuleInfo, ...]) -> None:
        self.store.clear()
        for module in modules:
            file_id = f"file:{module.location.path}"
            module_id = f"module:{module.name}"
            self.store.add_node(
                GraphNode(file_id, NodeType.FILE, module.location.path, module.location)
            )
            self.store.add_node(
                GraphNode(
                    module_id,
                    NodeType.MODULE,
                    module.name,
                    module.location,
                    {"source_hash": module.source_hash},
                )
            )
            self.store.add_edge(GraphEdge(file_id, RelationType.DEFINES, module_id))

        for module in modules:
            module_id = f"module:{module.name}"
            for port in module.ports:
                port_id = f"port:{module.name}.{port.name}"
                self.store.add_node(
                    GraphNode(
                        port_id,
                        NodeType.PORT,
                        port.name,
                        module.location,
                        {
                            "module": module.name,
                            "direction": port.direction,
                            "data_type": port.data_type,
                            "width": port.width,
                        },
                    )
                )
                self.store.add_edge(GraphEdge(module_id, RelationType.HAS_PORT, port_id))
            for instance in module.instances:
                instance_id = f"instance:{module.name}.{instance.name}"
                target_id = f"module:{instance.module}"
                if self.store.get_node(target_id) is None:
                    self.store.add_node(GraphNode(target_id, NodeType.MODULE, instance.module))
                self.store.add_node(
                    GraphNode(
                        instance_id,
                        NodeType.INSTANCE,
                        instance.name,
                        module.location,
                        {
                            "parent_module": module.name,
                            "module_type": instance.module,
                            "connections": dict(instance.connections),
                        },
                    )
                )
                self.store.add_edge(
                    GraphEdge(module_id, RelationType.HAS_INSTANCE, instance_id)
                )
                self.store.add_edge(GraphEdge(instance_id, RelationType.INSTANCE_OF, target_id))
                self.store.add_edge(GraphEdge(module_id, RelationType.INSTANTIATES, target_id))
