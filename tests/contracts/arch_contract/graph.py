from __future__ import annotations

from collections import defaultdict

from .config import ContractConfig
from .imports import ImportRecord, build_python_module_index
from .rules import Violation


def build_dependency_graph(config: ContractConfig, imports: list[ImportRecord]) -> dict[str, set[str]]:
    modules = build_python_module_index(config)
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for record in imports:
        if record.resolved_module and record.resolved_module != record.module_name:
            graph[record.module_name].add(record.resolved_module)
    return graph


def cycle_violations(config: ContractConfig, imports: list[ImportRecord]) -> list[Violation]:
    if not config.rules.get("forbid_import_cycles", False):
        return []
    graph = build_dependency_graph(config, imports)
    cycles = find_cycles(graph)
    return [
        Violation(
            rule="project_import_cycle",
            path="<dependency-graph>",
            line=0,
            detail=" -> ".join(cycle),
        )
        for cycle in cycles
    ]


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    cycles: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1:
                cycles.append(render_cycle(component, graph))
            elif component and component[0] in graph.get(component[0], set()):
                cycles.append([component[0], component[0]])

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return cycles


def render_cycle(component: list[str], graph: dict[str, set[str]]) -> list[str]:
    component_set = set(component)
    adjacency = defaultdict(list)
    for node in component:
        adjacency[node] = sorted(neighbor for neighbor in graph[node] if neighbor in component_set)

    start = sorted(component)[0]
    path = [start]
    seen = {start}
    current = start
    while True:
        next_nodes = adjacency[current]
        if start in next_nodes and len(path) > 1:
            path.append(start)
            return path
        next_node = next((node for node in next_nodes if node not in seen), next_nodes[0])
        path.append(next_node)
        if next_node == start:
            return path
        if next_node in seen:
            path.append(next_node)
            return path
        seen.add(next_node)
        current = next_node
