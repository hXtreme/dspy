"""Static-typing assertions for dspy.graph generics.

Exercised at runtime by pytest (no-op) and statically by ``ty``.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

from typing import assert_type

from dspy.graph import AsymmetricMultiDiGraph
from dspy.graph.storage import LinkedListGraphStorage
from dspy.graph.types import Edge, EdgeID, Vertex


def test_vertex_prop_propagates() -> None:
    """VertexProp generic flows through to property() return type."""
    v0 = Vertex(0)
    g: AsymmetricMultiDiGraph[int] = AsymmetricMultiDiGraph(
        LinkedListGraphStorage,
        vertices=[v0],
        vertex_property=lambda: 7,
    )
    assert_type(g.property(v0), int)


def test_edge_prop_propagates() -> None:
    """EdgeProp generic flows through to property() return type."""
    v0, v1 = Vertex(0), Vertex(1)
    e = Edge((EdgeID(0), v0, v1))
    g: AsymmetricMultiDiGraph[None, float] = AsymmetricMultiDiGraph(
        LinkedListGraphStorage,
        vertices=[v0, v1],
        edges=[e],
        edge_property=lambda: 1.5,
    )
    edge = next(iter(g.edges()))
    assert_type(g.property(edge), float)


def test_default_props_are_none() -> None:
    """Default VertexProp and EdgeProp resolve to None."""
    v0 = Vertex(0)
    g = AsymmetricMultiDiGraph(LinkedListGraphStorage, vertices=[v0])
    assert_type(g.property(v0), None)
