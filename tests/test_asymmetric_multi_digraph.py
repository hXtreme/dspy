"""Tests for AsymmetricMultiDiGraph construction."""

import pytest

from dspy.graph import AsymmetricMultiDiGraph
from dspy.graph.storage import (
    AdjecencyMatrixGraphStorage,
    LinkedListGraphStorage,
)
from dspy.graph.types import Edge, EdgeID, GraphStorageProvider, Vertex, VertexID

STORAGE_PROVIDERS = [
    pytest.param(AdjecencyMatrixGraphStorage, id="adjacency-matrix"),
    pytest.param(LinkedListGraphStorage, id="linked-list"),
]


class TestConstructEmpty:
    """Construct graphs with no vertices or edges."""

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_default_args(self, storage: GraphStorageProvider) -> None:
        """Graph with default args has no vertices or edges."""
        g = AsymmetricMultiDiGraph(storage)
        assert list(g.vertices()) == []
        assert list(g.edges()) == []

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_zero_vertices(self, storage: GraphStorageProvider) -> None:
        """Graph with vertices=0 has no vertices or edges."""
        g = AsymmetricMultiDiGraph(storage, vertices=0)
        assert list(g.vertices()) == []
        assert list(g.edges()) == []

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_empty_vertex_iterable(self, storage: GraphStorageProvider) -> None:
        """Graph from empty iterator has no vertices."""
        g = AsymmetricMultiDiGraph(storage, vertices=iter([]))
        assert list(g.vertices()) == []

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_empty_edges(self, storage: GraphStorageProvider) -> None:
        """Graph with explicit empty edges list has no edges."""
        g = AsymmetricMultiDiGraph(storage, vertices=3, edges=[])
        assert list(g.edges()) == []


class TestConstructWithVertices:
    """Construct graphs with vertices but no edges."""

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_int_vertices(self, storage: GraphStorageProvider) -> None:
        """Integer vertex count creates that many vertices."""
        g = AsymmetricMultiDiGraph(storage, vertices=5)
        verts = list(g.vertices())
        assert len(verts) == 5

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_iterable_vertices(self, storage: GraphStorageProvider) -> None:
        """Explicit vertex iterable is preserved."""
        vs = [Vertex(VertexID(10)), Vertex(VertexID(20)), Vertex(VertexID(30))]
        g = AsymmetricMultiDiGraph(storage, vertices=vs)
        verts = list(g.vertices())
        assert len(verts) == 3
        assert set(verts) == set(vs)

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_generator_vertices(self, storage: GraphStorageProvider) -> None:
        """Generator of vertices is consumed and stored."""
        gen = (Vertex(VertexID(i)) for i in range(4))
        g = AsymmetricMultiDiGraph(storage, vertices=gen)
        assert len(list(g.vertices())) == 4


class TestConstructWithEdges:
    """Construct graphs with both vertices and edges."""

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_single_edge(self, storage: GraphStorageProvider) -> None:
        """Graph with one edge reports it."""
        v0 = Vertex(VertexID(0))
        v1 = Vertex(VertexID(1))
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        edges = list(g.edges())
        assert len(edges) == 1

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_multiple_edges(self, storage: GraphStorageProvider) -> None:
        """Graph with multiple edges between distinct vertex pairs."""
        v0, v1, v2 = (Vertex(VertexID(i)) for i in range(3))
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v1, v2)),
            Edge((EdgeID(2), v0, v2)),
        ]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1, v2], edges=edges_in)
        edges_out = list(g.edges())
        assert len(edges_out) == 3

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_parallel_edges(self, storage: GraphStorageProvider) -> None:
        """Multiple edges between same pair of vertices (multi-graph)."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v0, v1)),
        ]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=edges_in)
        edges_out = list(g.edges())
        assert len(edges_out) == 2

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_self_loop(self, storage: GraphStorageProvider) -> None:
        """Self-loop edge from a vertex to itself."""
        v0 = Vertex(VertexID(0))
        e = Edge((EdgeID(0), v0, v0))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0], edges=[e])
        edges = list(g.edges())
        assert len(edges) == 1

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_asymmetric_edges(self, storage: GraphStorageProvider) -> None:
        """Edge u->v does not imply v->u."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        outgoing_v0 = list(g.outgoing_edges(v0))
        outgoing_v1 = list(g.outgoing_edges(v1))
        assert len(outgoing_v0) == 1
        assert len(outgoing_v1) == 0


class TestConstructWithProperties:
    """Construct graphs with vertex and edge properties."""

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_vertex_property(self, storage: GraphStorageProvider) -> None:
        """Vertex property factory is applied to each vertex."""
        v0 = Vertex(VertexID(0))
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0],
            vertex_property=lambda: {"color": "red"},
        )
        prop = g.property(v0)
        assert prop == {"color": "red"}

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_edge_property(self, storage: GraphStorageProvider) -> None:
        """Edge property factory is applied to each edge."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            edges=[e],
            edge_property=lambda: 1.0,
        )
        edge = next(iter(g.edges()))
        prop = g.property(edge)
        assert prop == 1.0

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_default_none_properties(self, storage: GraphStorageProvider) -> None:
        """Without property factories, properties default to None."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        assert g.property(v0) is None
        edge = next(iter(g.edges()))
        assert g.property(edge) is None

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_each_vertex_gets_own_property(self, storage: GraphStorageProvider) -> None:
        """Property factory called per-vertex, not shared."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            vertex_property=list,
        )
        p0 = g.property(v0)
        p1 = g.property(v1)
        assert p0 is not p1

    @pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
    def test_each_edge_gets_own_property(self, storage: GraphStorageProvider) -> None:
        """Property factory called per-edge, not shared."""
        v0, v1 = Vertex(VertexID(0)), Vertex(VertexID(1))
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v0, v1)),
        ]
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            edges=edges_in,
            edge_property=list,
        )
        edges_out = list(g.edges())
        p0 = g.property(edges_out[0])
        p1 = g.property(edges_out[1])
        assert p0 is not p1
