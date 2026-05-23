"""Tests for AsymmetricMultiDiGraph construction.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

from typing import TYPE_CHECKING, assert_type

import pytest

from dspy.graph import AsymmetricMultiDiGraph
from dspy.graph.types import Edge, EdgeID, GraphStorageProvider, Vertex

if TYPE_CHECKING:
    from types import NoneType


class TestConstructEmpty:
    """Construct graphs with no vertices or edges."""

    def test_default_args(self, storage: GraphStorageProvider) -> None:
        """Graph with default args has no vertices or edges."""
        g = AsymmetricMultiDiGraph(storage)
        assert list(g.vertices()) == []
        assert list(g.edges()) == []

    def test_zero_vertices(self, storage: GraphStorageProvider) -> None:
        """Graph with vertices=0 has no vertices or edges."""
        g = AsymmetricMultiDiGraph(storage, vertices=0)
        assert list(g.vertices()) == []
        assert list(g.edges()) == []

    def test_empty_vertex_iterable(self, storage: GraphStorageProvider) -> None:
        """Graph from empty iterator has no vertices."""
        g = AsymmetricMultiDiGraph(storage, vertices=iter([]))
        assert list(g.vertices()) == []

    def test_empty_edges(self, storage: GraphStorageProvider) -> None:
        """Graph with explicit empty edges list has no edges."""
        g = AsymmetricMultiDiGraph(storage, vertices=3, edges=[])
        assert list(g.edges()) == []


class TestConstructWithVertices:
    """Construct graphs with vertices but no edges."""

    def test_int_vertices(self, storage: GraphStorageProvider) -> None:
        """Integer vertex count creates that many vertices."""
        g = AsymmetricMultiDiGraph(storage, vertices=5)
        verts = list(g.vertices())
        assert len(verts) == 5

    def test_iterable_vertices(self, storage: GraphStorageProvider) -> None:
        """Explicit vertex iterable is preserved."""
        vs = [Vertex(10), Vertex(20), Vertex(30)]
        g = AsymmetricMultiDiGraph(storage, vertices=vs)
        verts = list(g.vertices())
        assert len(verts) == 3
        assert set(verts) == set(vs)

    def test_generator_vertices(self, storage: GraphStorageProvider) -> None:
        """Generator of vertices is consumed and stored."""
        gen = (Vertex(i) for i in range(4))
        g = AsymmetricMultiDiGraph(storage, vertices=gen)
        assert len(list(g.vertices())) == 4


class TestConstructWithEdges:
    """Construct graphs with both vertices and edges."""

    def test_single_edge(self, storage: GraphStorageProvider) -> None:
        """Graph with one edge reports it."""
        v0 = Vertex(0)
        v1 = Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        edges = list(g.edges())
        assert len(edges) == 1

    def test_multiple_edges(self, storage: GraphStorageProvider) -> None:
        """Graph with multiple edges between distinct vertex pairs."""
        v0, v1, v2 = (Vertex(i) for i in range(3))
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v1, v2)),
            Edge((EdgeID(2), v0, v2)),
        ]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1, v2], edges=edges_in)
        edges_out = list(g.edges())
        assert len(edges_out) == 3

    def test_parallel_edges(self, storage: GraphStorageProvider) -> None:
        """Multiple edges between same pair of vertices (multi-graph)."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v0, v1)),
        ]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=edges_in)
        edges_out = list(g.edges())
        assert len(edges_out) == 2
        out_ids = {e[0] for e in g.outgoing_edges(v0)}
        in_ids = {e[0] for e in g.incoming_edges(v1)}
        assert out_ids == {EdgeID(0), EdgeID(1)}
        assert in_ids == {EdgeID(0), EdgeID(1)}

    def test_self_loop(self, storage: GraphStorageProvider) -> None:
        """Self-loop edge from a vertex to itself."""
        v0 = Vertex(0)
        e = Edge((EdgeID(0), v0, v0))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0], edges=[e])
        edges = list(g.edges())
        assert len(edges) == 1
        assert len(list(g.outgoing_edges(v0))) == 1
        assert len(list(g.incoming_edges(v0))) == 1

    def test_asymmetric_edges(self, storage: GraphStorageProvider) -> None:
        """Edge u->v does not imply v->u."""
        v0, v1 = Vertex(0), Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        assert len(list(g.outgoing_edges(v0))) == 1
        assert len(list(g.outgoing_edges(v1))) == 0
        assert len(list(g.incoming_edges(v1))) == 1
        assert len(list(g.incoming_edges(v0))) == 0

    def test_edges_param_accepts_generator(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """edges= accepts any iterable, including a generator."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_gen = (Edge((EdgeID(i), v0, v1)) for i in range(3))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=edges_gen)
        assert len(list(g.edges())) == 3

    def test_vertices_iterable_multi_iteration(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """g.vertices() can be iterated more than once."""
        g = AsymmetricMultiDiGraph(storage, vertices=3)
        first = list(g.vertices())
        second = list(g.vertices())
        assert first == second

    def test_edges_returns_generator_single_shot(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """g.edges() returns a generator that exhausts after one pass."""
        v0, v1 = Vertex(0), Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        it = g.edges()
        assert len(list(it)) == 1
        assert list(it) == []

    def test_zero_vertices_with_edges_raises(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """vertices=0 paired with non-empty edges raises an index/key error."""
        e = Edge((EdgeID(0), Vertex(0), Vertex(1)))
        with pytest.raises((IndexError, KeyError)):
            AsymmetricMultiDiGraph(storage, vertices=0, edges=[e])


class TestConstructWithProperties:
    """Construct graphs with vertex and edge properties."""

    def test_vertex_property(self, storage: GraphStorageProvider) -> None:
        """Vertex property factory is applied to each vertex."""
        v0 = Vertex(0)
        g: AsymmetricMultiDiGraph[dict[str, str]] = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0],
            vertex_property=lambda: {"color": "red"},
        )
        prop = g.property(v0)
        assert_type(prop, dict[str, str])
        assert prop == {"color": "red"}

    def test_edge_property(self, storage: GraphStorageProvider) -> None:
        """Edge property factory is applied to each edge."""
        v0, v1 = Vertex(0), Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g: AsymmetricMultiDiGraph[NoneType, float] = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            edges=[e],
            edge_property=lambda: 1.0,
        )
        edge = next(iter(g.edges()))
        prop = g.property(edge)
        assert_type(prop, float)
        assert prop == 1.0

    def test_default_none_properties(self, storage: GraphStorageProvider) -> None:
        """Without property factories, properties default to None."""
        v0, v1 = Vertex(0), Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        assert g.property(v0) is None
        edge = next(iter(g.edges()))
        assert g.property(edge) is None

    def test_each_vertex_gets_own_property(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Property factory called per-vertex, not shared."""
        v0, v1 = Vertex(0), Vertex(1)
        g: AsymmetricMultiDiGraph[list] = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            vertex_property=list,
        )
        p0 = g.property(v0)
        p1 = g.property(v1)
        assert p0 is not p1

    def test_each_edge_gets_own_property(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Property factory called per-edge, not shared."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v0, v1)),
        ]
        g: AsymmetricMultiDiGraph[NoneType, list] = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1],
            edges=edges_in,
            edge_property=list,
        )
        edges_out = list(g.edges())
        p0 = g.property(edges_out[0])
        p1 = g.property(edges_out[1])
        assert p0 is not p1


class TestConnectivity:
    """Verify incoming_edges / outgoing_edges / edges(vertex=...) contracts."""

    def test_incoming_edges_single(self, storage: GraphStorageProvider) -> None:
        """incoming_edges returns the sole arriving edge."""
        v0, v1 = Vertex(0), Vertex(1)
        e = Edge((EdgeID(0), v0, v1))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=[e])
        incoming = list(g.incoming_edges(v1))
        assert len(incoming) == 1
        eid, src, dst = incoming[0]
        assert eid == EdgeID(0)
        assert src == v0, str(g)
        assert dst == v1

    def test_incoming_edges_empty(self, storage: GraphStorageProvider) -> None:
        """Vertex with no inbound edges yields an empty iterable."""
        v0, v1 = Vertex(0), Vertex(1)
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1])
        assert list(g.incoming_edges(v0)) == []

    def test_parallel_incoming_outgoing(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Parallel edges show up on both ends of the connection."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v0, v1)),
        ]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=edges_in)
        out_ids = {e[0] for e in g.outgoing_edges(v0)}
        in_ids = {e[0] for e in g.incoming_edges(v1)}
        assert out_ids == {EdgeID(0), EdgeID(1)}
        assert in_ids == {EdgeID(0), EdgeID(1)}

    def test_self_loop_in_both_directions(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """A self-loop appears in both incoming and outgoing edges."""
        v0 = Vertex(0)
        e = Edge((EdgeID(0), v0, v0))
        g = AsymmetricMultiDiGraph(storage, vertices=[v0], edges=[e])
        assert len(list(g.incoming_edges(v0))) == 1
        assert len(list(g.outgoing_edges(v0))) == 1

    def test_edges_filtered_by_vertex(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """edges(vertex=v) returns only edges departing from v."""
        v0, v1, v2 = (Vertex(i) for i in range(3))
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(1), v1, v2)),
            Edge((EdgeID(2), v0, v2)),
        ]
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1, v2],
            edges=edges_in,
        )
        filtered = {e[0] for e in g.edges(vertex=v0)}
        assert filtered == {EdgeID(0), EdgeID(2)}, str(g)

    def test_edges_no_filter_matches_all(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """edges() without a vertex argument returns every edge."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_in = [Edge((EdgeID(0), v0, v1)), Edge((EdgeID(1), v0, v1))]
        g = AsymmetricMultiDiGraph(storage, vertices=[v0, v1], edges=edges_in)
        assert {e[0] for e in g.edges()} == {EdgeID(0), EdgeID(1)}


class TestPropertyAccess:
    """Property lookup behavior, including error paths and mutation."""

    def test_property_invalid_item_raises_type_error(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Passing a non-Vertex / non-Edge value raises TypeError."""
        g = AsymmetricMultiDiGraph(storage, vertices=1)
        with pytest.raises(TypeError, match="Vertex or Edge"):
            g.property("not-a-vertex-or-edge")  # ty: ignore[no-matching-overload]

    def test_property_mutation_persists(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Mutations through a returned property are visible on re-read."""
        v0 = Vertex(0)
        g: AsymmetricMultiDiGraph[list[int]] = AsymmetricMultiDiGraph(
            storage,
            vertices=[v0],
            vertex_property=list,
        )
        g.property(v0).append(42)
        assert g.property(v0) == [42]

    def test_property_unknown_vertex_raises(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Looking up a vertex not in the graph raises KeyError."""
        g = AsymmetricMultiDiGraph(storage, vertices=1)
        with pytest.raises(KeyError):
            g.property(Vertex(999))

    def test_property_unknown_edge_raises(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Looking up an edge not in the graph raises KeyError."""
        g = AsymmetricMultiDiGraph(storage, vertices=1)
        bogus = Edge((EdgeID(999), Vertex(0), Vertex(0)))
        with pytest.raises(KeyError):
            g.property(bogus)

    def test_property_factory_call_count(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Vertex property factory is called exactly once per vertex."""
        calls = 0

        def factory() -> int:
            nonlocal calls
            calls += 1
            return calls

        v0, v1, v2 = (Vertex(i) for i in range(3))
        AsymmetricMultiDiGraph(
            storage,
            vertices=[v0, v1, v2],
            vertex_property=factory,
        )
        assert calls == 3


class TestNonContiguousVertices:
    """Regressions for Vertex (label) vs VertexID (internal index) mix-up.

    See thoughts/shared/research/001_test_coverage_gaps.md item 2.
    The graph should translate user-supplied Vertex labels into the
    internal contiguous VertexID space before calling storage.update_edge.
    Currently it does not, so edges referencing labels >= len(vertices)
    raise IndexError or address the wrong cell.
    """

    def test_edges_with_non_contiguous_labels(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Edge between labelled vertices should round-trip intact."""
        v10, v20, v30 = (Vertex(i) for i in (10, 20, 30))
        e = Edge((EdgeID(0), v10, v30))
        g = AsymmetricMultiDiGraph(
            storage,
            vertices=[v10, v20, v30],
            edges=[e],
        )
        edges_out = list(g.edges())
        assert len(edges_out) == 1
        _eid, src, dst = edges_out[0]
        assert (src, dst) == (v10, v30)


class TestInvalidEdgeInputs:
    """xfail regressions: graph should validate edge inputs.

    See thoughts/shared/research/001_test_coverage_gaps.md item 4.
    Today these inputs are silently accepted; future hardening should
    raise.
    """

    def test_duplicate_edge_id_rejected(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Constructing with a duplicate EdgeID should raise KeyError."""
        v0, v1 = Vertex(0), Vertex(1)
        edges_in = [
            Edge((EdgeID(0), v0, v1)),
            Edge((EdgeID(0), v0, v1)),
        ]
        with pytest.raises(KeyError, match="Edge already exists"):
            AsymmetricMultiDiGraph(
                storage,
                vertices=[v0, v1],
                edges=edges_in,
            )

    def test_edge_with_unknown_vertex_rejected(
        self,
        storage: GraphStorageProvider,
    ) -> None:
        """Edge whose endpoint is not in vertices= should raise KeyError."""
        v0 = Vertex(0)
        v_missing = Vertex(99)
        e = Edge((EdgeID(0), v0, v_missing))
        with pytest.raises(KeyError, match="Unknown Vertex"):
            AsymmetricMultiDiGraph(storage, vertices=[v0], edges=[e])
