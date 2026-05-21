"""Concrete graph implementations.

Contains :class:`AsymmetricMultiDiGraph`, an asymmetric directed multi-graph
that allows multiple parallel edges between any ordered pair of vertices,
with optional per-vertex and per-edge properties.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import typing
from collections.abc import Iterable
from types import NoneType
from typing import TYPE_CHECKING, Self

from dspy.graph.types import (
    Edge,
    EdgeID,
    EdgeMonad,
    Graph,
    GraphStorageProvider,
    PropertyConstructor,
    Vertex,
    VertexID,
)


class _MultiGraphEdgeMonad(EdgeMonad):
    """Edge monad that accumulates a set of parallel edge identifiers.

    Used internally by :class:`AsymmetricMultiDiGraph` to support multiple
    distinct edges between the same ``(source, destination)`` vertex pair.
    Each :meth:`chain` call adds a new :class:`EdgeID` to the set rather
    than replacing the previous one.
    """

    _e: set[EdgeID]

    def __init__(self, e: EdgeID) -> None:
        """Initialise the monad with a single edge identifier.

        :param e: The first edge identifier for this vertex pair.
        """
        self._e = {e}

    @classmethod
    def bind(cls, e: EdgeID) -> Self:
        """Create a new monad containing a single edge identifier.

        :param e: The initial edge identifier.
        :return: A fresh :class:`_MultiGraphEdgeMonad` wrapping *e*.
        """
        return cls(e)

    def chain(self, e: EdgeID) -> Self:
        """Add an additional edge identifier to this monad.

        :param e: The edge identifier to add to the set.
        :return: ``self``, updated in place.
        """
        self._e.add(e)
        return self

    def unwrap(self) -> Iterable[EdgeID]:
        """Return all accumulated edge identifiers as a frozen set.

        :return: A :class:`frozenset` of every :class:`EdgeID` in this monad.
        """
        return frozenset(self._e)


def _none_property() -> None:
    return None


class AsymmetricMultiDiGraph[VertexProp = NoneType, EdgeProp = NoneType](
    Graph[_MultiGraphEdgeMonad, VertexProp, EdgeProp],
):
    """Asymmetric directed multi-graph with optional vertex and edge properties.

    An asymmetric directed multi-graph where:

    * Edges are directed -- an edge from *u* to *v* does **not** imply an
      edge from *v* to *u*.
    * Multiple parallel edges between the same ordered vertex pair are
      permitted (multi-graph semantics).
    * Each vertex and each edge can carry an associated property object
      whose type is determined by the *VertexProp* and *EdgeProp* type
      parameters respectively.

    The underlying storage backend is injected at construction time via a
    :data:`GraphStorageProvider` callable, allowing callers to choose
    between adjacency-matrix and adjacency-list representations.

    :param VertexProp: Type of per-vertex property values (default
        :class:`NoneType`).
    :param EdgeProp: Type of per-edge property values (default
        :class:`NoneType`).
    """

    __vertices: dict[Vertex, tuple[VertexID, VertexProp]]
    __edges: dict[EdgeID, EdgeProp]
    __inv_vertices: dict[VertexID, Vertex]

    def __init__(
        self,
        storage: GraphStorageProvider[_MultiGraphEdgeMonad],
        vertices: Iterable[Vertex] | int = 0,
        edges: Iterable[Edge] | None = None,
        vertex_property: PropertyConstructor[VertexProp] | None = None,
        edge_property: PropertyConstructor[EdgeProp] | None = None,
    ) -> None:
        """Construct a new asymmetric directed multi-graph.

        :param storage: A callable that creates a :class:`GraphStorage` backend
            given a vertex count and the :class:`_MultiGraphEdgeMonad` type.
        :param vertices: Either an integer count (vertices are auto-generated as
            ``Vertex(VertexID(0))`` through ``Vertex(VertexID(n-1))``), or an
            explicit iterable of :class:`Vertex` objects.
        :param edges: An optional iterable of :class:`Edge` tuples to insert.
            Each edge is an ``(EdgeID, source, destination)`` triple.
        :param vertex_property: A zero-argument factory called once per vertex to
            produce its initial property value.  ``None`` means properties default
            to ``None``.
        :param edge_property: A zero-argument factory called once per edge to
            produce its initial property value.  ``None`` means properties default
            to ``None``.
        """
        if isinstance(vertices, int):
            vertices = (Vertex(VertexID(i)) for i in range(vertices))
        if vertex_property is None:
            if TYPE_CHECKING:
                assert VertexProp is NoneType
            vertex_property = _none_property
        if edge_property is None:
            if TYPE_CHECKING:
                assert EdgeProp is NoneType
            edge_property = _none_property

        self.__vertices = {
            v: (VertexID(v_id), vertex_property()) for v_id, v in enumerate(vertices)
        }
        self.__inv_vertices = {v: k for k, (v, _) in self.__vertices.items()}
        self.__storage = storage(len(self.__vertices), _MultiGraphEdgeMonad)
        self.__edges = {}
        for edge in edges or ():
            edge_id, u, v = edge
            self.__storage.update_edge(u, v, edge_id)
            self.__edges[edge_id] = edge_property()

    # Building blocks
    def vertices(self) -> Iterable[Vertex]:
        """Return all vertices in the graph.

        :return: An iterable of every :class:`Vertex` present in the graph.
        """
        return self.__vertices.keys()

    def edges(self, vertex: Vertex | None = None) -> Iterable[Edge]:
        """Return edges in the graph, optionally filtered by source vertex.

        When *vertex* is ``None`` every edge in the graph is returned.
        When a specific vertex is given, only edges originating from that
        vertex are returned.

        :param vertex: If provided, restrict results to edges departing from
            this vertex.  If ``None``, return all edges.
        :return: An iterable of :class:`Edge` tuples ``(EdgeID, source, dest)``.
        """
        raw_edges = (
            self.__storage.outgoing_edges(vertex)
            if vertex is not None
            else self.__storage.edges()
        )
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    # Connectivity
    def incoming_edges(self, vertex: Vertex) -> Iterable[Edge]:
        """Return all edges whose destination is *vertex*.

        :param vertex: The target vertex.
        :return: An iterable of :class:`Edge` tuples arriving at *vertex*.
        """
        raw_edges = self.__storage.incoming_edges(vertex)
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    def outgoing_edges(self, vertex: Vertex) -> Iterable[Edge]:
        """Return all edges whose source is *vertex*.

        :param vertex: The source vertex.
        :return: An iterable of :class:`Edge` tuples departing from *vertex*.
        """
        raw_edges = self.__storage.outgoing_edges(vertex)
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    # Meta-information
    @typing.overload
    def property(self, item: Vertex) -> VertexProp: ...
    @typing.overload
    def property(self, item: Edge) -> EdgeProp: ...

    def property(self, item: Vertex | Edge) -> VertexProp | EdgeProp:
        """Retrieve the property associated with a vertex or edge.

        :param item: A :class:`Vertex` (integer) or :class:`Edge` (tuple) whose
            property should be returned.
        :return: The *VertexProp* value when *item* is a vertex, or the
            *EdgeProp* value when *item* is an edge.
        :raises TypeError: If *item* is neither a vertex nor an edge.
        """
        if isinstance(item, tuple):
            return self.__edges[item[0]]
        if isinstance(item, int):
            return self.__vertices[item][1]
        error_msg = (
            f"Unexpected item of type {type(item)}, should be either Vertex or Edge"
        )
        raise TypeError(error_msg)
