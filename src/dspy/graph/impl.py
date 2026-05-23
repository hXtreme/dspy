"""Concrete graph implementations.

Contains :class:`AsymmetricMultiDiGraph`, an asymmetric directed multi-graph
that allows multiple parallel edges between any ordered pair of vertices,
with optional per-vertex and per-edge properties.

:copyright: 2026-present Harsh Parekh <harsh_parekh@outlook.com>
:license: MIT, see LICENSE.txt for details.
"""

import textwrap
import typing
from collections.abc import Iterable
from types import NoneType
from typing import TYPE_CHECKING, Self

from dspy.graph.types import (
    Edge,
    EdgeID,
    EdgeMonad,
    Graph,
    GraphStorage,
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

    def __init__(self, e: EdgeID | set[EdgeID]) -> None:
        """Initialise the monad with one or more edge identifiers.

        :param e: Either a single :class:`EdgeID` (wrapped into a new
            one-element set) or an existing set of edge identifiers that
            is adopted directly.
        """
        self._e = e if isinstance(e, set) else {e}

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

    def __str__(self) -> str:
        """Return the ``repr`` of the underlying edge-identifier set.

        :return: The Python ``repr`` of the internal ``set[EdgeID]``.
        """
        return repr(self._e)


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

    __vertices: dict[Vertex, VertexID]
    __edges: dict[EdgeID, Edge]
    __inv_vertices: dict[VertexID, Vertex]
    __storage: GraphStorage[_MultiGraphEdgeMonad]
    __vertex_properties: dict[Vertex, VertexProp]
    __edge_properties: dict[EdgeID, EdgeProp]
    __vertex_property_constructor: PropertyConstructor[VertexProp]
    __edge_property_constructor: PropertyConstructor[EdgeProp]

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
            ``Vertex(0)`` through ``Vertex(n-1)``), or an
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
            vertices = (Vertex(i) for i in range(vertices))
        if vertex_property is None:
            if TYPE_CHECKING:
                assert VertexProp is NoneType
            vertex_property = _none_property
        if edge_property is None:
            if TYPE_CHECKING:
                assert EdgeProp is NoneType
            edge_property = _none_property

        self.__vertex_property_constructor = vertex_property
        self.__edge_property_constructor = edge_property

        self.__vertices = {v: VertexID(v_id) for v_id, v in enumerate(vertices)}
        self.__inv_vertices = {v: k for k, v in self.__vertices.items()}
        self.__vertex_properties = {
            v: self.__vertex_property_constructor() for v in self.__vertices
        }

        self.__storage = storage(len(self.__vertices), _MultiGraphEdgeMonad)
        self.__edges: dict[EdgeID, Edge] = {}
        for edge in edges or ():
            self._add_edge(edge)
        self.__edge_properties = {
            e: self.__edge_property_constructor() for e in self.__edges
        }

    def _add_edge(self, edge: Edge) -> None:
        """Validate *edge* and record it in storage and the edge table.

        :param edge: The ``(EdgeID, source, destination)`` triple to add.
        :raises KeyError: If an edge with the same :class:`EdgeID` already
            exists, or if either *source* or *destination* is not a vertex
            of this graph.
        """
        edge_id, u, v = edge
        if existing := self.__edges.get(edge_id):
            error_msg = (
                f"Edge already exists: Faild to add Edge{edge!r}, "
                f"there is already an Edge{existing!r} with the same id"
            )
            raise KeyError(error_msg)
        if u not in self.__vertices:
            error_msg = (
                f"Unknown Vertex: source Vertex({u}) is not present in the graph"
            )
            raise KeyError(error_msg)
        if v not in self.__vertices:
            error_msg = (
                f"Unknown Vertex: destination Vertex({v}) is not present in the graph"
            )
            raise KeyError(error_msg)

        self.__storage.update_edge(self.__vertices[u], self.__vertices[v], edge_id)
        self.__edges[edge_id] = edge

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
            self.__storage.outgoing_edges(self.__vertices[vertex])
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
        raw_edges = self.__storage.incoming_edges(self.__vertices[vertex])
        yield from (
            (e, self.__inv_vertices[u], self.__inv_vertices[v]) for u, v, e in raw_edges
        )

    def outgoing_edges(self, vertex: Vertex) -> Iterable[Edge]:
        """Return all edges whose source is *vertex*.

        :param vertex: The source vertex.
        :return: An iterable of :class:`Edge` tuples departing from *vertex*.
        """
        raw_edges = self.__storage.outgoing_edges(self.__vertices[vertex])
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
        :raises KeyError: If *item* is a vertex or edge that is not part
            of this graph.
        :raises TypeError: If *item* is neither a vertex nor an edge.
        """
        if isinstance(item, tuple):
            return self.__edge_properties[item[0]]
        if isinstance(item, int):
            return self.__vertex_properties[item]
        error_msg = (
            f"Unexpected item of type {type(item)}, should be either Vertex or Edge"
        )
        raise TypeError(error_msg)

    def __str__(self) -> str:
        """Return a multi-line human-readable summary of the graph.

        The output lists each vertex (with its :class:`VertexID` and,
        when present, its property) followed by every edge and the
        backing storage's own ``str`` representation.

        :return: A formatted, indented description of the graph.
        """
        graph_id = id(self)

        compact_vertices = self.__vertex_property_constructor() is None
        vertices_joiner = ", " if compact_vertices else "\n"
        vertices = vertices_joiner.join(
            [
                f"Vertex({u}) <-> VertexID({u_id})"
                if compact_vertices
                else (
                    f"(Vertex({u}) <-> VertexID({u_id}), {self.__vertex_properties[u]})"
                )
                for u, u_id in self.__vertices.items()
            ]
        )
        if not compact_vertices:
            vertices = "\n" + textwrap.indent(vertices, "    ") + "\n"

        compact_edges = self.__edge_property_constructor() is None
        edges_joiner = ", " if compact_edges else "\n"
        edges = edges_joiner.join(
            [
                str((f"EdgeID({edge})", prop) if prop else f"EdgeID({edge})")
                for edge, prop in self.__edges.items()
            ]
        )
        if not compact_edges:
            edges = "\n" + textwrap.indent(edges, "        ")

        storage = textwrap.indent(str(self.__storage), "    ")
        return textwrap.dedent(
            """
            Graph (id={graph_id})
                Vertices: {vertices}
                Edges: {edges}
            {storage}
            """
        ).format(
            graph_id=graph_id,
            vertices=vertices,
            edges=edges,
            storage=storage,
        )
